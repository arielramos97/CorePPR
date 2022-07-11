import time
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow.compat.v1 as tf
import scipy.sparse as sp


from .utils import sparse_feeder
from .tf_utils import mixed_dropout


class PPRGo:
    def __init__(self, d, nc, hidden_size, nlayers, lr,
                 weight_decay, dropout, adj_matrix, intermediate_layer, sparse_features=True):
        self.nc = nc
        self.sparse_features = sparse_features

        if sparse_features:
            self.batch_feats = tf.sparse_placeholder(tf.float32, None, 'features')
            self.batch_feats_adj = tf.sparse_placeholder(tf.float32, None, 'features')
        else:
            self.batch_feats = tf.placeholder(tf.float32, [None, d], 'features')
            self.batch_feats_adj = tf.placeholder(tf.float32, [None, d], 'features')
        self.batch_pprw = tf.placeholder(tf.float32, [None], 'ppr_weights')
        self.batch_idx = tf.placeholder(tf.int32, [None], 'idx')
        self.batch_labels = tf.placeholder(tf.int32, [None], 'labels')

        self.adj_idx = tf.placeholder(tf.int32, [None], 'adj_idx')
        self.batch_adj_matrix = tf.placeholder(tf.float32, [None], 'adj_matrix')

        Ws = [tf.get_variable('W1', [d, hidden_size])]
        for i in range(nlayers - 2):
            Ws.append(tf.get_variable(f'W{i + 2}', [hidden_size, hidden_size]))
        Ws.append(tf.get_variable(f'W{nlayers}', [hidden_size, nc]))

        feats_drop = mixed_dropout(self.batch_feats, dropout)
        if sparse_features:
            h = tf.sparse.sparse_dense_matmul(feats_drop, Ws[0])
        else:
            h = tf.matmul(feats_drop, Ws[0])
        for W in Ws[1:]:
            h = tf.nn.relu(h)
            h_drop = tf.nn.dropout(h, rate=dropout)
            h = tf.matmul(h_drop, W)
        self.logits = h


        weighted_logits = tf.tensor_scatter_nd_add(tf.zeros((tf.shape(self.batch_labels)[0], nc)),
                                                   self.batch_idx[:, None],
                                                   self.logits * self.batch_pprw[:, None])

        # Add linear layer
        # W = tf.get_variable(f'W{nlayers+1}', [nc, nc])
        # Ws.append(W)
        # h = tf.matmul(weighted_logits, W)
        

        # feats_drop_adj= mixed_dropout(self.batch_feats_adj, dropout)
        # if sparse_features:
        #     h = tf.sparse.sparse_dense_matmul(feats_drop_adj, Ws[0])
        # else:
        #     h = tf.matmul(feats_drop_adj, Ws[0])
        # for W in Ws[1:]:
        #     h = tf.nn.relu(h)
        #     h_drop = tf.nn.dropout(h, rate=dropout)
        #     h = tf.matmul(h_drop, W)
        # self.logits_adj = h

        
        # w_1 = w_0 * self.batch_adj_matrix[:, None]
        # self.w_1 = w_1

        # adjacency_weighted_logits = tf.tensor_scatter_nd_add(weighted_logits, 
        #                                             self.adj_idx[:, None], 
        #                                             self.logits_adj * self.batch_adj_matrix[:, None])

        #Final layer
        # W = tf.get_variable(f'W{nlayers+1}', [intermediate_layer, nc])
        # Ws.append(W)
        # final_weights = tf.matmul(adjacency_weighted_logits, W)





        loss_per_node = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_labels,
                                                                       logits=weighted_logits)

        l2_reg = tf.add_n([tf.nn.l2_loss(weight) for weight in Ws])
        self.loss = tf.reduce_mean(loss_per_node) + weight_decay * l2_reg

        self.preds = tf.argmax(weighted_logits, 1)
        self.update_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

        self.cached = {}

    def feed_for_batch(self, attr_matrix, ppr_matrix, labels, adj_matrix, key=None):
        if key is None:
            return self.gen_feed(attr_matrix, ppr_matrix, labels, adj_matrix)
        else:
            if key in self.cached:
                return self.cached[key]
            else:
                feed = self.gen_feed(attr_matrix, ppr_matrix, labels, adj_matrix)
                self.cached[key] = feed
                return feed

    def gen_feed(self, attr_matrix, ppr_matrix, labels, adj_matrix):
        source_idx, neighbor_idx = ppr_matrix.nonzero()

        source_adj, neighbor_adj = adj_matrix.nonzero()

        batch_attr = attr_matrix[neighbor_idx]

        batch_attr_adj = attr_matrix[neighbor_adj]

        # print('source_idx: ', source_idx)
        # print('adj_matrix: ', adj_matrix.shape)
        # print('source_adj: ', source_adj.shape)
        # print('neighbor_adj: ', neighbor_adj.shape)
        # print('attr_matrix: ', attr_matrix.n_rows, attr_matrix.n_columns)

        # print('batch_feats: ', attr_matrix[neighbor_idx].shape)
        # print('batch_pprw: ', ppr_matrix[source_idx, neighbor_idx].A1.shape)
        # print('batch_idx: ', source_idx.shape)
        

        feed = {
            self.batch_feats: sparse_feeder(batch_attr) if self.sparse_features else batch_attr,
            self.batch_feats_adj: sparse_feeder(batch_attr_adj) if self.sparse_features else batch_attr_adj,
            self.batch_pprw: ppr_matrix[source_idx, neighbor_idx].A1,
            self.batch_labels: labels,
            self.batch_adj_matrix : adj_matrix[source_adj, neighbor_adj].A1,
            self.adj_idx: source_adj,
            self.batch_idx: source_idx,
        }
        return feed

    def _get_logits(self, sess, attr_matrix, nnodes, batch_size_logits=10000):
        logits = []
        for i in range(0, nnodes, batch_size_logits):
            batch_attr = attr_matrix[i:i + batch_size_logits]
            logits.append(sess.run(self.logits,
                                   {self.batch_feats: sparse_feeder(batch_attr) if self.sparse_features else batch_attr}
                                   ))
        logits = np.row_stack(logits)
        return logits

    def predict(self, sess, adj_matrix, attr_matrix, alpha,
                nprop=2, inf_fraction=1.0, ppr_normalization='sym', batch_size_logits=10000):

        start = time.time()
        if inf_fraction < 1.0:
            idx_sub = np.random.choice(adj_matrix.shape[0], int(inf_fraction * adj_matrix.shape[0]), replace=False)
            idx_sub.sort()
            attr_sub = attr_matrix[idx_sub]
            logits_sub = self._get_logits(sess, attr_sub, idx_sub.shape[0], batch_size_logits)
            local_logits = np.zeros([adj_matrix.shape[0], logits_sub.shape[1]], dtype=np.float32)
            local_logits[idx_sub] = logits_sub
        else:
            local_logits = self._get_logits(sess, attr_matrix, adj_matrix.shape[0], batch_size_logits)
        time_logits = time.time() - start

        start = time.time()
        row, col = adj_matrix.nonzero()
        logits = local_logits.copy()

        if ppr_normalization == 'sym':
            # Assume undirected (symmetric) adjacency matrix
            deg = adj_matrix.sum(1).A1
            deg_sqrt_inv = 1. / np.sqrt(np.maximum(deg, 1e-12))
            for _ in range(nprop):
                logits = (1 - alpha) * deg_sqrt_inv[:, None] * (adj_matrix @ (deg_sqrt_inv[:, None] * logits)) + alpha * local_logits
        elif ppr_normalization == 'col':
            deg_col = adj_matrix.sum(0).A1
            deg_col_inv = 1. / np.maximum(deg_col, 1e-12)
            for _ in range(nprop):
                logits = (1 - alpha) * (adj_matrix @ (deg_col_inv[:, None] * logits)) + alpha * local_logits
        elif ppr_normalization == 'row':
            deg_row = adj_matrix.sum(1).A1
            deg_row_inv_alpha = (1 - alpha) / np.maximum(deg_row, 1e-12)
            for _ in range(nprop):
                logits = deg_row_inv_alpha[:, None] * (adj_matrix @ logits) + alpha * local_logits
        else:
            raise ValueError(f"Unknown PPR normalization: {ppr_normalization}")
        predictions = logits.argmax(1)
        time_propagation = time.time() - start

        return predictions, time_logits, time_propagation

    def get_vars(self, sess):
        return sess.run(tf.trainable_variables())

    def set_vars(self, sess, new_vars):
        set_all = [
                var.assign(new_vars[i])
                for i, var in enumerate(tf.trainable_variables())]
        sess.run(set_all)

#Normalize adjacency matrix
def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr

def train(sess, model, attr_matrix, train_idx, val_idx, topk_train, topk_val, labels, adj_matrix,
          max_epochs=200, batch_size=512, batch_mult_val=4,
          eval_step=1, early_stop=False, patience=50, ex=None):

    normalized_adj_matrix = calc_A_hat(adj_matrix)

    step = 0
    best_loss = np.inf

    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}
    f1_hist = {'train': [], 'val': []}
    if ex is not None:
        ex.current_run.info['train'] = {'loss': [], 'acc': []}
        ex.current_run.info['val'] = {'loss': [], 'acc': []}

    for epoch in range(max_epochs):
        for i in range(0, len(train_idx), batch_size):
            feed_train = model.feed_for_batch(attr_matrix,
                                              topk_train[i:i + batch_size],
                                              labels[train_idx[i:i + batch_size]],
                                              normalized_adj_matrix[train_idx[i:i + batch_size]],
                                              key=i)

            _, train_loss, preds = sess.run([model.update_op, model.loss, model.preds],
                                             feed_train)

            step += 1
            if step % eval_step == 0:

                # update train stats
                train_acc = accuracy_score(labels[train_idx[i:i + batch_size]], preds)
                train_f1 = f1_score(labels[train_idx[i:i + batch_size]], preds, average='macro')

                loss_hist['train'].append(train_loss)
                acc_hist['train'].append(train_acc)
                f1_hist['train'].append(train_f1)
                if ex is not None:
                    ex.current_run.info['train']['loss'].append(train_loss)
                    ex.current_run.info['train']['acc'].append(train_acc)
                    ex.current_run.info['train']['f1'].append(train_f1)

                if topk_val is not None:

                    # update val stats
                    rnd_val = np.random.permutation(len(val_idx))[:batch_mult_val*batch_size]
                    feed_val = model.feed_for_batch(attr_matrix, topk_val[rnd_val],
                                                    labels[val_idx[rnd_val]])
                    val_loss, preds = sess.run([model.loss, model.preds], feed_val)

                    val_acc = accuracy_score(labels[val_idx[rnd_val]], preds)
                    val_f1 = f1_score(labels[val_idx[rnd_val]], preds, average='macro')

                    loss_hist['val'].append(val_loss)
                    acc_hist['val'].append(val_acc)
                    f1_hist['val'].append(val_f1)
                    if ex is not None:
                        ex.current_run.info['val']['loss'].append(val_loss)
                        ex.current_run.info['val']['acc'].append(val_acc)
                        ex.current_run.info['val']['f1'].append(val_f1)

                    logging.info(f"Epoch {epoch}, step {step}: train {train_loss:.5f}, val {val_loss:.5f}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_epoch = epoch
                        best_trainables = model.get_vars(sess)
                    # early stop only if this variable is set to True
                    elif early_stop and epoch >= best_epoch + patience:
                        model.set_vars(sess, best_trainables)
                        return epoch + 1, loss_hist, acc_hist, f1_hist
                else:
                    logging.info(f"Epoch {epoch}, step {step}: train {train_loss:.5f}")
    if topk_val is not None:
        model.set_vars(sess, best_trainables)
    return epoch + 1, loss_hist, acc_hist, f1_hist