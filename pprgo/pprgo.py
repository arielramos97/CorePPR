from distutils import core
import time
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow.compat.v1 as tf

from .utils import sparse_feeder
from .tf_utils import mixed_dropout


import scipy.sparse as sp


class PPRGo:
    def __init__(self, d, nc, hidden_size, nlayers, lr, gamma,
                 weight_decay, dropout, sparse_features=True):
        self.nc = nc
        self.sparse_features = sparse_features

        if sparse_features:
            self.batch_feats = tf.sparse_placeholder(tf.float32, None, 'features')
        else:
            self.batch_feats = tf.placeholder(tf.float32, [None, d], 'features')
        self.batch_pprw = tf.placeholder(tf.float32, [None], 'ppr_weights')
        self.batch_core = tf.placeholder(tf.float32, [None], 'core_weights')
        self.batch_idx = tf.placeholder(tf.int32, [None], 'idx')
        self.batch_labels = tf.placeholder(tf.int32, [None], 'labels')
        self.batch_size = tf.placeholder(tf.int32, None, 'batch_size')

        self.gamma = tf.get_variable('gamma', dtype=tf.float32, initializer=gamma, trainable=True)
        # self.gamma = tf.get_variable('gamma', [2,1], trainable=True) #, constraint=lambda x: tf.clip_by_value(x, 0, 1))

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

        # self.gamma = self.gamma / tf.reduce_sum(self.gamma)
        self.gamma = tf.math.sigmoid(self.gamma)

        # self.gamma = tf.nn.softmax(self.gamma)


        self.core_ppr = ((1-self.gamma)*self.batch_pprw) + ((self.gamma) * self.batch_core)


        self.weighted_logits = tf.tensor_scatter_nd_add(tf.zeros((self.batch_size, nc)),
                                                   self.batch_idx[:, None],
                                                   self.logits * self.core_ppr[:, None])

        loss_per_node = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_labels,
                                                                       logits=self.weighted_logits)

        l2_reg = tf.add_n([tf.nn.l2_loss(weight) for weight in Ws])
        self.loss = tf.reduce_mean(loss_per_node) + weight_decay * l2_reg

        self.preds = tf.argmax(self.weighted_logits, 1)
        self.update_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

        self.cached = {}

    def feed_for_batch(self, attr_matrix, ppr_matrix, core_matrix, labels, key=None):
        if key is None:
            return self.gen_feed(attr_matrix, ppr_matrix, core_matrix, labels)
        else:
            if key in self.cached:
                return self.cached[key]
            else:
                feed = self.gen_feed(attr_matrix, ppr_matrix, core_matrix, labels)
                self.cached[key] = feed
                return feed

    def gen_feed(self, attr_matrix, ppr_matrix, core_matrix, labels):
        source_idx, neighbor_idx = ppr_matrix.nonzero()

        # print('source_idx shape: ', source_idx.shape)
        # print('labels shape: ', labels.shape)
        print('ppr shape: ', ppr_matrix.shape )
        print('type: ', type(ppr_matrix.shape[0]))

        batch_attr = attr_matrix[neighbor_idx]
        feed = {
            self.batch_feats: sparse_feeder(batch_attr) if self.sparse_features else batch_attr,
            self.batch_pprw: ppr_matrix[source_idx, neighbor_idx].A1,
            self.batch_core: core_matrix[source_idx, neighbor_idx].A1,
            self.batch_labels: labels,
            self.batch_idx: source_idx,
            self.batch_size: ppr_matrix.shape[0],
        }
        return feed

    def _get_logits(self, sess, attr_matrix, nnodes, ppr_matrix, core_matrix, batch_size_logits=10000):
        logits = []
        final_logits = []

        for i in range(0, nnodes, batch_size_logits):

            current_ppr_matrix = ppr_matrix[i:i + batch_size_logits]
            current_core_matrix = core_matrix[i:i + batch_size_logits]

            source_idx, neighbor_idx = current_ppr_matrix.nonzero()
            batch_attr = attr_matrix[neighbor_idx]

            current_logits, gamma, weighted_logits = sess.run([self.logits, self.gamma, self.weighted_logits],
                                    {self.batch_feats: sparse_feeder(batch_attr) if self.sparse_features else batch_attr,
                                        self.batch_pprw: current_ppr_matrix[source_idx, neighbor_idx].A1,
                                        self.batch_core: current_core_matrix[source_idx, neighbor_idx].A1,
                                        self.batch_idx: source_idx,
                                        self.batch_size: current_ppr_matrix.shape[0],
                                    }
                                    )

            logits.append(current_logits)
            final_logits.append(weighted_logits)
            
        logits = np.row_stack(logits)
        final_logits = np.row_stack(final_logits)
        return logits, gamma, final_logits



        

        for i in range(0, nnodes, batch_size_logits):

            current_ppr_matrix = ppr_matrix[i:i + batch_size_logits]
            current_core_matrix = core_matrix[i:i + batch_size_logits]

            print('current_ppr_matrix: ', current_ppr_matrix.shape)
            print('current_core_matrix: ', current_core_matrix.shape)
            
            source_idx, neighbor_idx = current_ppr_matrix.nonzero()

            print('batch_attr: ', attr_matrix.shape)

            batch_attr = attr_matrix[i:i + batch_size_logits]

            print('batch_attr: ', batch_attr.shape)

            current_logits, gamma, weighted_logits = sess.run([self.logits, self.gamma, self.weighted_logits],
                                   {self.batch_feats: sparse_feeder(batch_attr) if self.sparse_features else batch_attr,
                                    self.batch_pprw: current_ppr_matrix[source_idx, neighbor_idx].A1,
                                    self.batch_core: current_core_matrix[source_idx, neighbor_idx].A1,
                                    self.batch_idx: source_idx,
                                    self.batch_size: batch_size_logits,
                                   }
                                   )
            # current_logits = None
            # weighted_logits = None
            # gamma = None

            logits.append(current_logits)
            final_logits.append(weighted_logits)
        logits = np.row_stack(logits)
        final_logits = np.row_stack(final_logits)
        return logits, gamma, final_logits

    def predict(self, sess, adj_matrix, attr_matrix, alpha, ppr_topk_test, core_topk_test,
                nprop=2, inf_fraction=1.0, ppr_normalization='sym', batch_size_logits=10000):

        start = time.time()
        if inf_fraction < 1.0:
            idx_sub = np.random.choice(adj_matrix.shape[0], int(inf_fraction * adj_matrix.shape[0]), replace=False)
            idx_sub.sort()
            attr_sub = attr_matrix[idx_sub]
            logits_sub = self._get_logits(sess, attr_sub, idx_sub.shape[0], ppr_topk_test, core_topk_test, batch_size_logits)
            local_logits = np.zeros([adj_matrix.shape[0], logits_sub.shape[1]], dtype=np.float32)
            local_logits[idx_sub] = logits_sub
        else:
            local_logits, gamma, weighted_logits = self._get_logits(sess, attr_matrix, ppr_topk_test.shape[0], ppr_topk_test, core_topk_test, batch_size_logits)
        time_logits = time.time() - start

        print('Inference gamma: ', gamma)
        start = time.time()
        predictions = weighted_logits.argmax(1)
        time_propagation = time.time() - start

        return predictions, time_logits, time_propagation, weighted_logits





        row, col = adj_matrix.nonzero()
        logits = local_logits.copy()

        if ppr_normalization == 'sym':
            # Assume undirected (symmetric) adjacency matrix
            deg = adj_matrix.sum(1).A1
            deg_sqrt_inv = 1. / np.sqrt(np.maximum(deg, 1e-12))

            # coreRank_matrix = (adj_matrix).multiply(coreRank)
            # normalized_core_matrix = coreRank_matrix.multiply(1/coreRank_matrix.sum(axis=1).A1[:, None])

            for _ in range(nprop):
                logits = (1 - alpha) * deg_sqrt_inv[:, None] * (adj_matrix @ (deg_sqrt_inv[:, None] * logits)) + alpha * local_logits

            
            # right_term = sp.eye(adj_matrix.shape[0])
            # adj_power = adj_matrix.multiply(deg_sqrt_inv[:, None])


            # for k in range(1, nprop):
            #     if k ==1:
            #         right_term += (1-alpha)  * adj_power
            #     else:
            #         adj_power = adj_power @ adj_power
            #         right_term += np.power(1-alpha, k) * adj_power

            # right_term = alpha * right_term

            # adj_power = adj_power @ adj_power

            # left_term = np.power(1-alpha, nprop) * adj_power

            # new_logits = (left_term +right_term) @  logits


            
            #Perform partition
            # idx_logits = np.argpartition(logits, 32, axis =1)
            # print('idx_logits: ', idx_logits.shape)
            
            # coreRank_matrix = (adj_matrix).multiply(coreRank)
            # normalized_core_matrix = coreRank_matrix.multiply(1/coreRank_matrix.sum(axis=1).A1[:, None])

            # # print(normalized_core_matrix.sum(axis=1))

            # logits_core = deg_sqrt_inv[:, None] * (normalized_core_matrix @ (deg_sqrt_inv[:, None] * local_logits))

            # logits = ((1 -gamma) * logits) + (gamma * logits_core)

            # print('coreRank_matrix: ', coreRank_matrix.shape)
            print('local_logits: ', local_logits.shape)
            print('logits: ', logits.shape)
            # print('logits_core: ', logits_core.shape)

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

        return predictions, time_logits, time_propagation, logits

    def get_vars(self, sess):
        return sess.run(tf.trainable_variables())

    def set_vars(self, sess, new_vars):
        set_all = [
                var.assign(new_vars[i])
                for i, var in enumerate(tf.trainable_variables())]
        sess.run(set_all)


def train(sess, model, attr_matrix, train_idx, val_idx, topk_train, topk_val, core_topk_train, labels,
          max_epochs=200, batch_size=512, batch_mult_val=4,
          eval_step=1, early_stop=False, patience=50, ex=None):
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
                                              core_topk_train[i:i + batch_size],
                                              labels[train_idx[i:i + batch_size]],
                                              key=i)

            gamma, core_ppr, _, train_loss, preds = sess.run([model.gamma, model.core_ppr, model.update_op, model.loss, model.preds],
                                             feed_train)

            # if epoch % 20 == 0:
                # print('core_ppr: ', core_ppr.shape, core_ppr)
                # print('gamma: ', gamma)

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