import numba
import numpy as np
import scipy.sparse as sp
from sklearn import neighbors
import random

from kneed import KneeLocator

@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())


@numba.njit(cache=True)
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
    js = []
    vals = []
    for i, node in enumerate(nodes):
        j, val = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        js.append(j)
        vals.append(val)
    return js, vals

@numba.njit(cache=True)
def three_hop_neighbourhood(node, indptr, indices):

    hop = set()

    for v_node_1 in indices[indptr[node]:indptr[node + 1]]:
        hop.add(v_node_1)
        for v_node_2 in indices[indptr[v_node_1]:indptr[v_node_1 + 1]]:
            hop.add(v_node_2)
            for v_node_3 in indices[indptr[v_node_2]:indptr[v_node_2 + 1]]:
                hop.add(v_node_3)

    hop_np = np.array(list(hop))
    hop_np = hop_np.astype(np.int64)
    return list(hop_np)

@numba.njit(cache=True)
def filter_mask(arr, threshold):
    return arr[arr > threshold]

# @numba.njit(cache=True)
def get_kn(x, y, S):
    kn = KneeLocator(x, y, curve='convex', direction='decreasing', S=S) 
    return kn.knee + 1 #Recover ignored element
    

# @numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):

    
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)


    #EXPERIMENT 6

    all_kn = 0
    for i in numba.prange(len(nodes)):


        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
   
        #Improved Exp2 - with range between [half k, k + half k]
        half_k = int(topk/2)
        k_values = np.arange(half_k, topk + half_k)
        random_k  = np.random.choice(k_values)

        if i < 5:
            print('random_k: ', random_k)


        j_np, val_np = np.array(j), np.array(val)

        idx_topk = np.argsort(val_np)[-random_k:]

        # x = np.arange(0, len(val) - 1)  #Size is len of val minus largest element
        # idx_y = np.argsort(val_np)[::-1]  #Sort in descending order
        # y = val_np[idx_y]
        # y = y[1:]    #ignore largest element (root node)
        # kn = get_kn(x, y, 8)
        # all_kn += kn
        # idx_topk = idx_y[0:kn]

        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    
    print('Mean kn: ', int(all_kn/len(nodes)))
    return js, vals


def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk):
    """Calculate the PPR matrix approximately using Anderson."""

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk)

    
    return construct_sparse(neighbors, weights, (len(nodes), nnodes))


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk, normalization='row'):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    topk_matrix = ppr_topk(adj_matrix, alpha, eps, idx, topk).tocsr()


    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_inv = 1. / np.maximum(deg, 1e-12)

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

    return topk_matrix
