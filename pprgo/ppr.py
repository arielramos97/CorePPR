from statistics import mean
import numba
import numpy as np
import scipy.sparse as sp
import math
# from elbow_point import get_elbow_point

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

@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node2(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)

    r = np.zeros((len(indptr)-1), dtype=np.float32)
    p = np.zeros((len(indptr)-1), dtype=np.float32)

    p[inode] =f32_0
    r[inode] = alpha
    q = [inode]

    while len(q) > 0:
        unode = q.pop()

        res = r[unode] 
        p[unode] += res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            r[vnode] += _val
            
            res_vnode = r[vnode] 
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)
    
    j = np.nonzero(p)[0]
    val = p[j]

    return j, val

@numba.njit(cache=True)
def get_elbow_point(sorted_scores):

    if (len(sorted_scores)<2):
        elbow_point = 1
        return elbow_point
    else:  

        first_point = np.zeros((2), dtype=np.float32)
        first_point[0] = 0
        first_point[1] = sorted_scores[0]

        last_point = np.zeros((2), dtype=np.float32)
        last_point[0] = len(sorted_scores)-1
        last_point[1] = sorted_scores[-1]


        k =1
        distances = np.zeros((len(sorted_scores)), dtype=np.float32)

        for i in numba.prange(len(sorted_scores)):
            numerator = abs((last_point[1] - first_point[1])*k - (last_point[0] - first_point[0])*sorted_scores[i] + last_point[0]*first_point[1] - last_point[1]*first_point[0])
            denominator = (pow((last_point[1] - first_point[1]),2) + pow(pow((last_point[0] - first_point[0]),2),0.5))
            distances[i] = numerator/denominator
            k = k + 1

        elbow_point = np.argmax(distances)
        return elbow_point


@numba.njit(cache=True)
def k_core(indptr, indices, deg):


    nodes = np.argsort(deg)

    bin_boundaries = [0]
    curr_degree = 0

    # for i, v in enumerate(nodes):
    #     if deg[v] > curr_degree:
    #         bin_boundaries.extend([i] * (deg[v] - curr_degree))
    #         curr_degree = deg[v]
    
    for i in numba.prange(len(nodes)):
        if deg[nodes[i]] > curr_degree:
            bin_boundaries.extend([i] * (deg[nodes[i]] - curr_degree))
            curr_degree = deg[nodes[i]]

    node_pos = np.zeros((len(indptr)-1), dtype=np.int64)

    for i in numba.prange(len(nodes)):
        node_pos[nodes[i]] = i 

    core = deg.copy()

    nbrs = []
    for i in numba.prange(len(indptr)-1):
        nbrs.append(list(indices[indptr[i]:indptr[i + 1]]))
    # nbrs = {v: list(indices[indptr[v]:indptr[v + 1]]) for v in nodes}

    # printing = int(len(nodes) / 10)

    for i in numba.prange(len(nodes)):
        # if i % printing ==0:
        #     print(i, ' nodes processed')

        for j in numba.prange(len(nbrs[nodes[i]])):
            
            if core[nbrs[nodes[i]][j]] > core[nodes[i]]:
                nbrs[nbrs[nodes[i]][j]].remove(nodes[i])
                pos = node_pos[nbrs[nodes[i]][j]]
                bin_start = bin_boundaries[core[nbrs[nodes[i]][j]]]
                node_pos[nbrs[nodes[i]][j]] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[nbrs[nodes[i]][j]]] += 1
                core[nbrs[nodes[i]][j]] -= 1
    return core
   

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
def coreRank(indptr, indices, cores):

    CR = np.zeros((len(indptr)-1), dtype=np.float32)

    for i in numba.prange(len(indptr) -1):
        CR[i] = np.sum(cores[indices[indptr[i]:indptr[i + 1]]], dtype=np.float32)
    
    return CR


@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk, CR, elbow):

    sum_k = 0
    # len_y = []

    js = [np.zeros(0, dtype=np.int64)] * (len(nodes))
    pprs = [np.zeros(0, dtype=np.float32)] * (len(nodes))
    coreRanks = [np.zeros(0, dtype=np.float32)] * (len(nodes))
   
    for i in numba.prange(len(nodes)):
        
        j, ppr = _calc_ppr_node2(nodes[i], indptr, indices, deg, alpha, epsilon)


        #For statistics (min, max, mean) purposes
        # len_y.append(len(val_ppr_np))

        if elbow == True:
            idx_topk = np.argsort(ppr) 
            elbow_point = get_elbow_point(ppr[idx_topk[:-1][::-1]]) + 1  #Ignore last because it contains most of the probability mass, then include count + 1
            idx_topk = idx_topk[-elbow_point:]
        
        else:

            #Take only inital 32 if topk provided, else take them all
            if topk is None:
                idx_topk = np.argsort(ppr)
            else:
                idx_topk = np.argsort(ppr)[-topk:]
            
        
        sum_k += idx_topk.shape[0]

        js[i] = j[idx_topk]
        pprs[i] = ppr[idx_topk] /np.sum(ppr[idx_topk])  #Normalization

        coreRanks[i] = CR[j[idx_topk]] / np.sum(CR[j[idx_topk]])

    # global mean_kn 
    mean_k = sum_k/len(nodes)
    print('Mean k: ', mean_k)
    # print('Overall len y: ', (sum(len_y)/len(len_y)), 'max: ', max(len_y), ' min: ', min(len_y))
    return js, pprs, coreRanks, mean_k


def ppr_topk(adj_matrix, alpha, epsilon, nodes, data_file, topk, core_numbers, elbow):
    """Calculate the PPR matrix approximately using Anderson."""

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    if core_numbers is None:
        core_numbers = k_core(adj_matrix.indptr, adj_matrix.indices, out_degree)
        np.save('coredata/'+data_file[5:-4]+'-cores', core_numbers)

    
    CR = coreRank(adj_matrix.indptr, adj_matrix.indices, core_numbers)

    neighbors, weights, core_weights, mean_k = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk, CR, elbow)

    
    return construct_sparse(neighbors, weights, (len(nodes), nnodes)), construct_sparse(neighbors, core_weights, (len(nodes), nnodes)), mean_k


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(adj_matrix, alpha, eps, idx, data_file, topk, core_numbers, normalization='row', elbow=False):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    topk_matrix, core_topk_matrix, mean_k = ppr_topk(adj_matrix, alpha, eps, idx, data_file, topk, core_numbers, elbow)

    topk_matrix, core_topk_matrix = topk_matrix.tocsr(), core_topk_matrix.tocsr()

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
        core_topk_matrix.data = deg_sqrt[idx[row]] * core_topk_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_inv = 1. / np.maximum(deg, 1e-12)

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
        core_topk_matrix.data = deg[idx[row]] * core_topk_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

   
    return topk_matrix, core_topk_matrix, mean_k
