import numba
import numpy as np
import scipy.sparse as sp
import math
from elbow_point import get_elbow_point

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
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk, CR):

    
    # print('CR: ', np.max(CR))
    # js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    # vals = [np.zeros(0, dtype=np.float32)] * len(nodes)


    # all_kn = numba.float32(0)
    # len_y = []



    js_weighted = [np.zeros(0, dtype=np.int64)] * (len(nodes))
    vals_weighted = [np.zeros(0, dtype=np.float32)] * (len(nodes))

    vals_core_weighted = [np.zeros(0, dtype=np.float32)] * (len(nodes))
   
    for i in numba.prange(len(nodes)):

        # if i % 100 ==0:
        #     print(i)

        
        # j_ppr, val_ppr = powerPush(nodes[i], indptr, indices, deg, alpha, lambda_, all_nodes, m, W)
        # p = powerPush2(nodes[i], indptr, indices, deg, alpha, lambda_, all_nodes, m, W, epoch_num, scanThreshold)

        
        j_ppr_np, val_ppr_np = _calc_ppr_node2(nodes[i], indptr, indices, deg, alpha, epsilon)


        #For statistics (min, max, mean) purposes
        # len_y.append(len(val_ppr_np))

        

        # if i ==0:
        #     print('len: ', len(j_ppr_np))
        #     print('sum: ', np.sum(val_ppr_np))
        #     print('j: ', j_ppr_np)
        #     print('val: ', val_ppr_np)


        # j_ppr_np, val_ppr_np = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        # j_ppr_np, val_ppr_np = np.array(j_ppr_np), np.array(val_ppr_np)
        


        #Take only inital 32 if topk provide, else take them all
        if topk is None:
            idx_topk = np.argsort(val_ppr_np)
        else:
            idx_topk = np.argsort(val_ppr_np)[-topk:]
            

        # val_ppr_np_topk = val_ppr_np[idx_topk] /np.sum(val_ppr_np[idx_topk])


        # cores = 
        # if i ==0:
        #     print('cores: ', cores)
        # if i ==0:
        #     print('cores: ', cores)

        # if i ==0:
        #     print('type cores: ', cores.dtype)

        # val_ppr_np_topk = val_ppr_np_topk * cores

        # new_idx = (gamma* val_ppr_np_topk) + ((1-gamma) * cores)
 

        

        # shortest_paths = np.array([np.array(shortest_path(graph, nodes[i], k)) for k in key_nodes])
        # length_paths = np.array([path.size for path in shortest_paths])
        # print('length_paths: ', length_paths)

        # #Take the first top k
        # idx_shortest_nodes = np.argsort(length_paths)[:topk]
        # closest_paths = shortest_paths[idx_shortest_nodes]
        # closest_nodes = key_nodes[idx_shortest_nodes]
        # closest_lenghts = length_paths[idx_shortest_nodes]

        # print('closest_paths: ', closest_paths.shape)
        # print('closest_nodes: ', closest_nodes.shape, closest_nodes)

        

        # print(shortest_paths.shape)

        # j_core, val_core =  _calc_core_node(nodes[i], core_numbers, CR,  indices, indptr, deg, alpha, epsilon, key_nodes)
    
        

        # j_core_np, val_core_np = np.array(j_core), np.array(val_core)

        # val_ppr_np = val_ppr_np/ np.sum(val_ppr_np)

        # val_ppr_np2 = val_ppr_np2/ np.sum(val_ppr_np2)

        # idx_topk = np.argsort(new_idx)[::-1]  #decreasing order

        # # #Get elbow point
        # elbow = get_elbow_point(new_idx[idx_topk[1:]]) + 1


        # idx_topk32 = np.argsort(new_idx)[-elbow:]

       
        # all_kn += elbow

        # idx_topk = np.argsort(val_ppr_np)[:]
        
        # idx_topk32 = np.argsort(new_idx)

        # all_kn += idx_topk32.shape[0]

        # if i ==0:
        #     print('sum top k: ', np.sum(val_ppr_np[idx_topk32]))

        # sum_rest = sum_all - np.sum(val_ppr_np[idx_topk32])

        # val_ppr_np = val_ppr_np[idx_topk32] + ((val_ppr_np[idx_topk32]/np.sum(val_ppr_np[idx_topk32]))*sum_rest)

        # if i ==0:
        #     print('j_ppr: ', j_ppr_np[idx_topk32])
            # print('new_idx: ', new_idx[idx_topk32])
            # print('val_ppr: ', val_ppr_np[idx_topk32])
            # print('sum final: ', np.sum(val_ppr_np))

        js_weighted[i] = j_ppr_np[idx_topk]
        vals_weighted[i] = val_ppr_np[idx_topk] /np.sum(val_ppr_np[idx_topk])

        vals_core_weighted[i] = CR[j_ppr_np[idx_topk]] / np.sum(CR[j_ppr_np[idx_topk]])

        # continue        

    # global mean_kn 
    # mean_kn = all_kn/len(nodes)
    # print('Mean kn: ', mean_kn)
    # print('gamma: ', gamma)
    # print('Overall len y: ', (sum(len_y)/len(len_y)), 'max: ', max(len_y), ' min: ', min(len_y))
    # return js_weighted, vals_weighted
    return js_weighted, vals_weighted, vals_core_weighted


def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk, core_numbers):
    """Calculate the PPR matrix approximately using Anderson."""

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    if core_numbers is None:
        core_numbers = k_core(adj_matrix.indptr, adj_matrix.indices, out_degree)
        np.save('pubmed-cores', core_numbers)
    
    CR = coreRank(adj_matrix.indptr, adj_matrix.indices, core_numbers)

    neighbors, weights, core_weights = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk, CR)

    
    return construct_sparse(neighbors, weights, (len(nodes), nnodes)), construct_sparse(neighbors, core_weights, (len(nodes), nnodes)), CR


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk, core_numbers, normalization='row'):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    topk_matrix, core_topk_matrix, coreRank = ppr_topk(adj_matrix, alpha, eps, idx, topk, core_numbers)

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

   
    return topk_matrix, core_topk_matrix, coreRank, 0
