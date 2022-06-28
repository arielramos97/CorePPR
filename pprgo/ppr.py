from math import gamma
import numba
import numpy as np
import scipy.sparse as sp
import igraph


from scipy.signal import savgol_filter
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
    return hop_np

@numba.njit(cache=True)
def filter_mask(arr, threshold):
    return arr[arr > threshold]

# @numba.njit(cache=True)
def get_kn(x, y, S=1):
    kn = KneeLocator(x, y, curve='convex', direction='decreasing', S=S) 
    return kn.knee 
    

# @numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk, core_numbers, S=None, gamma=0.1):

    
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)


    all_kn = 0
    truncated_S = 0
    len_y = []

    all_nodes = np.max(indices) + 1
    ppr_general = np.zeros(all_nodes, dtype=np.float32)

    for i in numba.prange(len(nodes)):

        # #Get core value of this nodes
        # k_core = core_numbers[hop_3]
        # if i ==0:
        #     print('k_core: ', k_core.tolist())

        # #Normalize values
        # k_core_normalised = k_core / np.sum(k_core)

        # if i ==0:
        #     print('k_core_normalised: ', k_core_normalised.tolist())

        

        # #sort them
        # idx_topk = np.argsort(k_core_normalised)[-topk:]
        # if i ==0:
        #     print('sorted k core: ', k_core[idx_topk].tolist())
        # vals[i] = k_core[idx_topk]


        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)


        #For statistics (min, max, mean) purposes
        len_y.append(len(val))


        #BASELINE--------
        idx_topk = np.argsort(val_np)[-topk:]
        all_kn += topk
        js[i] = j_np
        vals[i] = val_np


        # js[i] = j_np[idx_topk]
        # vals[i] = val_np[idx_topk]

        ppr_general[j_np] += val_np

        continue


        #----------------

        #if len < 3 --> TAKE ALL

        if len(val) <= 3:
            idx_topk = np.argsort(val_np)
            all_kn += len(val)
            js[i] = j_np[idx_topk]
            vals[i] = val_np[idx_topk]
            if i < 5:
                print('kn: ', len(val))
            continue

        #Ignore first entry (largest)
        ignore = 1
        x = np.arange(0, len(val) - ignore) 
        idx_y = np.argsort(val_np)[::-1]  #Sort in descending order
        y = val_np[idx_y]
        y = y[ignore:]    #ignore largest element (root node)

        #Else compute the knee point
        kn = KneeLocator(x, y, curve='convex', direction='decreasing', S=S, interp_method='polynomial')
        
        #If no knee point --> TAKE ALL
        if kn.knee is None or kn.knee ==0:
            idx_topk = np.argsort(val_np)
            all_kn += len(val)
            js[i] = j_np[idx_topk]
            vals[i] = val_np[idx_topk]
            if i < 5:
                print('kn: ', len(val))
            continue

        #If there is ACTUALLY a knee point
        if i < 5:
            print('kn: ', kn.knee)

        kn = kn.knee +1 # + 1 to recover ignored first element

        all_kn += kn

        idx_topk = idx_y[0:kn]


        #----------------
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    
    #Collect general ppr (average of all personalised ppr values)
    ppr_general = ppr_general/len(nodes)

    js_weighted = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals_weighted = [np.zeros(0, dtype=np.float32)] * len(nodes)

    for i in range(len(nodes)):
        left = vals[i]


        #First sort pageRank values
        idx_topk = np.argsort(vals[i])[-topk:]

        j_ranked = js[i][idx_topk]
        val_ranked = vals[i][idx_topk]

        core = core_numbers[j_ranked]

        if i == 0:
            print('j_ranked: ', j_ranked)
            print('core of j_ranked: ', core)


        #Rank by core
        idx_topk = np.argsort(core)[-topk:]

        #Take the pageRank values in theres indeces

        core = core / sum(core)

        val_ranked = val_ranked + core

        # right = 
        # right = 
        # right = ppr_general[js[i]]

        
    #     right = ppr_general[js[i]]

        # new_exp = (gamma*left) + (1-gamma)*right
        # new_exp = left - right
        # new_exp = new_exp/np.sum(new_exp)

        # new_exp = (gamma*left) + (1-gamma)*core

        # idx_topk = np.argsort(new_exp)[-topk:]

        js_weighted[i] = j_ranked[idx_topk]
        vals_weighted[i] = val_ranked[idx_topk]

        if i ==0:
            print('js[i]: ', j_ranked[idx_topk])
            print('idx: ', idx_topk)
            print('core top: ', core[idx_topk])


    global mean_kn 
    mean_kn = all_kn/len(nodes)
    print('Mean kn: ', mean_kn)
    print('Overall len y: ', (sum(len_y)/len(len_y)), 'max: ', max(len_y), ' min: ', min(len_y))
    print('Truncated windows: ', truncated_S, ' over ', len(nodes), ' nodes')
    # return js_weighted, vals_weighted
    return js_weighted, vals_weighted


def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk, core_numbers, S=None, gamma=0.1):
    """Calculate the PPR matrix approximately using Anderson."""

    # print('ppr_topk adj matrix: ', adj_matrix.shape)

    # #Build grpah using igraph
    # g = igraph.Graph.Adjacency((adj_matrix.todense()> 0).tolist())
    # core_numbers = np.array(g.coreness())
    # print('core_numbers: ', core_numbers.shape)
    # max_c_n = np.max(core_numbers)
    # print('max core: ', max_c_n)

    # idx = np.where(core_numbers==max_c_n)

    # print('idx: ', idx)
    # print('core_numbers[idx]', core_numbers[idx].shape, core_numbers[idx])

    #Get coreness

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk, core_numbers, S=S, gamma=gamma)

    
    return construct_sparse(neighbors, weights, (len(nodes), nnodes))


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk, core_numbers, normalization='row', S=None, gamma=0.1):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    topk_matrix = ppr_topk(adj_matrix, alpha, eps, idx, topk, core_numbers, S=S, gamma=gamma).tocsr()


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

    return topk_matrix, mean_kn
