from distutils import core
from math import gamma
import numba
import numpy as np
import scipy.sparse as sp
import igraph
from sklearn import neighbors
from elbow_point import get_elbow_point


from scipy.signal import savgol_filter
from kneed import KneeLocator

from networkx import from_scipy_sparse_matrix, k_truss


@numba.njit(cache=True, locals={'_val': numba.float32, 'sum_cr':numba.float32, 'percentage':numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, CR, core_numbers, indices, indptr,  deg, alpha, epsilon):

    # nodes[i], CR, core_numbers, indices, indptr, deg, alpha, epsilon

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

        CR_neigbours = [core_numbers[vnode] for vnode in indices[indptr[unode]:indptr[unode + 1]]]

        sum_cr = add_elements(CR_neigbours)
        print(sum_cr)

        for vnode in indices[indptr[unode]:indptr[unode + 1]]:

            # percentage = core_numbers[vnode]/ sum_cr


            _val = (1 - alpha) * res /deg[unode]


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
def add_elements(list):
    array = np.array(list)
    return np.sum(array)

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

@numba.njit(cache=True, locals={'_val': numba.float32, 'percentage': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def get_nodes(node, CR, core_numbers, indices, indptr, deg, alpha, epsilon):

    alpha_eps = alpha * epsilon

    # max_core = np.max(CR)
    # alpha = core_numbers[node]/np.max(core_numbers)  
    # print('max core: ', max_core)
    # print('current CR: ', CR[node])

    f32_0 = numba.float32(0)
    p = {node: f32_0}
    r = {}
    r[node] = alpha

    q = [node]

    i =0

    #Check with neigbours   
    while len(q): 

        current_node = q.pop()

        res = r[current_node] if current_node in r else f32_0
        if current_node in p:
            p[current_node] += res
        else:
            p[current_node] = res

        r[current_node] = f32_0

        neighbours = np.array([vnode for vnode in indices[indptr[current_node]:indptr[current_node + 1]]])

        CR_neighbours = core_numbers[neighbours]

        if i ==0:
            print('CR_neighbours: ', CR_neighbours)

        # count_deg = len(np.where(CR_neighbours >= CR[current_node])[0])
        
        for vnode in indices[indptr[current_node]:indptr[current_node + 1]]:
            # if CR[vnode] >= CR[current_node]:

            percentage = core_numbers[vnode] / np.sum(CR_neighbours)
            
            # print('gets here even though count_deg is ',count_deg)
            _val = (1 - alpha) * res * percentage

            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val
            
            res_vnode = r[vnode] if vnode in r else f32_0

            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)
                
        i +=1
    
    return list(p.keys()), list(p.values())



    


    

# @numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk, core_numbers, graph, truss, S=None, gamma=0.1):

    
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)


    all_kn = 0
    len_y = []

    CR = np.zeros((len(indptr) -1))
    for i in range(len(indptr) -1):

        #CRE method
        neighbours_cores =  [core_numbers[n_v] for n_v in indices[indptr[i]:indptr[i + 1]]]

        CR[i] = sum(neighbours_cores)
    
    #sort CR in decresing order
    idx_decreasing_CR = np.argsort(CR)[::-1]
    n_best = get_elbow_point(CR[idx_decreasing_CR])
    # n_best = 4

    # print('CRE n_best: ', n_best)
    #set of key nodes
    idx_key_nodes = idx_decreasing_CR[:n_best]
    # print('idx_key_nodes: ', idx_key_nodes.shape)
    # print('indptr - 1 ', len(indptr) -1)



    for i in numba.prange(len(nodes)):

        # j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j, val = _calc_ppr_node(nodes[i], CR, core_numbers, indices, indptr, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)


        #Normalize pageRank values
        val_np = val_np / np.sum(val_np)

        #For statistics (min, max, mean) purposes
        len_y.append(len(val))


        #BASELINE--------
        idx_topk = np.argsort(val_np)[-topk:]

        # if i ==0:
        #     print('nodes with page rank: ', j_np[idx_topk].tolist())
        # all_kn += topk
        js[i] = j_np
        vals[i] = val_np


        # js[i] = j_np[idx_topk]
        # vals[i] = val_np[idx_topk]

        continue
        


        #----------------

        #if len < 3 --> TAKE ALL

        if len(val) <= 3:
            idx_topk = np.argsort(val_np)
            all_kn += len(val)
            js[i] = j_np[idx_topk]
            vals[i] = val_np[idx_topk]
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
        

        kn = kn.knee +1 # + 1 to recover ignored first element

        if i < 5:
            print('kn: ', kn)

        all_kn += kn

        idx_topk = idx_y[0:kn]


        #----------------
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]

    js_weighted = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals_weighted = [np.zeros(0, dtype=np.float32)] * len(nodes)

    
    

    for i in numba.prange(len(nodes)):

        #Analuyze ppr

        # CR_ppr = CR[js[i]]
        # core_numbers = core_numbers[js[i]]
        # shortest_paths = graph.get_shortest_paths(nodes[i], to=js[i])
        # shortest_paths = np.array([len(s_path) for s_path in shortest_paths])


        # if i ==0:
        #     print('ppr: ', vals[i].tolist())
        #     print('CR_ppr:', CR_ppr.tolist())
        #     print('core_numbers: ', core_numbers.tolist())
        #     print('shortest_paths: ', shortest_paths.tolist())


        # #K-cores of shortest paths
        # cores = CR[idx_key_nodes[idx_sort_shortest_paths]]

        # #Normalize core values
        # cores = cores/np.sum(cores)

        # closest_nodes = idx_key_nodes[idx_sort_shortest_paths]

        # #Normalize short paths
        # norm_shortest_paths = np.sum(shortest_paths) - shortest_paths[idx_sort_shortest_paths]
        # norm_shortest_paths = norm_shortest_paths/np.sum(norm_shortest_paths)
        # if i ==0:
        #     print('norm_shortest_paths: ', norm_shortest_paths)



        # new_exp = (gamma*cores) + ((1-gamma)*norm_shortest_paths)

        # #Find elbow point
        # idx_topk = np.argsort(new_exp)[-topk:]

        # if i ==0:
        #     print('this cr: ', np.sort(cores)[::-1].tolist())

        # elbow = get_elbow_point(new_exp[idx_topk[1:]]) + 1


        # all_kn += topk

        # new_exp = new_exp[idx_topk]/np.sum(new_exp[idx_topk])


        # js_weighted[i] = idx_key_nodes[idx_sort_shortest_paths][idx_topk]
        # vals_weighted[i] = weighted_cores[idx_topk]/np.sum(weighted_cores[idx_topk])

        # continue

        
        # CR_decreasing = CR[idx_decreasing_CR]

        # current_core_idx = np.where(idx_decreasing_CR == nodes[i])[0][0]

        # interval = 20

        # # if i ==0:
        # #     print(current_core_idx)

        # if current_core_idx < interval:
        #     returned_nodes = idx_decreasing_CR[:current_core_idx +interval + 1 ]
        # else:
        #     returned_nodes = idx_decreasing_CR[current_core_idx - interval:current_core_idx +interval + 1 ]
        
        # if i ==0:
        #     print('returned_nodes: ', returned_nodes.shape, returned_nodes)
        #     print(CR[returned_nodes])
        


        j, val =  _calc_ppr_node(nodes[i], CR, core_numbers, indices, indptr, deg, alpha, epsilon)

        j_np, val_np = np.array(j), np.array(val)

        
        val_np = val_np / np.sum(val_np)

        cores = core_numbers[j_np]
        cores = cores / np.sum(cores)

        new_idx = (gamma* val_np) + ((1-gamma) * cores)

        

        idx_topk = np.argsort(new_idx)[::-1]

        #Get elbow point
        elbow = get_elbow_point(new_idx[idx_topk[1:]]) + 1


        idx_topk = np.argsort(new_idx)[-elbow:]

        # if i == 0:
        #     print('For node: ', i)
        #     print('j_np: ', j_np[idx_topk].tolist())
        #     print('val_np: ', val_np[idx_topk].tolist())
        #     print('idx_key_nodes: ', idx_key_nodes.tolist())

        all_kn += idx_topk.shape[0]

        js_weighted[i] = j_np[idx_topk]
        vals_weighted[i] = new_idx[idx_topk]

        continue



        #TOTALLY NEW EXPERIMENT:
        if i == 0:
            print('Node: ', nodes[i])
            print('with core number: ', CR[i].tolist())
            print('nodes related: ', returned_nodes.tolist())
            these_cores = [CR[i] for i in returned_nodes]
            print('with cores: ', these_cores)
            print('where max is : ', np.max(these_cores))

        
        # Get all CRs for this node
        current_cr = CR[returned_nodes]

        current_cr = current_cr/np.sum(current_cr)

         #Get distance from this node to all other key nodes
        shortest_paths = graph.get_shortest_paths(nodes[i], to=returned_nodes)
        shortest_paths = np.array([len(s_path) for s_path in shortest_paths])

        #sort them (increasing)
        idx_crs = np.argsort(shortest_paths)

         #Normalize short paths
        norm_shortest_paths = np.sum(shortest_paths) - shortest_paths[idx_crs]
        norm_shortest_paths = norm_shortest_paths/np.sum(norm_shortest_paths)



        new_exp = (gamma*current_cr[idx_crs]) + ((1-gamma)*norm_shortest_paths)

        idx_topk = np.argsort(new_exp)[-topk:]

        # n_best = get_elbow_point(new_exp[idx_crs][1:]) + 1

        # current_cr = current_cr/ np.sum(current_cr)


        all_kn += idx_topk.shape[0]

        js_weighted[i] = returned_nodes[idx_crs][idx_topk]
        vals_weighted[i] = current_cr[idx_crs][idx_topk]
        

    global mean_kn 
    mean_kn = all_kn/len(nodes)
    print('Mean kn: ', mean_kn)
    print('gamma: ', gamma)
    print('Overall len y: ', (sum(len_y)/len(len_y)), 'max: ', max(len_y), ' min: ', min(len_y))
    # return js_weighted, vals_weighted
    return js_weighted, vals_weighted


def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk, core_numbers, graph, S=None, gamma=0.1):
    """Calculate the PPR matrix approximately using Anderson."""

    g_networkX = from_scipy_sparse_matrix(adj_matrix)
    truss = k_truss(g_networkX, 3)


    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk, core_numbers, graph, truss, S=S, gamma=gamma)

    
    return construct_sparse(neighbors, weights, (len(nodes), nnodes))


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk, core_numbers, graph, normalization='row', S=None, gamma=0.1):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    topk_matrix = ppr_topk(adj_matrix, alpha, eps, idx, topk, core_numbers, graph, S=S, gamma=gamma).tocsr()


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
