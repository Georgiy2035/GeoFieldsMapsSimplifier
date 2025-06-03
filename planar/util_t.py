import numpy as np
from numba import njit
from numba import vectorize, float64, int32, guvectorize
from util.util import getNeighboursNB_arr, indexNB, getdifxy, lengs_xyz, calc_cord_norms, line_grads_calc_mx, gen_params, multi_mx_on_vec_per_row, calc_sum_mx, next_neighbor_smoother_mx
NANINT = 2147483646


# @njit
# def getxy(c, lst, lptr, n):
#     if n == 0:
#         return np.array([c[ptr] if ptr != NANINT else np.nan for ptr in lst])
#     else:
#         return np.array([c[lst[ptr]] if ptr != NANINT else np.nan for ptr in lptr])
    

def calc_graphic_hardness_NB(t_p, areas, iso_list, scale, width):
    gh = np.zeros(t_p.trs.shape[0])
    res_iso_vecs = np.array([[0., 0.]])
    for i, tr in enumerate(t_p.trs):
        iso_side_points_all = np.array([[0., 0.]])
        iso_side_all = np.array([0.])
        iso_side_num = np.array([0, 0, 0])
        
        for j, p in enumerate(tr):
            if t_p.values[p] > t_p.values[tr[j-1]]:
                iso_side_bool = np.logical_and((iso_list < t_p.values[p]), (iso_list > t_p.values[tr[j-1]]))
                iso_side = iso_list[iso_side_bool]
                p0 = np.array([t_p.x[tr[j-1]], t_p.y[tr[j-1]], t_p.values[tr[j-1]]])
                p1 = np.array([t_p.x[p], t_p.y[p], t_p.values[p]])    
            elif t_p.values[p] < t_p.values[tr[j-1]]:
                iso_side_bool = np.logical_and(iso_list > t_p.values[p], iso_list < t_p.values[tr[j-1]])
                iso_side = iso_list[iso_side_bool]
                p1 = np.array([t_p.x[tr[j-1]], t_p.y[tr[j-1]], t_p.values[tr[j-1]]])
                p0 = np.array([t_p.x[p], t_p.y[p], t_p.values[p]])
            else:
                continue
            # if True in iso_side_bool:
            #     print(iso_side)
            # print(tr)
            vec = p1 - p0
            #print(iso_side_all)
            iso_side_num[j] = iso_side.shape[0]
            if iso_side_num[j] == 0:
                continue
            iso_start_ps = np.array(list(zip((iso_side - p0[2]) * vec[0] / vec[2] + p0[0],
                                            (iso_side - p0[2]) * vec[1] / vec[2] + p0[1])))
            iso_side_points_all = np.concatenate((iso_side_points_all, iso_start_ps))
            iso_side_all = np.concatenate((iso_side_all, iso_side))
            
            

        if iso_side_num[0] == 0 and iso_side_num[1] == 0:
            gh[i] = 0
            continue
        
        iso_side_all = iso_side_all[1:]
        iso_side0 = iso_side_all[0:iso_side_num[0]]
        iso_side1 = iso_side_all[iso_side_num[0]:iso_side_num[1]]
        iso_side2 = iso_side_all[iso_side_num[1]:]

        iso_side_points_all = iso_side_points_all[1:]
        iso_side_points0 = iso_side_points_all[0:iso_side_num[0]]
        iso_side_points1 = iso_side_points_all[iso_side_num[0]:iso_side_num[1]]
        iso_side_points2 = iso_side_points_all[iso_side_num[1]:]
        for j, iso in enumerate(iso_side0):
            if iso in iso_side1:
                ind_iso = indexNB(iso_side1, iso)
                res_iso_vecs = np.concatenate((res_iso_vecs, (iso_side_points0[j] - iso_side_points1[ind_iso]).reshape(1, 2)))
            elif iso in iso_side2:
                ind_iso = indexNB(iso_side2, iso)
                res_iso_vecs = np.concatenate((res_iso_vecs, (iso_side_points0[j] - iso_side_points2[ind_iso]).reshape(1, 2)))
        for j, iso in enumerate(iso_side1):
            if iso in iso_side2:
                ind_iso = indexNB(iso_side2, iso)
                res_iso_vecs = np.concatenate((res_iso_vecs, (iso_side_points1[j] - iso_side_points2[ind_iso]).reshape(1, 2))) 

        iso_lengs = lengs_xy(res_iso_vecs[:, 0], res_iso_vecs[:, 1])
        lengs_sum = np.sum(iso_lengs)
        square = lengs_sum * width / 1000 / scale
        gh[i] = square / areas[i]
    return gh





@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def vec_calc_dists(x0, y0, x1, y1):
    return np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

# def calculate_distances_NB_mx(x, y, nbr_mx):
#     xdif = getdifxy(x, nbr_mx)
#     ydif = getdifxy(y, nbr_mx)
#     res = np.zeros(nbr_mx.shape)
#     lengths_mx(xdif, ydif, res)
    
#     return res[1:]#np.sqrt(xdif**2 + ydif**2)




@njit
def calculate_gradients_NB(t_p, g_p:gen_params, values, areas, dists_in_t):
    grad = np.zeros((values.shape[0], 3))
    for i in range(values.shape[0]):
        if np.isnan(values[i]):
            for j in range(3):
                grad[i][j] = np.nan
    dists_in_t2 = dists_in_t.T**2
    dists_in_t2_forv = np.array(list(zip(dists_in_t2[:,0] * dists_in_t2[:,1], dists_in_t2[:,1] * dists_in_t2[:,2], dists_in_t2[:,0] * dists_in_t2[:,2])))
    
    p0x = t_p.x[t_p.trs[:, 0]]
    p0y = t_p.y[t_p.trs[:, 0]]
    p0v = values[t_p.trs[:, 0]]
    p1x = t_p.x[t_p.trs[:, 1]]
    p1y = t_p.y[t_p.trs[:, 1]]
    p1v = values[t_p.trs[:, 1]]
    p2x = t_p.x[t_p.trs[:, 2]]
    p2y = t_p.y[t_p.trs[:, 2]]
    p2v = values[t_p.trs[:, 2]]
    longx = calc_cord_norms(p1y, p0y, p2v, p0v, p2y, p1v)
    longy = calc_cord_norms(p1v, p0v, p2x, p0x, p2v, p1x)
    longz = calc_cord_norms(p1x, p0x, p2y, p0y, p2x, p1y)
    leng = lengs_xyz(longx, longy, longz)
    norms = np.array(list(zip(longx / leng, longy / leng, longz / leng)))

    vdif = getdifxy(values, g_p.nbr_mx)
    line_grads = np.zeros(g_p.nbr_mx.shape)
    line_grads_calc_mx(vdif, g_p.nbr_dists, line_grads)

    for i in range(norms.shape[0]):
        grad[t_p.trs[i]] -= np.array([list(norms[i])]*3) * areas[i] / np.array([list(dists_in_t2_forv[i])]*3).T

    lengs = lengs_xyz(grad[:, 0], grad[:, 1], grad[:, 2])
    lengs = np.array([leng if leng != 0 else 1 for leng in lengs])            
    pnt_grads = np.array(list(zip(grad[:, 0] / lengs, grad[:, 1] / lengs)))

    return pnt_grads, line_grads

@vectorize([float64(float64, float64)], nopython=True)
def lengs_xy(x, y):
    return np.sqrt((x**2) + (y**2))



@njit
def idw_smoother_NB(t_p, depth: int, strength, g_p):
    #lengths = sorted([e.lin.length for e in self.edges])
    #maxl = max(lengths)
    #minl = lengths[int(len(lengths)/1000)]
    #norml = maxl - minl
    new_v = np.zeros(t_p.x.shape[0])           
    for i in range(t_p.x.shape[0]):
        if np.isnan(t_p.x[i]):
            new_v[i] = np.nan
        else:
            #nbrs_set=next_neighbor_smoother(i, np.array([i]), 0, depth, t_p)
            nbrs_set=next_neighbor_smoother_mx(i, np.array([i]), 0, depth, g_p.nbr_mx)
            nbrs = np.array(list(nbrs_set))
            nbrs_x = g_p.nbr_x[nbrs]
            nbrs_y = g_p.nbr_y[nbrs]
            nbrs_v = t_p.values[nbrs]
            dists = lengs_xy(nbrs_x, nbrs_y)
            ws = np.ones(nbrs_x.shape[0]) / dists
            w_nbrs_v = nbrs_v * ws
            new_v[i] = t_p.values[i] * (1-strength) + np.sum(w_nbrs_v) / sum(ws) * strength
    return new_v

@njit
def nbr_smoother_NB(t_p, depth: int, strength, g_p):
    #lengths = sorted([e.lin.length for e in self.edges])
    #maxl = max(lengths)
    #minl = lengths[int(len(lengths)/1000)]
    #norml = maxl - minl
    new_v = np.zeros(t_p.x.shape[0])           
    for i in range(t_p.x.shape[0]):
        if np.isnan(t_p.x[i]):
            new_v[i] = np.nan
        else:
            #nbrs_set=next_neighbor_smoother(i, np.array([i]), 0, depth, t_p)
            nbrs_set=next_neighbor_smoother_mx(i, np.array([i]), 0, depth, g_p.nbr_mx)
            nbrs = np.array(list(nbrs_set))
            nbrs_x = t_p.x[nbrs] - t_p.x[i]
            nbrs_y = t_p.y[nbrs] - t_p.y[i]
            nbrs_v = t_p.values[nbrs]
            #dists = lengs_xy(nbrs_x, nbrs_y)
            ws = np.ones(nbrs_x.shape[0])
            w_nbrs_v = nbrs_v * ws
            new_v[i] = t_p.values[i] * (1-strength) + np.sum(w_nbrs_v) / sum(ws) * strength
    return new_v



@njit
def calculate_non_simb_ws_NB(t_p, g_p:gen_params, nbrs, vor_centers, vor_centers_nbrs):
    vor_edgeslen_nbrs_mx = np.zeros(g_p.nbr_mx.shape)
    for i, tr in enumerate(t_p.trs):
        for j, p in enumerate(tr):
            ind1 = indexNB(g_p.nbr_mx[p], tr[j-2])
            ind2 = indexNB(g_p.nbr_mx[p], tr[j-1])
            if vor_edgeslen_nbrs_mx[p][ind1] == 0:
                edge1 = vor_centers[i] - vor_centers_nbrs[i][j-1] if nbrs[i][j-1] != -1 else vor_centers[i] - (
                    np.array([t_p.x[p], t_p.y[p]]) + np.array([t_p.x[tr[j-2]], t_p.y[tr[j-2]]])) / 2
                vor_edgeslen_nbrs_mx[p][ind1] = np.sqrt(edge1[0]**2+edge1[1]**2)
            if vor_edgeslen_nbrs_mx[p][ind2] == 0:
                edge2 = vor_centers[i] - vor_centers_nbrs[i][j-2] if nbrs[i][j-2] != -1 else vor_centers[i] - (
                    np.array([t_p.x[p], t_p.y[p]]) + np.array([t_p.x[tr[j-1]], t_p.y[tr[j-1]]])) / 2
                vor_edgeslen_nbrs_mx[p][ind2] = np.sqrt(edge2[0]**2+edge2[1]**2)

    non_simb_ws = vor_edgeslen_nbrs_mx / g_p.nbr_dists
    ws_sum = calc_sum_mx(non_simb_ws)
    w_parts = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(non_simb_ws, 1 / ws_sum, w_parts)
    return w_parts
