import stripy as stp
import numpy as np
from pyproj import Transformer
from shapely import MultiPolygon, Polygon
from typing import Union
from util.util import dif_mx_and_vec_per_col, lengs_xyz, calc_cord_norms, line_grads_calc_mx, indexNB, getdifxy, get_nbr_val, multi_mx_on_vec_per_row, gen_params, calc_sum_mx, calc_sum_mx0, next_neighbor_smoother_mx
from numba import vectorize, float64, njit, guvectorize, objmode
NANINT = 2147483646


@njit(fastmath=True)
def getX_NB(bound):
    sumlon = 0
    flagmer = 0
    lattoR = []
    lattoL = []
    for i in range(len(bound) - 1):
        delta = bound[i+1][0] - bound[i][0]
        if delta < -(3.14159 + 3):
            delta += 2 * 3.14159
            flagmer += 1
            lattoR.append((bound[i][1] + bound[i+1][1]) / 2)
            lattoL.append(0)
        elif delta > 3.14159 + 3:
            delta -= 2 * 3.14159
            flagmer += 1
            lattoR.append(0)
            lattoL.append((bound[i][1] + bound[i+1][1]) / 2)
        sumlon += delta
    if (3.14159 * 2 - 0.1 < sumlon < 3.14159 * 2 + 0.1) and (flagmer % 2 == 1):
        return (0, 3.14159 / 2)
    elif flagmer > 0:
        nummer = len(lattoR) // 2
        lat1 = lattoR[nummer - 1] + lattoL[nummer - 1]
        lat2 = lattoR[nummer] + lattoL[nummer]
        return (3.14159, (lat1 + lat2) / 2)
    else:
        return (-100, -100)

def getX(bound):
    X = getX_NB(bound)
    if X == (-100, -100):
        lonc, latc = Polygon(bound).point_on_surface().xy
        return np.array((lonc[0], latc[0]))
    else:
        return np.array(X)
    


@njit(fastmath=True)
def getTranLon(p, q):
    t = np.sin(q[0] - p[0]) * np.cos(q[1])
    b = np.sin(q[1]) * np.cos(p[1]) - np.cos(q[1]) * np.sin(p[1]) * np.cos(q[0] - p[0])
    return -np.arctan2(t, b)

@njit(fastmath=True)
def eastOrWest(lon_c, lon_d):
    delta = lon_d - lon_c
    delta = delta - 3.14159 * 2 if delta > 3.14159 else delta + 3.14159 * 2 if delta < -3.14159 else delta
    res = -1 if delta > 0 and delta != 3.14159 else 1 if delta < 0 and delta != -3.14159 else 0
    return res

@njit(cache=True)
def containsXY_NB(p, x, bound, tlon_v):
    tlon_p = getTranLon(x, p)
    cross = 0
    for i in range(bound.shape[0] - 1):
        a = bound[i]
        tlon_a = tlon_v[i]
        b = bound[i+1]
        tlon_b = tlon_v[i+1]
        
        strike = 0
        if tlon_p == tlon_a:
            strike = 1 
        else:
            ab = eastOrWest(tlon_a, tlon_b)
            ap = eastOrWest(tlon_a, tlon_p)
            pb = eastOrWest(tlon_p, tlon_b)
            if ap==ab and pb == ab:
                strike = 1
        
        if strike == 1:
            if a[0] == p[0] and a[1] == p[1]:
                return 0
            tlon_xnew = getTranLon(a, x)
            tlon_bnew = getTranLon(a, b)
            tlon_pnew = getTranLon(a, p)
            if tlon_pnew == tlon_bnew:
                return 0
            bx = eastOrWest(tlon_bnew, tlon_xnew)
            bp = eastOrWest(tlon_bnew, tlon_pnew)
            if bx == -bp:
                cross += 1

    if cross % 2 == 0:
        return 1 
    return 0

@njit(cache=True)
def containsXY_NB_array(ps, x, bound, ext=False):
    res = np.zeros(ps.shape[0], dtype=np.int32)
    tlon_v = np.array([getTranLon(x, v) for v in bound])

    if not ext:
        bll = np.swapaxes(bound, 0, 1)
        params = [x[1] != 3.14159 / 2 and x[0] != 3.14159,
                    np.min(bll[0]), np.max(bll[0]),
                    np.min(bll[1]), np.max(bll[1])]
    
        for i in range(ps.shape[0]):
            if params[0]:
                if not((params[1] < ps[i][0] < params[2]) and (params[3] < ps[i][1] < params[4])):
                    continue
            res[i] = containsXY_NB(ps[i], x, bound, tlon_v)
    else:
        for i in range(ps.shape[0]):
            res[i] = containsXY_NB(ps[i], x, bound, tlon_v)
    
    return res

def containsXY_mp_array(ps, mp: MultiPolygon, transformer: Union[Transformer, None]):
    res_in = np.zeros(ps.shape[0], dtype=np.int32)
    for g in mp.geoms:
        ext = g.exterior
        extx, exty = ext.xy
        if type(transformer) != type(None):
            extll = list(transformer.itransform(list(zip(extx, exty)), radians=True))
            extll = np.array(list(map(lambda x: (x[1], x[0]), extll)))
        else: 
            extll = np.array(list(zip(np.array(extx) / 180 * 3.14159, np.array(exty) / 180 * 3.14159)))
        ext_x = getX(extll)
        ext_in = containsXY_NB_array(ps, ext_x, extll, ext=True)
        inters_in = np.zeros(ps.shape[0], dtype=np.int32)

        for inter in g.interiors:          
            interx, intery = inter.xy
            if type(transformer) != type(None):
                interll = list(transformer.itransform(list(zip(interx, intery)), radians=True))
                interll = np.array(list(map(lambda x: (x[1], x[0]), interll)))     
            else: 
                interll = np.array(list(zip(np.array(interx) / 180 * 3.14159, np.array(intery) / 180 * 3.14159)))
            curr_x = getX(interll) 
            curr_inter_in = containsXY_NB_array(ps, curr_x, interll, ext=False)

            inters_in = np.bitwise_or(curr_inter_in, inters_in)

        first_res_in = np.bitwise_xor(ext_in, inters_in)
        res_in = np.bitwise_or(first_res_in, res_in)
    return res_in


@njit
def getAzimXY(q, a, lens):
    dl = np.zeros((a.shape[0], a.shape[1]))
    dif_mx_and_vec_per_col(a[:, :, 0], q[:, 0], dl)
    sin_lens = np.sin(lens)
    cos_lens = np.cos(lens)
    sin_delta = np.sin(-dl)*np.cos(a[:, :, 1])/sin_lens

    coslens_multi_sinlatsq = np.zeros((a.shape[0], a.shape[1]))
    multi_mx_on_vec_per_row(cos_lens, np.sin(q[:, 1]), coslens_multi_sinlatsq)
    sinlens_multi_coslatsq = np.zeros((a.shape[0], a.shape[1]))
    multi_mx_on_vec_per_row(sin_lens, np.cos(q[:, 1]), sinlens_multi_coslatsq)

    cos_delta = (np.sin(a[:, :, 1]) - coslens_multi_sinlatsq) / sinlens_multi_coslatsq
    x = -lens * sin_delta
    y = lens * cos_delta
    return x, y




@njit
def calculate_gradients_NB(st_p, g_p:gen_params, values, areas, dists_in_t):
    grad = np.zeros((values.shape[0], 3))
    for i in range(values.shape[0]):
        if np.isnan(values[i]):
            for j in range(3):
                grad[i][j] = np.nan
    dists_in_t2 = dists_in_t.T**2
    dists_in_t2_forv = np.array(list(zip(dists_in_t2[:,0] * dists_in_t2[:,1], dists_in_t2[:,1] * dists_in_t2[:,2], dists_in_t2[:,0] * dists_in_t2[:,2])))

    vdif = getdifxy(values, g_p.nbr_mx)
    line_grads = np.zeros(g_p.nbr_mx.shape)
    line_grads_calc_mx(vdif, g_p.nbr_dists, line_grads)

    values_mx = get_nbr_val(values, g_p.nbr_mx)
    for i in range(st_p.trs.shape[0]):
        tr = st_p.trs[i]
        i1in_p0, i0in_p1, i0in_p2 = indexNB(g_p.nbr_mx[tr[0]], tr[1]), indexNB(g_p.nbr_mx[tr[1]], tr[0]), indexNB(g_p.nbr_mx[tr[2]], tr[0])
        i2in_p0, i2in_p1, i1in_p2 = indexNB(g_p.nbr_mx[tr[0]], tr[2]), indexNB(g_p.nbr_mx[tr[1]], tr[2]), indexNB(g_p.nbr_mx[tr[2]], tr[1])
        p0x = np.array([0,                         g_p.nbr_x[tr[1]][i0in_p1], g_p.nbr_x[tr[2]][i0in_p2]])
        p0y = np.array([0,                         g_p.nbr_y[tr[1]][i0in_p1], g_p.nbr_y[tr[2]][i0in_p2]])
        p0v = np.array([values[tr[0]],             values[tr[0]],             values[tr[0]]            ])
        p1x = np.array([g_p.nbr_x[tr[0]][i1in_p0], 0,                         g_p.nbr_x[tr[2]][i1in_p2]])
        p1y = np.array([g_p.nbr_y[tr[0]][i1in_p0], 0,                         g_p.nbr_y[tr[2]][i1in_p2]])
        p1v = np.array([values[tr[1]],             values[tr[1]],             values[tr[1]]            ])
        p2x = np.array([g_p.nbr_x[tr[0]][i2in_p0], g_p.nbr_x[tr[1]][i2in_p1], 0                        ])
        p2y = np.array([g_p.nbr_y[tr[0]][i2in_p0], g_p.nbr_y[tr[1]][i2in_p1], 0                        ])
        p2v = np.array([values[tr[2]],             values[tr[2]],             values[tr[2]]            ])
        longx = calc_cord_norms(p1y, p0y, p2v, p0v, p2y, p1v)
        longy = calc_cord_norms(p1v, p0v, p2x, p0x, p2v, p1x)
        longz = calc_cord_norms(p1x, p0x, p2y, p0y, p2x, p1y)
        leng = lengs_xyz(longx, longy, longz)
        norms = np.array(list(zip(longx / leng, longy / leng, longz / leng)))

        grad[st_p.trs[i]] -= norms * areas[i] / np.array([list(dists_in_t2_forv[i])]*3).T

    lengs = lengs_xyz(grad[:, 0], grad[:, 1], grad[:, 2])
    lengs = np.array([leng if leng != 0 else 1 for leng in lengs])            
    pnt_grads = np.array(list(zip(grad[:, 0] / lengs, grad[:, 1] / lengs)))

    return pnt_grads, line_grads

@njit
def idw_smoother_NB(st_p, depth: int, strength, nbr_mx):
    #lengths = sorted([e.lin.length for e in self.edges])
    #maxl = max(lengths)
    #minl = lengths[int(len(lengths)/1000)]
    #norml = maxl - minl
    new_v = np.zeros(st_p.lons.shape[0])           
    for i in range(st_p.lons.shape[0]):
        if np.isnan(st_p.lons[i]):
            new_v[i] = np.nan
        else:
            nbrs_set=next_neighbor_smoother_mx(i, np.array([i]), 0, depth, nbr_mx)
            nbrs = np.array(list(nbrs_set))
            # print(nbrs)
            lons1 = st_p.lons[nbrs]
            lons2 = np.array([st_p.lons[i]]*lons1.shape[0])
            lats1 = st_p.lats[nbrs]
            lats2 = np.array([st_p.lats[i]]*lons1.shape[0])

            nbrs_d = np.arccos(np.sin(lats1) * np.sin(lats2) + np.cos(lats1) * np.cos(lats2) * np.cos(lons1 - lons2))
            nbrs_v = st_p.values[nbrs]
            ws = np.ones(nbrs_d.shape[0]) / nbrs_d
            w_nbrs_v = nbrs_v * ws
            new_v[i] = st_p.values[i] * (1 - strength) + np.sum(w_nbrs_v) / sum(ws) * strength
    return new_v

@njit
def nbr_smoother_NB(st_p, depth: int, strength, nbr_mx):
    #lengths = sorted([e.lin.length for e in self.edges])
    #maxl = max(lengths)
    #minl = lengths[int(len(lengths)/1000)]
    #norml = maxl - minl
    new_v = np.zeros(st_p.lons.shape[0])           
    for i in range(st_p.lons.shape[0]):
        if np.isnan(st_p.lons[i]):
            new_v[i] = np.nan
        else:
            nbrs_set=next_neighbor_smoother_mx(i, np.array([i]), 0, depth, nbr_mx)
            nbrs = np.array(list(nbrs_set))
            # print(nbrs)
            # lons1 = st_p.lons[nbrs]
            # lons2 = np.tile(st_p.lons[i], nbrs.shape[0])
            # lats1 = st_p.lats[nbrs]
            # lats2 = np.tile(st_p.lats[i], nbrs.shape[0])

            # nbrs_d = np.arccos(np.sin(lats1) * np.sin(lats2) + np.cos(lats1) * np.cos(lats2) * np.cos(lons1 - lons2))
            nbrs_v = st_p.values[nbrs]
            ws = np.ones(nbrs_v.shape[0])
            w_nbrs_v = nbrs_v * ws
            new_v[i] = st_p.values[i] * (1 - strength) + np.sum(w_nbrs_v) / sum(ws) * strength
    return new_v

def calculate_non_simb_ws_NB(st_p, g_p:gen_params, nbrs, vor_centers, vor_centers_nbrs):
    vor_edgeslen_nbrs_mx = np.zeros(g_p.nbr_mx.shape)
    for i, tr in enumerate(st_p.trs):
        for j, p in enumerate(tr):
            ind1 = indexNB(g_p.nbr_mx[p], tr[j-2])
            ind2 = indexNB(g_p.nbr_mx[p], tr[j-1])
            if vor_edgeslen_nbrs_mx[p][ind1] == 0:
                if nbrs[i][j-1] != -1:
                    arg = np.sin(vor_centers[i][1]) * np.sin(vor_centers_nbrs[i][j-1][1]) + np.cos(vor_centers[i][1]) * np.cos(
                        vor_centers_nbrs[i][j-1][1]) * np.cos(vor_centers[i][0] - vor_centers_nbrs[i][j-1][0])
                    if -1<=arg<=1:
                        edge_len = np.arccos(arg)
                    else:
                        edge_len = 0
                else:
                    dist_to_c = stp.sTriangulation.angular_separation(stp.sTriangulation, vor_centers[i][0], vor_centers[i][1], st_p.lons[p], st_p.lons[p]).item()
                    dist_to_nbr = stp.sTriangulation.angular_separation(stp.sTriangulation, st_p.lons[tr[j-2]], st_p.lats[tr[j-2]], st_p.lons[p], st_p.lons[p]).item()
                    edge_len = np.sqrt(dist_to_c**2 - (dist_to_nbr/2)**2)
                vor_edgeslen_nbrs_mx[p][ind1] = edge_len 
            if vor_edgeslen_nbrs_mx[p][ind2] == 0:
                if nbrs[i][j-2] != -1:
                    arg = np.sin(vor_centers[i][1]) * np.sin(vor_centers_nbrs[i][j-2][1]) + np.cos(vor_centers[i][1]) * np.cos(
                        vor_centers_nbrs[i][j-2][1]) * np.cos(vor_centers[i][0] - vor_centers_nbrs[i][j-2][0])
                    if -1<=arg<=1:
                        edge_len = np.arccos(arg)
                    else:
                        edge_len = 0
                else:
                    dist_to_c = stp.sTriangulation.angular_separation(stp.sTriangulation, vor_centers[i][0], vor_centers[i][1], st_p.lons[p], st_p.lons[p]).item()
                    dist_to_nbr = stp.sTriangulation.angular_separation(stp.sTriangulation, st_p.lons[tr[j-1]], st_p.lats[tr[j-1]], st_p.lons[p], st_p.lons[p]).item()
                    edge_len = np.sqrt(dist_to_c**2 - (dist_to_nbr/2)**2)
                vor_edgeslen_nbrs_mx[p][ind2] = edge_len 

    
    non_simb_ws = vor_edgeslen_nbrs_mx * 6371000 / g_p.nbr_dists
    ws_sum = calc_sum_mx(non_simb_ws)
    w_parts = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(non_simb_ws, 1 / ws_sum, w_parts)
    return w_parts