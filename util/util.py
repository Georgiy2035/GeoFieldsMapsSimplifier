import numpy as np
import shapely
from osgeo import ogr
from shapely import MultiLineString, MultiPolygon, GeometryCollection, build_area
from numba import vectorize, float64, int32, guvectorize, njit
from numba.experimental import jitclass
from numba.core import types

NANINT = 2147483646

spec_gen = [
    ('nbr_mx', types.int32[:, :]),
    ('nbr_x', types.float64[:, :]),
    ('nbr_y', types.float64[:, :]),
    ('nbr_values', types.float64[:, :]),
    ('nbr_dists', types.float64[:, :]),
    ('grads1', types.float64[:, :]),
    ('grads2x', types.float64[:, :]),
    ('grads2y', types.float64[:, :]),
    ('lin_grads1', types.float64[:, :]),
    ('lin_grads2x', types.float64[:, :]),
    ('lin_grads2y', types.float64[:, :]),
    ('non_simb_ws', types.float64[:, :]),
]

@jitclass(spec_gen)
class gen_params(object):
    def __init__(self, nbr_mx, nbr_x, nbr_y, nbr_values, nbr_dists, 
                 grads1, grads2x, grads2y, lin_grads1, lin_grads2x, lin_grads2y, non_simb_ws):
        self.nbr_mx = nbr_mx
        self.nbr_x = nbr_x
        self.nbr_y = nbr_y
        self.nbr_values = nbr_values
        self.nbr_dists = nbr_dists
        self.grads1 = grads1
        self.grads2x = grads2x
        self.grads2y = grads2y
        self.lin_grads1 = lin_grads1
        self.lin_grads2x = lin_grads2x
        self.lin_grads2y = lin_grads2y
        self.non_simb_ws = non_simb_ws

    def set_params(self, nbr_mx, nbr_x, nbr_y, nbr_values, nbr_dists, 
                   grads1, grads2x, grads2y, lin_grads1, lin_grads2x, lin_grads2y, non_simb_ws):
        self.__init__(nbr_mx, nbr_x, nbr_y, nbr_values, nbr_dists, 
                      grads1, grads2x, grads2y, lin_grads1, lin_grads2x, lin_grads2y, non_simb_ws)
    
    def copy(self):
        return gen_params(self.nbr_mx.copy(), self.nbr_x.copy(), self.nbr_y.copy(), self.nbr_values.copy(),
                          self.nbr_dists.copy(), self.grads1.copy(), self.grads2x.copy(), self.grads2y.copy(),
                          self.lin_grads1.copy(), self.lin_grads2x.copy(), self.lin_grads2y.copy(), self.non_simb_ws.copy())

@njit # функция возвращает список соседей определённой вершины для триангуляции с параметрами st_p
def getNeighboursNB_arr(p: int, st_p):#: Union[t_params, st_params]): 
    endptr = st_p.lend[p]
    end = st_p.lst[endptr]
    nextptr = st_p.lptr[endptr]
    now = st_p.lst[nextptr]
    n_ptrs = np.array([nextptr])
    n_list = np.array([now])
    while now != end:
        nextptr = st_p.lptr[nextptr]
        now = st_p.lst[nextptr]
        n_ptrs = np.concatenate((n_ptrs, np.array([nextptr])))
        n_list = np.concatenate((n_list, np.array([now])))
    return n_ptrs, n_list

@njit
def indexNB(arr, v):
    if np.isnan(v):
        for i, v_arr in enumerate(arr):
            if np.isnan(v_arr):
                return i
    else:
        for i, v_arr in enumerate(arr):
            if v_arr == v:
                return i
    return NANINT

@njit(fastmath=True) # функция удаляет треугольник для триангуляции с параметрами st_p
def deleteTriangleNB_arr(n: int, st_p):#: Union[t_params, st_params]):
    if st_p.trs[n][0] == NANINT:
        return False
    
    for i, p in enumerate(st_p.trs[n]):
        n_ptrs, n_list = getNeighboursNB_arr(p, st_p)
        if len(n_list) < 3:
            st_p.lend[p] = NANINT
            st_p.lons[p] = np.nan
            st_p.lats[p] = np.nan
            st_p.values[p] = np.nan
            for ptr in n_ptrs:
                st_p.lst[ptr] = NANINT
                st_p.lptr[ptr] = NANINT
        else:
            nextDeleted = False
            p_prev = st_p.trs[n][i-1]
            p_next = st_p.trs[n][i-2]
            p_next_i = indexNB(n_list, p_next)
            p_prev_i = p_next_i + 1 - len(n_list)
            p_prev_ptr = n_ptrs[p_prev_i]
            p_next_ptr = n_ptrs[p_next_i]
            before_p_next_ptr = n_ptrs[p_next_i - 1]
            if st_p.lst[before_p_next_ptr] > 0: # если грань до следующей вершины остаётся
                st_p.lst[p_next_ptr] = -p_next
                st_p.lend[p] = p_next_ptr
            else:
                st_p.lptr[before_p_next_ptr] = p_prev_ptr
                st_p.lst[p_next_ptr] = NANINT
                st_p.lptr[p_next_ptr] = NANINT
                st_p.lend[p] = before_p_next_ptr
                nextDeleted = True
            if p_prev not in n_list: # если грань до предыдущей вершины не остаётся
                if nextDeleted:
                    st_p.lptr[before_p_next_ptr] = st_p.lptr[p_prev_ptr]
                else:
                    st_p.lptr[p_next_ptr] = st_p.lptr[p_prev_ptr]
                st_p.lst[p_prev_ptr] = NANINT
                st_p.lptr[p_prev_ptr] = NANINT

    st_p.trs[n] = np.array([NANINT, NANINT, NANINT])
    return True

@njit(fastmath=True) # функция удаляет треугольник для триангуляции с параметрами st_p
def deleteTriangleNB_arr2(n: int, st_p):#: Union[t_params, st_params]):
    if st_p.trs[n][0] == NANINT:
        return False

    for i, p in enumerate(st_p.trs[n]):
        n_ptrs, n_list = getNeighboursNB_arr(p, st_p)
        if len(n_list) < 3:
            st_p.lend[p] = NANINT
            st_p.x[p] = np.nan
            st_p.y[p] = np.nan
            st_p.values[p] = np.nan
            for ptr in n_ptrs:
                st_p.lst[ptr] = NANINT
                st_p.lptr[ptr] = NANINT
        else:
            nextDeleted = False
            p_prev = st_p.trs[n][i-1]
            p_next = st_p.trs[n][i-2]
            p_next_i = indexNB(n_list, p_next)
            p_prev_i = p_next_i + 1 - len(n_list)
            p_prev_ptr = n_ptrs[p_prev_i]
            p_next_ptr = n_ptrs[p_next_i]
            before_p_next_ptr = n_ptrs[p_next_i - 1]
            if st_p.lst[before_p_next_ptr] > 0: # если грань до следующей вершины остаётся
                st_p.lst[p_next_ptr] = -p_next
                st_p.lend[p] = p_next_ptr
            else:
                st_p.lptr[before_p_next_ptr] = p_prev_ptr
                st_p.lst[p_next_ptr] = NANINT
                st_p.lptr[p_next_ptr] = NANINT
                st_p.lend[p] = before_p_next_ptr
                nextDeleted = True
            if p_prev not in n_list: # если грань до предыдущей вершины не остаётся
                if nextDeleted:
                    st_p.lptr[before_p_next_ptr] = st_p.lptr[p_prev_ptr]
                else:
                    st_p.lptr[p_next_ptr] = st_p.lptr[p_prev_ptr]
                st_p.lst[p_prev_ptr] = NANINT
                st_p.lptr[p_prev_ptr] = NANINT

    st_p.trs[n] = np.array([NANINT, NANINT, NANINT])
    return True


@njit
def calcNbrMatrixNB(st_p):
    maxlen = 0

    for i in range(st_p.lend.shape[0]):
        if st_p.lend[i] != NANINT:
            leng = len(getNeighboursNB_arr(i, st_p)[1])
            if leng > maxlen:
                maxlen = leng

    nbr_matrix = np.zeros((st_p.lend.shape[0], maxlen + 1)).astype(np.int32)

    for i in range(st_p.lend.shape[0]):
        if st_p.lend[i] != NANINT:
            for j, nbr in enumerate(np.abs(getNeighboursNB_arr(i, st_p)[1])):
                nbr_matrix[i][j] = nbr

    return nbr_matrix





@guvectorize([(float64[:, :], float64[:, :], float64[:, :])], '(n, m),(n, m)->(n, m)')
def lengths_mx(x, y, res):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            res[i][j] = np.sqrt(x[i][j]**2 + y[i][j]**2)
@njit
def calculate_distances_NB_mx(g_p: gen_params):
    res = np.zeros(g_p.nbr_mx.shape)
    lengths_mx(g_p.nbr_x, g_p.nbr_y, res)
    return res

@guvectorize([(float64[:, :], float64[:], float64[:, :])], '(n, m),(n)->(n, m)')
def dif_mx_and_vec_per_col(mx, vec, res):
    for i in range(mx.shape[0]):
        for j in range(mx.shape[1]):
            res[i][j] = mx[i][j] - vec[i]
@njit
def getdifxy(c, nbr_mx):
    new = np.zeros(nbr_mx.shape)
    for i in range(nbr_mx.shape[0]):
        new[i] = c[nbr_mx[i]] - c[i]
    return new
@njit
def get_nbr_val(c, nbr_mx):
    new = np.zeros(nbr_mx.shape)
    for i in range(nbr_mx.shape[0]):
        new[i] = c[nbr_mx[i]]
    return new
@vectorize([float64(float64, float64, float64)], nopython=True)
def lengs_xyz(x, y, z):
    return np.sqrt((x**2) + (y**2) + (z**2))
@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def calc_cord_norms(vec1, vec2, vec3, vec4, vec5, vec6):
    return (vec1 - vec2) * (vec3 - vec4) - (vec5 - vec2) * (vec6 - vec4)
@guvectorize([(float64[:, :], float64[:, :], float64[:, :])], '(n, m),(n, m)->(n, m)')
def line_grads_calc_mx(v, ln, res):
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            res[i][j] = v[i][j] / ln[i][j]
@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def lengths_mx_flat(x, y, res):
    for i in range(x.shape[0]):
        res[i] = np.sqrt(x[i]**2 + y[i]**2)


@guvectorize([(float64[:, :], float64[:], float64[:, :])], '(n, m),(n)->(n, m)')
def multi_mx_on_vec_per_row(mx, vec, res):
    for i in range(mx.shape[0]):
        for j in range(mx.shape[1]):
            res[i][j] = mx[i][j] * vec[i]

@njit
def calc_sum_mx(mx):
    sum_mx = np.zeros(mx.shape[0])
    for i in range(mx.shape[0]):
        last = indexNB(mx[i], np.nan)
        sum_mx[i] = np.sum(mx[i][:last]) if last != 0 else np.nan
    return sum_mx
@njit
def calc_sum_mx0(mx):
    sum_mx = np.zeros(mx.shape[0])
    for i in range(mx.shape[0]):
        last = indexNB(mx[i], np.nan)
        sum_mx[i] = np.sum(mx[i][:last]) if last != 0 else 0
    return sum_mx
@njit
def calc_sum_mx1(mx):
    sum_mx = np.zeros(mx.shape[0])
    for i in range(mx.shape[0]):
        last = indexNB(mx[i], np.nan)
        s = np.sum(mx[i][:last])
        sum_mx[i] = s if last != 0 and s != 0 else 1
    return sum_mx

@njit
def calc_perc_mx(mx, p):
    mx_flat = mx.flatten()
    mx_flat = mx_flat[~np.isnan(mx_flat)]
    mx_flat.sort()
    return mx_flat[int(p*mx_flat.shape[0])]

@njit
def push_grad_min_NB(t_p, g_p:gen_params, s, g):
    vdif = getdifxy(t_p.values, g_p.nbr_mx)
    xpart = np.zeros(g_p.nbr_mx.shape)
    ypart = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(g_p.nbr_x, g_p.grads1[:, 0], xpart)
    multi_mx_on_vec_per_row(g_p.nbr_y, g_p.grads1[:, 1], ypart)
    grad_len = np.zeros(g_p.nbr_mx.shape[0])
    lengths_mx_flat(g_p.grads1[:, 0], g_p.grads1[:, 1], grad_len)
    nomer = xpart + ypart
    denom = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(g_p.nbr_dists, grad_len, denom)
    a = nomer / denom

    push = np.zeros(t_p.values.shape[0])
    notzero = calc_sum_mx0((vdif < 0) * a * (a < 0)) != 0
    lin_grads_edited = (np.abs(g_p.lin_grads1) / g)**s
    for (i, j), el in np.ndenumerate(lin_grads_edited > 1):
        if el:
            lin_grads_edited[i][j] = 1. 
    push[notzero] = calc_sum_mx0((vdif < 0) * (a < 0) * vdif * lin_grads_edited * a)[notzero] / calc_sum_mx0((vdif < 0) * a * (a < 0))[notzero]
    new_v = t_p.values + push

    return new_v


@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def gauss_curv_calc(xx, xy, yx, yy):
    return (xx * 10**6 * yy - xy * 10**6 * yx) #/ (10**3 + x**2 * 10**3 + y**2 * 10**3)**2 

@njit
def push_grad_NB(t_p, g_p:gen_params, s, g):
    
    vdif = getdifxy(t_p.values, g_p.nbr_mx)
    xpart = np.zeros(g_p.nbr_mx.shape)
    ypart = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(g_p.nbr_x, g_p.grads1[:, 0], xpart)
    multi_mx_on_vec_per_row(g_p.nbr_y, g_p.grads1[:, 1], ypart)
    grad_len = np.zeros(g_p.nbr_mx.shape[0])
    lengths_mx_flat(g_p.grads1[:, 0], g_p.grads1[:, 1], grad_len)
    nomer = xpart + ypart
    denom = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(g_p.nbr_dists, grad_len, denom)
    a = nomer / denom

    gauss_curv = -gauss_curv_calc(g_p.grads2x[:, 0], g_p.grads2x[:, 1], g_p.grads2y[:, 0], g_p.grads2y[:, 1])

    push = np.zeros(t_p.values.shape[0])
    pointsToLow = np.logical_and(calc_sum_mx0((vdif < 0) * a * (a < 0)) != 0, gauss_curv > 0)
    pointsToIncrs = np.logical_and(calc_sum_mx0((vdif > 0) * a * (a > 0)) != 0, gauss_curv < 0)
    lin_grads_edited = (np.abs(g_p.lin_grads1) / g)**s
    for (i, j), el in np.ndenumerate(lin_grads_edited > 1):
        if el:
            lin_grads_edited[i][j] = 1. 
    push[pointsToLow] = calc_sum_mx0((vdif < 0) * (a < 0) * vdif * lin_grads_edited * a)[pointsToLow] / calc_sum_mx0((vdif < 0) * a * (a < 0))[pointsToLow]
    push[pointsToIncrs] = calc_sum_mx0((vdif > 0) * (a > 0) * vdif * lin_grads_edited * a)[pointsToIncrs] / calc_sum_mx0((vdif > 0) * a * (a > 0))[pointsToIncrs]
    new_v = t_p.values + push

    return new_v#(gauss_curv > 0)*1.

@vectorize([float64(float64, float64, float64, float64, float64, float64)], nopython=True)
def gauss_curv_calc_spec(xx, xy, yx, yy, x, y):
    return 10**6 * (xx*y**2 + 2*x*y*xy*yx* + x**2*yy) #/ (10**3 + x**2 * 10**3 + y**2 * 10**3)**2 



@njit
def comb_NB(pool, r):
    n = len(pool)
    indices = list(range(r))
    empty = not(n and (0 < r <= n))

    if not empty:
        result = [pool[i] for i in indices]
        yield result

    while not empty:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1

            result = [pool[i] for i in indices]
            yield result

@njit
def isAbovePlane(c, p1, p2, p3):
    s1 = p1 - p2
    s2 = p3 - p2
    n = np.cross(s1, s2)
    return -np.sign(np.dot(p2 - c, n))

@njit
def expand(p_set, nbr_mx):
    res_set = p_set
    for p in p_set:
        n_list = nbr_mx[p][:indexNB(nbr_mx[p], 0)]
        for n in n_list:
            if n in p_set:
                continue
            else:
                res_set = np.concatenate((res_set, np.array([n])))
    return res_set

@njit
def next_neighbor_infenum(vi, found_set, border_difsum, distsum, n_nbrs, depth, max_depth, nbr_mx, lingrads, dists, grad_limit, max_d): 
    n_list = nbr_mx[vi][:indexNB(nbr_mx[vi], 0)]
    # print(n_list)
    new_n_nbrs = n_nbrs
    new_border_difsum = border_difsum
    new_set = found_set.copy()
    if depth != 0:
        # print(new_n_nbrs, n_nbrs)
        for i, nbr in enumerate(n_list):
            if nbr not in found_set: #and lingrads[vi][i] >= 0:
                # print(nbr)
                new_border_difsum += lingrads[vi][i]
                new_n_nbrs += 1
            elif nbr in found_set: #and -lingrads[vi][i] >= 0:
                # print(-nbr)
                new_border_difsum -= -lingrads[vi][i]
                new_n_nbrs -= 1

        if new_n_nbrs == 0:
            return found_set, border_difsum, n_nbrs
        # if new_border_difsum / new_n_nbrs < border_difsum / n_nbrs * 0.95:
        #     return found_set, border_difsum, n_nbrs
        if new_border_difsum < border_difsum:
            # print('old', n_nbrs)
            return found_set, border_difsum, n_nbrs
        
        if vi not in found_set:
            new_set = np.concatenate((found_set, np.array([vi])))
        else:
            new_set = found_set.copy()
                
        if depth == max_depth or distsum > max_d:
            # print('new', new_n_nbrs)
            return new_set, new_border_difsum, new_n_nbrs

    for i, nbr in enumerate(n_list):
        if nbr not in new_set and lingrads[vi][i] < grad_limit:
            # print('start', new_n_nbrs, nbr)
            new_set, new_border_difsum, new_n_nbrs = next_neighbor_infenum(nbr, new_set, new_border_difsum, distsum + dists[vi][i],
                                                                            new_n_nbrs, depth + 1, max_depth, nbr_mx, lingrads, dists, grad_limit, max_d)
    
    return new_set, new_border_difsum, new_n_nbrs

@njit
def next_neighbor_supremum(vi, found_set, border_difsum, distsum, n_nbrs, depth, max_depth, nbr_mx, lingrads, dists, grad_limit, max_d): 
    n_list = nbr_mx[vi][:indexNB(nbr_mx[vi], 0)]
    
    new_n_nbrs = n_nbrs
    new_border_difsum = border_difsum
    new_set = found_set.copy()
    if depth != 0:
        # print(new_n_nbrs, n_nbrs)
        for i, nbr in enumerate(n_list):
            if nbr not in found_set: #and lingrads[vi][i] >= 0:
                new_border_difsum += lingrads[vi][i]
                new_n_nbrs += 1
            elif nbr in found_set: #and -lingrads[vi][i] >= 0:
                new_border_difsum -= -lingrads[vi][i]
                new_n_nbrs -= 1
        # print(new_n_nbrs)
        if new_n_nbrs == 0:
            return found_set, border_difsum, n_nbrs
        # if new_border_difsum / new_n_nbrs < border_difsum / n_nbrs * 0.99:
        #     return found_set, border_difsum, n_nbrs
        if new_border_difsum > border_difsum:
            # print('old', n_nbrs)
            return found_set, border_difsum, n_nbrs
        
        if vi not in found_set:
            new_set = np.concatenate((found_set, np.array([vi])))
        else:
            new_set = found_set
                
        if depth == max_depth or distsum > max_d:
            # print('new', new_n_nbrs)
            return new_set, new_border_difsum, new_n_nbrs

    for i, nbr in enumerate(n_list):
        if nbr not in new_set: #and lingrads[vi][i] > -grad_limit:
            # print('start', new_n_nbrs, nbr)
            new_set, new_border_difsum, new_n_nbrs = next_neighbor_supremum(nbr, new_set, new_border_difsum, distsum + dists[vi][i],
                                                                            new_n_nbrs, depth + 1, max_depth, nbr_mx, lingrads, dists, grad_limit, max_d)
    
    return new_set, new_border_difsum, new_n_nbrs

@njit
def get_border_grads(p_set, nbr_mx, lingrads):
    grad_sum = 0
    n_sum = 0
    for p in p_set:
        n_list = nbr_mx[p][:indexNB(nbr_mx[p], 0)]
        for i, n in enumerate(n_list):
            if n in p_set:
                continue
            grad_sum += lingrads[p][i]
            n_sum += 1
    return grad_sum / n_sum

@njit
def next_neighbor_infenum2(vi, found_set, distsum, depth, max_depth, nbr_mx, lingrads, dists, grad_threshold, max_d): 
    n_list = nbr_mx[vi][:indexNB(nbr_mx[vi], 0)]

    new_set = found_set.copy()
    if depth != 0:
        if vi not in found_set:
            new_set = np.concatenate((found_set, np.array([vi])))
        else:
            new_set = found_set.copy()
                
        if depth == max_depth or distsum > max_d:
            return new_set

    for i, nbr in enumerate(n_list):
        if nbr not in new_set and 0 <= lingrads[vi][i] < grad_threshold:
            new_set = next_neighbor_infenum2(nbr, new_set, distsum + dists[vi][i], depth + 1, max_depth, nbr_mx, lingrads, dists, grad_threshold, max_d)
    
    return new_set

@njit
def next_neighbor_supremum2(vi, found_set, distsum, depth, max_depth, nbr_mx, lingrads, dists, grad_threshold, max_d): 
    n_list = nbr_mx[vi][:indexNB(nbr_mx[vi], 0)]

    new_set = found_set.copy()
    if depth != 0:
        if vi not in found_set:
            new_set = np.concatenate((found_set, np.array([vi])))
        else:
            new_set = found_set.copy()
                
        if depth == max_depth or distsum > max_d:
            return new_set

    for i, nbr in enumerate(n_list):
        if nbr not in new_set and 0 <= -lingrads[vi][i] < grad_threshold:
            new_set = next_neighbor_infenum2(nbr, new_set, distsum + dists[vi][i], depth + 1, max_depth, nbr_mx, lingrads, dists, grad_threshold, max_d)
    
    return new_set

@njit
def find_block_Inc(g_p:gen_params, grad_threshold, d_threshold_min, d_threshold_max, s_c, vdif, infs, i_numb):
    block_push_Inc = []
    block_push_Inc2 = []
    
    for inf in infs:
        extr_ps = np.array([inf])
        border_difsum = np.sum(g_p.lin_grads1[inf][~np.isnan(vdif[inf])])
        n_nbrs = vdif[inf][~np.isnan(vdif[inf])].shape[0]
        
        extr_ps, border_difsum, n_nbrs = next_neighbor_infenum(inf, extr_ps, border_difsum, 0, n_nbrs, 0, 100,
                                                                g_p.nbr_mx, g_p.lin_grads1, g_p.nbr_dists, grad_threshold * 2, d_threshold_max)

        if border_difsum / n_nbrs >= grad_threshold * s_c:
            block_push_Inc.extend(extr_ps)

        border_grads = []
        extr_ps2 = []
        for i, diam in enumerate(np.linspace(d_threshold_min, d_threshold_max, i_numb)):
            extr_ps2.append(np.array([inf]))
            extr_ps2[i] = next_neighbor_infenum2(inf, extr_ps2[i], 0, 0, 100, g_p.nbr_mx, g_p.lin_grads1, g_p.nbr_dists,
                                                grad_threshold * s_c, diam)
            border_grads.append(get_border_grads(extr_ps2[i], g_p.nbr_mx, g_p.lin_grads1))
        border_grads = np.array(border_grads)
        if np.max(border_grads) < grad_threshold * s_c:
            continue
        block_push_Inc2.extend(extr_ps2[np.argmax(border_grads)])
       
    block_push_Inc = np.array(list(set(block_push_Inc)))
    block_push_Inc2.extend(block_push_Inc)
    block_push_Inc = np.array(list(set(block_push_Inc2)))
    block_push_Inc = expand(block_push_Inc, g_p.nbr_mx)

    return block_push_Inc

@njit
def find_block_Dec(g_p:gen_params, grad_threshold, d_threshold_min, d_threshold_max, s_c, vdif, sups, i_numb):
    block_push_Dec = []
    block_push_Dec2 = []
    for sup in sups:
        extr_ps = np.array([sup])
        border_difsum = np.sum(g_p.lin_grads1[sup][~np.isnan(vdif[sup])])
        n_nbrs = vdif[sup][~np.isnan(vdif[sup])].shape[0]
        
        extr_ps, border_difsum, n_nbrs = next_neighbor_supremum(sup, extr_ps, border_difsum, 0, n_nbrs, 0, 100,
                                                                g_p.nbr_mx, g_p.lin_grads1, g_p.nbr_dists, grad_threshold * 2, d_threshold_max)

        if -border_difsum / n_nbrs >= grad_threshold * s_c:
            block_push_Dec.extend(extr_ps)
        border_grads = []
        extr_ps2 = []
        for i, diam in enumerate(np.linspace(d_threshold_min, d_threshold_max, i_numb)):
            extr_ps2.append(np.array([sup]))
            extr_ps2[i] = next_neighbor_supremum2(sup, extr_ps2[i], 0, 0, 100, g_p.nbr_mx, g_p.lin_grads1, g_p.nbr_dists,
                                                grad_threshold * s_c, diam)
            border_grads.append(-get_border_grads(extr_ps2[i], g_p.nbr_mx, g_p.lin_grads1))
        border_grads = np.array(border_grads)
        if np.max(border_grads) < grad_threshold * s_c:
            continue
        block_push_Dec2.extend(extr_ps2[np.argmax(border_grads)])
       
    block_push_Dec = np.array(list(set(block_push_Dec)))
    block_push_Dec2.extend(block_push_Dec)
    block_push_Dec = np.array(list(set(block_push_Dec2)))
    block_push_Dec = expand(block_push_Dec, g_p.nbr_mx)

    return block_push_Dec

@njit
def calc_a_param(g_p:gen_params):
    xpart = np.zeros(g_p.nbr_mx.shape)
    ypart = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(g_p.nbr_x, g_p.grads1[:, 0], xpart)
    multi_mx_on_vec_per_row(g_p.nbr_y, g_p.grads1[:, 1], ypart)
    grad_len = np.zeros(g_p.nbr_mx.shape[0])
    lengths_mx_flat(g_p.grads1[:, 0], g_p.grads1[:, 1], grad_len)
    nomer = xpart + ypart
    denom = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(g_p.nbr_dists, grad_len, denom)
    return nomer / denom

@njit
def push_grad_extremum_part1(t_p, g_p:gen_params, extr_threshold, d_threshold_min, d_threshold_max, s_c=1, ret_form_mask=False):
    vdif = getdifxy(t_p.values, g_p.nbr_mx)
    infs, sups = [], []
    for i, p_nbrs in enumerate(vdif):
        if np.isnan(p_nbrs[0]):
            continue
        notnan_nbrs = p_nbrs[~np.isnan(p_nbrs)]
        if np.sort(notnan_nbrs)[0] >= 0:
            infs.append(i)
        elif np.sort(notnan_nbrs)[-1] <= 0:
            sups.append(i)
    infs = np.array(infs)
    sups = np.array(sups)

    av_dist = calc_perc_mx(g_p.nbr_dists, 0.5)
    grad_threshold = extr_threshold / av_dist
    i_numb = int(np.round((d_threshold_max - d_threshold_min) / av_dist))

    block_push_Inc = find_block_Inc(g_p, grad_threshold, d_threshold_min, d_threshold_max, s_c, vdif, infs, i_numb)
    block_push_Dec = find_block_Dec(g_p, grad_threshold, d_threshold_min, d_threshold_max, s_c, vdif, sups, i_numb)
    
    if ret_form_mask:
        new_f_mask = np.zeros(t_p.values.shape) 
        new_f_mask[block_push_Inc] = -np.ones(block_push_Inc.shape)
        new_f_mask[block_push_Dec] = np.ones(block_push_Dec.shape)
        both = np.array(list(set(block_push_Inc) & set(block_push_Dec)))
        new_f_mask[both] = np.ones(both.shape) * 2
        return block_push_Inc, block_push_Dec, new_f_mask

    return block_push_Inc, block_push_Dec, np.ones(1)
    
@njit
def push_grad_extremum_part2(block_push_Dec, t_p, g_p:gen_params, s, g):
    vdif = getdifxy(t_p.values, g_p.nbr_mx)
    push = np.zeros(t_p.values.shape)
    pointsToLow = np.array(list(set(range(vdif.shape[0])) - set(block_push_Dec.astype(np.int64)) - set([0])))
    a = calc_a_param(g_p)

    lin_grads_edited = (np.abs(g_p.lin_grads1) / g)**s
    for (i, j), el in np.ndenumerate(lin_grads_edited > 1):
        if el:
            lin_grads_edited[i][j] = 1. 

    push[pointsToLow] = calc_sum_mx0((vdif < 0) * (a < 0) * vdif * lin_grads_edited * a)[pointsToLow] / calc_sum_mx1((vdif < 0) * a * (a < 0))[pointsToLow]
    
    return t_p.values + push

@njit
def push_grad_extremum_part3(block_push_Inc, t_p, g_p:gen_params, s, g):
    vdif = getdifxy(t_p.values, g_p.nbr_mx)
    push = np.zeros(t_p.values.shape)
    a = calc_a_param(g_p)
    pointsToIncrs = np.array(list(set(range(vdif.shape[0])) - set(block_push_Inc.astype(np.int64)) - set([0])))

    lin_grads_edited = (np.abs(g_p.lin_grads1) / g)**s
    for (i, j), el in np.ndenumerate(lin_grads_edited > 1):
        if el:
            lin_grads_edited[i][j] = 1. 

    push[pointsToIncrs] = calc_sum_mx0((vdif > 0) * (a > 0) * vdif * lin_grads_edited * a)[pointsToIncrs] / calc_sum_mx1((vdif > 0) * a * (a > 0))[pointsToIncrs]
    
    return t_p.values + push


@njit
def next_neighbor_smoother_mx(vi, found_set, depth, max_depth, nbr_mx):
    if depth == max_depth:
        return found_set
    n_list = nbr_mx[vi][:indexNB(nbr_mx[vi], 0)]
    new_set = found_set
    for nbr in n_list:
        if nbr not in found_set:
            if nbr not in new_set:
                new_set = np.concatenate((new_set, np.array([nbr])))
            new_set = next_neighbor_smoother_mx(nbr, new_set, depth + 1, max_depth, nbr_mx)
    
    return new_set

@njit
def non_simbson_smooth_NB(values, non_simb_ws, nbr_values, strength):
    v_parts = nbr_values * non_simb_ws
    new_v = calc_sum_mx(v_parts)
    new_v = new_v * strength + values * (1 - strength)

    return new_v




@njit
def ang_order(ang1, ang2):
    if ang2 - ang1 > 3.14159:
        return ang2, ang1 + 3.14159 * 2, True
    elif ang1 - ang2 > 3.14159:
        return ang1, ang2 + 3.14159 * 2, True
    elif ang2 - ang1 > 0:
        return ang1, ang2, False
    else:
        return ang2, ang1, False


def getFirstShapefrom(filename, returnCRS=False):
    file = ogr.Open(filename)
    layer = file.GetLayer(0)
    feature = layer.GetFeature(0)
    geom = feature.GetGeometryRef()
    geomWkt = geom.ExportToWkt() # (GeoJSON format)
    if returnCRS:
        return shapely.from_wkt(geomWkt), layer.GetSpatialRef().ExportToWkt() 
    return shapely.from_wkt(geomWkt)

def getAllShapesfrom(filename, returnCRS=False):
    file = ogr.Open(filename)
    layer = file.GetLayer(0)
    featureList = [layer.GetFeature(i) for i in range(layer.GetFeatureCount())]
    if returnCRS:
        return [shapely.from_wkt(i.GetGeometryRef().ExportToWkt()) for i in featureList], layer.GetSpatialRef().ExportToWkt() 
    return [shapely.from_wkt(i.GetGeometryRef().ExportToWkt()) for i in featureList]

def getMultiPolygonFromFile(filename, returnCRS=False):
    shp = getAllShapesfrom(filename, returnCRS=returnCRS)
    if returnCRS:
        crs = shp[1]
        shp = shp[0]
    ml = MultiLineString(shp)
    mp = MultiPolygon([build_area(GeometryCollection(shp))])
    #plotting.plot_polygon(mp)
    if returnCRS:
        return mp, crs
    else:
        return mp


@njit
def select_points(lats, lons, radius=1_000_000, units='degrees'):
    if units == 'degrees':
        lats = lats / 180 * 3.14159
        lons = lons / 180 * 3.14159
    saved_p = []
    max_latd = radius / 111000
    for i, lat in enumerate(lats):
        lon = lons[i]
        # if -0.00001 < lon < 0.00001:
        #     continue
        # if 3.14158 < lon < 3.14160:
        #     continue

        point_is_near = False
        for s_p in saved_p:
            if np.abs(lats[s_p] - lat) < max_latd:
                sep_ang = np.arccos(np.sin(lat) * np.sin(lats[s_p]) + np.cos(lat) * np.cos(lats[s_p]) * np.cos(lon - lons[s_p]))
                if sep_ang * 6371000 < radius:
                    point_is_near = True
                    break
        
        if not point_is_near:
            saved_p.append(i)

    return np.array(saved_p)




#ANOTHER WAY TO CALC EXTREMUMS
# while True:
        # for e in range(10):
        #     for p in extr_ps:
        #         n_list = g_p.nbr_mx[p][:indexNB(g_p.nbr_mx[p], 0)]
        #         for n in n_list:
        #             if n not in extr_ps:
        #                 break
        #         else:
        #             continue
        #         extr_ps, border_difsum, n_nbrs = next_neighbor_infenum(p, extr_ps, border_difsum, n_nbrs, 0, 1, g_p.nbr_mx, lin_grads, grad_threshold)
        #     if border_difsum == border_difsum_last:
        #         break
        #     border_difsum_last = border_difsum

#CURVATURE
# f = True
    # val_dif = getdifxy(t_p.values, g_p.nbr_mx)
    # infs, sups = [], []
    # for i, p_nbrs in enumerate(val_dif):
    #     if np.isnan(p_nbrs[0]):
    #         continue
    #     notnan_nbrs = p_nbrs[~np.isnan(p_nbrs)]
    #     combs = comb_NB(np.arange(notnan_nbrs.shape[0]), 3)
    #     sumconv = 0
    #     for comb in combs:
    #         xs = g_p.nbr_x[i][np.array(comb)]
    #         ys = g_p.nbr_y[i][np.array(comb)]
    #         vs = val_dif[i][np.array(comb)]
    #         if f:
                
    #             f = False
    #         p1 = np.array([xs[0], ys[0], vs[0]])
    #         p2 = np.array([xs[1], ys[1], vs[1]])
    #         p3 = np.array([xs[2], ys[2], vs[2]])
    #         sumconv += isAbovePlane(np.zeros(3), p1, p2, p3)#np.array([xs[0], ys[0], vs[0]]), np.array([xs[1], ys[1], vs[1]]), np.array([xs[2], ys[2], vs[2]]))
    #     if sumconv <= 0:
    #         infs.append(i)
    #     elif sumconv > 0:
    #         sups.append(i)
    # infs = np.array(infs)
    # sups = np.array(sups)
    # new_v = np.zeros(t_p.values.shape)
    # new_v[infs] = -np.ones(infs.shape)
    # new_v[sups] = np.ones(sups.shape)
    #print(vdif)

    
@njit
def push_grad_extremum_old(t_p, g_p:gen_params, s, g, extr_threshold, d_threshold_min, d_threshold_max, s_c=1):
    
    vdif = getdifxy(t_p.values, g_p.nbr_mx)
    xpart = np.zeros(g_p.nbr_mx.shape)
    ypart = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(g_p.nbr_x, g_p.grads1[:, 0], xpart)
    multi_mx_on_vec_per_row(g_p.nbr_y, g_p.grads1[:, 1], ypart)
    grad_len = np.zeros(g_p.nbr_mx.shape[0])
    lengths_mx_flat(g_p.grads1[:, 0], g_p.grads1[:, 1], grad_len)
    nomer = xpart + ypart
    denom = np.zeros(g_p.nbr_mx.shape)
    multi_mx_on_vec_per_row(g_p.nbr_dists, grad_len, denom)
    a = nomer / denom

    # gauss_curv = -gauss_curv_calc_spec(g_p.grads2x[:, 0], g_p.grads2x[:, 1], g_p.grads2y[:, 0], g_p.grads2y[:, 1], g_p.grads1[:, 0], g_p.grads1[:, 1])
    # pointsToLow = np.logical_and(calc_sum_mx0((vdif < 0) * a * (a < 0)) != 0, gauss_curv > 0)
    # pointsToIncrs = np.logical_and(calc_sum_mx0((vdif > 0) * a * (a > 0)) != 0, gauss_curv < 0)
    lin_grads_edited = (np.abs(g_p.lin_grads1) / g)**s
    for (i, j), el in np.ndenumerate(lin_grads_edited > 1):
        if el:
            lin_grads_edited[i][j] = 1. 

    # g_sign = np.sign(gauss_curv)

    infs, sups = [], []
    for i, p_nbrs in enumerate(vdif):
        if np.isnan(p_nbrs[0]):
            continue
        notnan_nbrs = p_nbrs[~np.isnan(p_nbrs)]
        if np.sort(notnan_nbrs)[0] >= 0:
            infs.append(i)
        elif np.sort(notnan_nbrs)[-1] <= 0:
            sups.append(i)
    infs = np.array(infs)
    sups = np.array(sups)
    # print(infs, sups)
    
    block_push_Inc = []
    block_push_Inc2 = []
    av_dist = calc_perc_mx(g_p.nbr_dists, 0.5)
    i_numb = int(np.round((d_threshold_max - d_threshold_min) / av_dist))
    lin_grads = vdif / g_p.nbr_dists
    grad_threshold = extr_threshold / av_dist
    for inf in infs:
        extr_ps = np.array([inf])
        border_difsum = np.sum(lin_grads[inf][~np.isnan(vdif[inf])])
        n_nbrs = vdif[inf][~np.isnan(vdif[inf])].shape[0]
        
        extr_ps, border_difsum, n_nbrs = next_neighbor_infenum(inf, extr_ps, border_difsum, 0, n_nbrs, 0, 100,
                                                                g_p.nbr_mx, lin_grads, g_p.nbr_dists, grad_threshold * 2, d_threshold_max)

        if border_difsum / n_nbrs < grad_threshold * s_c:
            border_grads = []
            extr_ps2 = []
            for i, diam in enumerate(np.linspace(d_threshold_min, d_threshold_max, i_numb)):
                extr_ps2.append(np.array([inf]))
                extr_ps2[i] = next_neighbor_infenum2(inf, extr_ps2[i], 0, 0, 100, g_p.nbr_mx, g_p.lin_grads1, g_p.nbr_dists,
                                                    grad_threshold * s_c, diam)
                border_grads.append(get_border_grads(extr_ps2[i], g_p.nbr_mx, g_p.lin_grads1))
            border_grads = np.array(border_grads)
            if np.max(border_grads) < grad_threshold * s_c:
                continue
            block_push_Inc2.extend(extr_ps2[np.argmax(border_grads)])
       
        block_push_Inc.extend(extr_ps)
    block_push_Inc = np.array(list(set(block_push_Inc)))

    for p in block_push_Inc:
        border_grads = []
        extr_ps2 = []
        for i, diam in enumerate(np.linspace(d_threshold_min, d_threshold_max, i_numb)):
            extr_ps2.append(np.array([p]))
            extr_ps2[i] = next_neighbor_infenum2(p, extr_ps2[i], 0, 0, 100, g_p.nbr_mx, g_p.lin_grads1, g_p.nbr_dists,
                                                grad_threshold * s_c, diam)
            border_grads.append(get_border_grads(extr_ps2[i], g_p.nbr_mx, g_p.lin_grads1))
        border_grads = np.array(border_grads)
        if np.max(border_grads) < grad_threshold * s_c:
            continue
        block_push_Inc2.extend(extr_ps2[np.argmax(border_grads)])
    block_push_Inc2.extend(block_push_Inc)
    block_push_Inc = np.array(list(set(block_push_Inc2)))
    block_push_Inc = expand(block_push_Inc, g_p.nbr_mx)
    # new_v = np.zeros(t_p.values.shape) 
    # new_v[block_push_Inc] = np.ones(block_push_Inc.shape)


    block_push_Dec = []
    block_push_Dec2 = []
    for sup in sups:
        extr_ps = np.array([sup])
        border_difsum = np.sum(lin_grads[sup][~np.isnan(vdif[sup])])
        n_nbrs = vdif[sup][~np.isnan(vdif[sup])].shape[0]
        
        extr_ps, border_difsum, n_nbrs = next_neighbor_supremum(sup, extr_ps, border_difsum, 0, n_nbrs, 0, 100,
                                                                g_p.nbr_mx, lin_grads, g_p.nbr_dists, grad_threshold * 2, d_threshold_max)

        if -border_difsum / n_nbrs < grad_threshold * s_c:
            border_grads = []
            extr_ps2 = []
            for i, diam in enumerate(np.linspace(d_threshold_min, d_threshold_max, i_numb)):
                extr_ps2.append(np.array([inf]))
                extr_ps2[i] = next_neighbor_supremum2(sup, extr_ps2[i], 0, 0, 100, g_p.nbr_mx, g_p.lin_grads1, g_p.nbr_dists,
                                                    grad_threshold * s_c, diam)
                border_grads.append(-get_border_grads(extr_ps2[i], g_p.nbr_mx, g_p.lin_grads1))
            border_grads = np.array(border_grads)
            if np.max(border_grads) < grad_threshold * s_c:
                continue
            block_push_Dec2.extend(extr_ps2[np.argmax(border_grads)])
       
        block_push_Dec.extend(extr_ps)
    block_push_Dec = np.array(list(set(block_push_Dec)))

    for p in block_push_Dec:
        border_grads = []
        extr_ps2 = []
        for i, diam in enumerate(np.linspace(d_threshold_min, d_threshold_max, i_numb)):
            extr_ps2.append(np.array([p]))
            extr_ps2[i] = next_neighbor_supremum2(p, extr_ps2[i], 0, 0, 100, g_p.nbr_mx, g_p.lin_grads1, g_p.nbr_dists,
                                                grad_threshold * s_c, diam)
            border_grads.append(-get_border_grads(extr_ps2[i], g_p.nbr_mx, g_p.lin_grads1))
        border_grads = np.array(border_grads)
        if np.max(border_grads) < grad_threshold * s_c:
            continue
        block_push_Dec2.extend(extr_ps2[np.argmax(border_grads)])
    block_push_Dec2.extend(block_push_Dec)
    block_push_Dec = np.array(list(set(block_push_Dec2)))
    block_push_Dec = expand(block_push_Dec, g_p.nbr_mx)

    pointsToIncrs = np.array(list(set(range(vdif.shape[0])) - set(block_push_Inc) - set([0])))
    pointsToLow = np.array(list(set(range(vdif.shape[0])) - set(block_push_Dec) - set([0])))

    # new_v = np.zeros(t_p.values.shape) 
    # new_v[block_push_Dec] = np.ones(block_push_Dec.shape)
    new_v = np.zeros(t_p.values.shape)
    push = np.zeros(t_p.values.shape[0])
    
    # new_v[pointsToLow] = -np.ones(pointsToLow.shape)
    # new_v[pointsToIncrs] += np.ones(pointsToIncrs.shape)
    push[pointsToLow] = calc_sum_mx0((vdif < 0) * (a < 0) * vdif * lin_grads_edited * a)[pointsToLow] / calc_sum_mx1((vdif < 0) * a * (a < 0))[pointsToLow]
    push[pointsToIncrs] += calc_sum_mx0((vdif > 0) * (a > 0) * vdif * lin_grads_edited * a)[pointsToIncrs] / calc_sum_mx1((vdif > 0) * a * (a > 0))[pointsToIncrs]
    # print(list(calc_sum_mx1((vdif < 0) * a * (a < 0))[pointsToLow]))
    # print(list(calc_sum_mx1((vdif > 0) * a * (a > 0))[pointsToIncrs]))
    new_v = t_p.values + push

    return new_v #(gauss_curv > 0)*1.

