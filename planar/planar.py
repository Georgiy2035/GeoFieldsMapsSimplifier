import stripy as stp
import numpy as np
from pyproj import CRS, Transformer
from shapely import contains_xy
from numba.experimental import jitclass
from numba import njit, vectorize, float64
from numba.core import types
from typing import Union, Sequence
from plyfile import PlyData, PlyElement
from copy import deepcopy

from util.util import deleteTriangleNB_arr, getMultiPolygonFromFile, calcNbrMatrixNB, indexNB, gen_params, calculate_distances_NB_mx, dif_mx_and_vec_per_col, push_grad_min_NB, push_grad_NB, non_simbson_smooth_NB
from planar.util_t import calculate_gradients_NB, smart_smoother_NB, lengs_xy, calculate_non_simb_ws_NB

NANINT = 2147483646

spec_t = [
    ('x', types.float64[:]),
    ('y', types.float64[:]),
    ('values', types.float64[:]),
    ('lst', types.int32[:]),
    ('lptr', types.int32[:]),
    ('lend', types.int32[:]),
    ('trs', types.int32[:, :])
]

@jitclass(spec_t)
class t_params(object):
    def __init__(self, x, y, values, lst, lptr, lend, trs):
        self.x = np.concatenate((np.array([np.nan]), x))
        self.y = np.concatenate((np.array([np.nan]), y))
        self.values = np.concatenate((np.array([np.nan]), values))
        
        self.lst = lst.astype(np.int32)
        self.lptr = (lptr - 1).astype(np.int32)
        self.lend = np.concatenate((np.array([NANINT]), lend - 1)).astype(np.int32)

        self.trs = (trs + 1).astype(np.int32)

    def set_params(self, x, y, values, lst, lptr, lend, trs):
        self.__init__(x, y, values, lst, lptr, lend, trs)



# класс триангуляции с методами генерализации
class t_gen(stp.Triangulation):
    def __init__(self, x: Union[Sequence[float], None] = None,
                    y: Union[Sequence[float], None] = None,
                    values: Union[Sequence[float], None] = None,
                    data_src: Union[str, None] = None,
                    data_crs: Union[str, None] = None,
                    skip_columns: int = 1,
                    skip_rows: int = 1,
                    delimeter: str = '\t',
                    values_column: Union[int, None] = None,
                    border_src: Union[str, None] = None,
                    t_p: Union[t_params, None] = None,
                    g_p: Union[gen_params, None] = None
                    ):
        
        if type(t_p) != type(None):
            super().__init__(t_p.x[1:], t_p.y[1:])
            self.t_p = t_p
            self.upd_st()

        else:
        
            if type(data_src) != type(None):
                data = np.genfromtxt(data_src, delimiter=delimeter, skip_header=skip_rows)
                if type(x) == type(None):
                    x = data[:, skip_columns]
                if type(y) == type(None):
                    y = data[:, skip_columns + 1]
                if type(values) == type(None):
                    if type(values_column) == type(None):
                        values_column = skip_columns + 2
                    values = data[:, values_column]
            if type(x) == type(None):
                raise TypeError("There is no x coordinates in input")
            if type(y) == type(None):
                raise TypeError("There is no y coordinates in input")
            if type(values) == type(None):
                raise TypeError("There is no point values in input")
            if len(x) != len(values):
                raise IndexError("The lenghts of x coordinates and values sequences do not match")
            super().__init__(x, y)
            self.values = values
            self.upd_params()

        if type(data_crs) == type(None):
            self.data_crs = CRS.from_epsg(4326)
        else:
            self.data_crs = CRS.from_user_input(data_crs)

        if type(border_src) != type(None):
            self.cutTriangulation(border_src)

        self.gen_params = False
        self.non_simb_ws_flag = False
        if type(g_p) != type(None):
            self.gen_params = True
            self.g_p = g_p
            if g_p.non_simb_ws[1][0] != 0:
                self.non_simb_ws_flag = True

        self.gh = np.array([])

    def upd_params(self):
        self.t_p = t_params(self.x, self.y, self.values,
                    self.lst, self.lptr, self.lend,
                    self.simplices)  
        
    def upd_st(self):
        self._x = self.t_p.x[1:]
        self._y = self.t_p.y[1:]
        self._points = np.array(list(zip(self._x, self._y)))
        self.lst = self.t_p.lst
        self.lptr = self.t_p.lptr + 1
        self.lend = self.t_p.lend[1:] + 1
        self._simplices = np.array([i for i in self.t_p.trs - 1 if i[0] + 1 != NANINT])
        self.values = self.t_p.values[1:]
        self.upd_params()
    
    def deleteTriangle(self, n):
        deleteTriangleNB_arr(n, self.t_p)
        self.upd_st()
        
    def cutTriangulation(self, mask_src: str):
        mask, mask_crs_str = getMultiPolygonFromFile(mask_src, returnCRS=True)
        mask_crs = CRS.from_user_input(mask_crs_str)
        transformer = Transformer.from_crs(self.data_crs, mask_crs)
        midx, midy = self.face_midpoints()
        mid_p = list(transformer.itransform(list(zip(midy, midx))))
    
        for i, p in enumerate(mid_p):
            if not contains_xy(mask, p[0], p[1]): #and not cross:
                deleteTriangleNB_arr(i, self.t_p)
        self.upd_st()

    def calc_gen_params(self):
        self.gen_params = True
        nbr_mx = calcNbrMatrixNB(self.t_p)
        nbr_x = dif_mx_and_vec_per_col(self.t_p.x[nbr_mx], self.t_p.x)
        nbr_y = dif_mx_and_vec_per_col(self.t_p.y[nbr_mx], self.t_p.y)
        
        self.g_p = gen_params(nbr_mx, nbr_x, nbr_y,
                              self.t_p.values[nbr_mx], np.zeros(nbr_mx.shape),
                              np.zeros((nbr_mx.shape[0], 2)), np.zeros((nbr_mx.shape[0], 2)), np.zeros((nbr_mx.shape[0], 2)),
                              np.zeros(nbr_mx.shape), np.zeros(nbr_mx.shape), np.zeros(nbr_mx.shape), np.zeros(nbr_mx.shape))
        self.g_p.nbr_dists = calculate_distances_NB_mx(self.g_p)

    # def calculate_distances(self):
    #     if self.calc_gen_params == False:
    #         self.calc_gen_params()
    #     self.g_p.nbr_dists = calculate_distances_NB_mx(self.g_p)

    def calculate_gradients(self, degree: int):
        if self.gen_params == False:
            self.calc_gen_params()
        
        areas = self.areas()
        dists_in_t = np.array(self.edge_lengths())
        
        if degree == 1:
            self.g_p.grads1, self.g_p.lin_grads1 = calculate_gradients_NB(self.t_p, self.g_p, self.t_p.values, areas, dists_in_t)
        elif degree == 2:
            self.g_p.grads2x, self.g_p.lin_grads2x = calculate_gradients_NB(self.t_p, self.g_p, self.g_p.grads1[:, 0], areas, dists_in_t) 
            self.g_p.grads2y, self.g_p.lin_grads2y = calculate_gradients_NB(self.t_p, self.g_p, self.g_p.grads1[:, 1], areas, dists_in_t) 


    def smart_smoother(self, depth=1):       
        if self.gen_params == False:
            self.calc_gen_params()
        new_v = smart_smoother_NB(self.t_p, depth, self.g_p.nbr_mx)
        new_t = t_gen(t_p=self.t_p, data_crs=self.data_crs, g_p=self.g_p)
        new_t.t_p.values = new_v
        new_t.upd_st()
        return new_t

    def push_grad_min(self, s, g):
        self.calculate_gradients(degree=1)
        new_v = push_grad_min_NB(self.t_p, self.g_p, s, g)
        new_t = t_gen(t_p=self.t_p, data_crs=self.data_crs, g_p=self.g_p)
        new_t.t_p.values = new_v
        new_t.upd_st()
        return new_t
    
    def push_grad(self, s, g):
        self.calculate_gradients(degree=1)
        self.calculate_gradients(degree=2)
        new_v = push_grad_NB(self.t_p, self.g_p, s, g)
        new_t = t_gen(t_p=self.t_p, data_crs=self.data_crs, g_p=self.g_p)
        new_t.t_p.values = new_v
        new_t.upd_st()
        return new_t
    
    def calculate_non_simb_ws(self):
        self.non_simbs_ws_flag = True
        vx, vy = self.voronoi_points()
        nbrs = self.neighbour_simplices()
        vor_centers = np.column_stack([vx, vy])
        vor_centers_nbrs = vor_centers[nbrs]
        
        self.g_p.non_simb_ws = calculate_non_simb_ws_NB(self.t_p, self.g_p, nbrs, vor_centers, vor_centers_nbrs)

    def non_simbson_smooth(self, strength=1):
        if self.gen_params == False:
            self.calc_gen_params()
        if self.non_simb_ws_flag == False:
            self.calculate_non_simb_ws()

        new_v = non_simbson_smooth_NB(self.t_p.values, self.g_p.non_simb_ws,
                                      self.g_p.nbr_values, strength=strength)
        new_t = t_gen(t_p=self.t_p, data_crs=self.data_crs, g_p=self.g_p)
        new_t.t_p.values = new_v
        new_t.upd_st()
        return new_t
        
    def calc_graphic_hardness(self, iso_list, scale=1/1000000, width=0.001):
        @njit
        def calc_graphic_hardness_NB(t_p: t_params, areas, iso_list, scale, width):
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
                    vec = p1 - p0
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
                square = lengs_sum * width / scale
                gh[i] = square / areas[i]
            return gh

        areas = self.areas()
        self.gh = calc_graphic_hardness_NB(self.t_p, areas, iso_list, scale=scale, width=width)


    def export_to_ply(self, export_file_name: str, export_crs_str: Union[str, None] = None):
        if type(export_crs_str) == type(None):
            export_crs = self.data_crs
            proj_x, proj_y = self.x, self.y
        else:
            export_crs = CRS.from_user_input(export_crs_str)
            transformer = Transformer.from_crs(self.data_crs, export_crs)
            proj_x, proj_y = zip(*list(transformer.itransform(list(zip(self.y, self.x)))))

        vertex = np.array(list(zip(proj_x, proj_y, self.values)), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        face = self.simplices.copy()
        face.dtype = [('vertex_indices', 'i4', (3,))]
        el1 = PlyElement.describe(vertex, 'vertex')
        el2 = PlyElement.describe(face.flatten(), 'face', val_types={'vertex_indices': 'i4'}, len_types={'vertex_indices': 'u4'})
        PlyData([el1, el2], text=True).write(export_file_name)
