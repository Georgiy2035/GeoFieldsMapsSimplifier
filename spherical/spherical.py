import stripy as stp
import numpy as np
from pyproj import CRS, Transformer
from numba.experimental import jitclass
from numba.core import types
from numba import njit
from typing import Union, Sequence
from plyfile import PlyData, PlyElement

from util.util import deleteTriangleNB_arr, getMultiPolygonFromFile, getNeighboursNB_arr, calcNbrMatrixNB, gen_params, push_grad_min_NB, push_grad_NB, non_simbson_smooth_NB, ang_order, deleteTriangleNB_arr2
from spherical.util_st import containsXY_mp_array, getAzimXY, calculate_gradients_NB, calculate_non_simb_ws_NB, smart_smoother_NB
from planar.planar import t_params, t_gen 

NANINT = 2147483646

spec_st = [
    ('lons', types.float64[:]),
    ('lats', types.float64[:]),
    ('values', types.float64[:]),
    ('lst', types.int32[:]),
    ('lptr', types.int32[:]),
    ('lend', types.int32[:]),
    ('trs', types.int32[:, :])
]

@jitclass(spec_st)
class st_params(object):
    def __init__(self, lons, lats, values, lst, lptr, lend, trs):
        self.lons = np.concatenate((np.array([np.nan]), lons))
        self.lats = np.concatenate((np.array([np.nan]), lats))
        self.values = np.concatenate((np.array([np.nan]), values))
        
        self.lst = lst.astype(np.int32)
        self.lptr = (lptr - 1).astype(np.int32)
        self.lend = np.concatenate((np.array([NANINT]), lend - 1)).astype(np.int32)

        self.trs = (trs + 1).astype(np.int32)

    def set_params(self, lons, lats, values, lst, lptr, lend, trs):
        self.__init__(lons, lats, values, lst, lptr, lend, trs)



# класс триангуляции с методами генерализации
class st_gen(stp.sTriangulation):
    def __init__(self, lons: Union[Sequence[float], None] = None,
                    lats: Union[Sequence[float], None] = None,
                    values: Union[Sequence[float], None] = None,
                    data_src: Union[str, None] = None,
                    data_crs: Union[str, None] = None,
                    lon_type: str = '180',
                    skip_columns: int = 1,
                    skip_rows: int = 1,
                    delimeter: str = '\t',
                    values_column: Union[int, None] = None,
                    border_src: Union[str, None] = None,
                    st_p: Union[st_params, None] = None,
                    g_p: Union[gen_params, None] = None
                    ):
        
        if type(st_p) != type(None):
            super().__init__(st_p.lons[1:], st_p.lats[1:])
            self.st_p = st_p
            self.upd_st()

        else:
            if type(data_src) != type(None):
                data = np.genfromtxt(data_src, delimiter=delimeter, skip_header=skip_rows)
                if type(lons) == type(None):
                    if lon_type == '360':
                        lons = data[:, skip_columns]
                        lons = ((lons - 180) > 0) * (lons - 360) + ((lons - 180) <= 0) * lons
                        lons = lons / 180 * 3.14159
                    else:
                        lons = data[:, skip_columns]
                        lons = lons / 180 * 3.14159
                if type(lats) == type(None):
                    lats = data[:, skip_columns + 1] / 180 * 3.14159
                if type(values) == type(None):
                    if type(values_column) == type(None):
                        values_column = skip_columns + 2
                    values = data[:, values_column]
            if type(lons) == type(None):
                raise TypeError("There is no points longitudes in input")
            if type(lats) == type(None):
                raise TypeError("There is no points latitudes in input")
            if type(values) == type(None):
                raise TypeError("There is no point values in input")
            if len(lons) != len(values):
                raise IndexError("The lenghts of lats and values sequences do not match")
            super().__init__(lons, lats)
            self.values = values
            self.upd_params()

        if type(data_crs) == type(None):
            self.data_crs = CRS.from_epsg(4326)
        else:
            self.data_crs = CRS.from_user_input(data_crs)

        if type(border_src) != type(None):
            self.cutTriangulation(border_src)

        self.orig_lats = self.lats
        self.orig_lons = self.lons
        self.gen_params = False
        self.non_simb_ws_flag = False
        if type(g_p) != type(None):
            self.gen_params = True
            self.g_p = g_p
            if g_p.non_simb_ws[1][0] != 0:
                self.non_simb_ws_flag = True
            

       

    def upd_params(self):
        self.st_p = st_params(self.lons, self.lats, self.values,
                    self.lst, self.lptr, self.lend,
                    self.simplices)  
        
    def upd_st(self):
        self._lons = self.st_p.lons[1:]
        self._lats = self.st_p.lats[1:]
        self._x, self._y, self._z = stp.spherical.lonlat2xyz(self._lons, self._lats)
        self._points = np.array(list(zip(self._x, self._y, self._z)))
        self.lst = self.st_p.lst
        self.lptr = self.st_p.lptr + 1
        self.lend = self.st_p.lend[1:] + 1
        self._simplices = np.array([i for i in self.st_p.trs - 1 if i[0] + 1 != NANINT])
        self.values = self.st_p.values[1:]
        self.upd_params()
    
    def deleteTriangle(self, n):
        deleteTriangleNB_arr(n, self.st_p)
        self.upd_st()

    def cutTriangulation(self, mask_src: str):
        #print('StartCut')
        mask, mask_crs_str = getMultiPolygonFromFile(mask_src, returnCRS=True)
        mask_crs = CRS.from_user_input(mask_crs_str)
        transformer = Transformer.from_crs(mask_crs, self.data_crs)
        
        midlons, midlats = self.face_midpoints()
        
        t_in = containsXY_mp_array(np.array(list(zip(midlons, midlats))), mask, transformer)

        for i in range(len(t_in)):
            if t_in[i] == 0: #and not cross:
                deleteTriangleNB_arr(i, self.st_p)
        self.upd_st()
    

    def calc_gen_params(self):
        self.gen_params = True
        nbr_mx = calcNbrMatrixNB(self.st_p)
        ps = np.array(list(zip(self.st_p.lons, self.st_p.lats)))
        q = np.dstack((self.st_p.lons[nbr_mx], self.st_p.lats[nbr_mx]))
        # print(np.tile(ps[:, 0], (q.shape[1], 1)).T)
        ang_lens = self.angular_separation(np.tile(ps[:, 0], (q.shape[1], 1)).T.flatten(), np.tile(ps[:, 1], (q.shape[1], 1)).T.flatten(), q[:, :, 0].flatten(), q[:, :, 1].flatten())
        ang_lens = np.reshape(ang_lens, nbr_mx.shape)
        # print(ang_lens)
        nbr_x, nbr_y = getAzimXY(ps, q, ang_lens)
        self.g_p = gen_params(nbr_mx, nbr_x * 6371000, nbr_y * 6371000,
                              self.st_p.values[nbr_mx], ang_lens * 6371000,
                              np.zeros((nbr_mx.shape[0], 2)), np.zeros((nbr_mx.shape[0], 2)), np.zeros((nbr_mx.shape[0], 2)),
                              np.zeros(nbr_mx.shape), np.zeros(nbr_mx.shape), np.zeros(nbr_mx.shape), np.zeros(nbr_mx.shape))

    # def calculate_distances(self):
    #     if self.nbr_mx.shape[0] == 0:
    #         self.calcNbrMatrix()
    #     # self.dists = calculate_distances_NB(self.t_p.x, self.t_p.y,
    #     #                                     self.t_p.lst, self.t_p.lptr)
    #     self.dists = calculate_distances_NB_mx(self.t_p.x, self.t_p.y, np.vstack((np.array([0]*self.nbr_mx.shape[1]), self.nbr_mx+1)))

    def calculate_gradients(self, degree: int):
        if self.gen_params == False:
            self.calc_gen_params()
        
        areas = self.areas() * 6371000**2
        dists_in_t = np.array(self.edge_lengths()) * 6371000
        
        if degree == 1:
            self.g_p.grads1, self.g_p.lin_grads1 = calculate_gradients_NB(self.st_p, self.g_p, self.st_p.values, areas, dists_in_t)
        elif degree == 2:
            self.g_p.grads2x, self.g_p.lin_grads2x = calculate_gradients_NB(self.st_p, self.g_p, self.g_p.grads1[:, 0], areas, dists_in_t) 
            self.g_p.grads2y, self.g_p.lin_grads2y = calculate_gradients_NB(self.st_p, self.g_p, self.g_p.grads1[:, 1], areas, dists_in_t) 


    def push_grad_min(self, s, g):
        self.calculate_gradients(degree=1)
        new_v = push_grad_min_NB(self.st_p, self.g_p, s, g)
        new_st = st_gen(st_p=self.st_p, data_crs=self.data_crs, g_p=self.g_p)
        new_st.st_p.values = new_v
        new_st.upd_st()
        return new_st
    
    def push_grad(self, s, g):
        self.calculate_gradients(degree=1)
        self.calculate_gradients(degree=2)
        new_v = push_grad_NB(self.st_p, self.g_p, s, g)
        new_st = st_gen(st_p=self.st_p, data_crs=self.data_crs, g_p=self.g_p)
        new_st.st_p.values = new_v
        new_st.upd_st()
        return new_st
    

    def smart_smoother(self, depth=1):       
        if self.gen_params == False:
            self.calc_gen_params()
        new_v = smart_smoother_NB(self.st_p, depth, self.g_p.nbr_mx)
        new_st = st_gen(st_p=self.st_p, data_crs=self.data_crs, g_p=self.g_p)
        new_st.st_p.values = new_v
        new_st.upd_st()
        return new_st

    def calculate_non_simb_ws(self):
        self.non_simb_ws_flag = True
        vlons, vlats = self.voronoi_points()
        nbrs = self.neighbour_simplices()
        vor_centers = np.column_stack([vlons, vlats])
        vor_centers_nbrs = vor_centers[nbrs]
        
        self.g_p.non_simb_ws = calculate_non_simb_ws_NB(self.st_p, self.g_p, nbrs, vor_centers, vor_centers_nbrs)

    def non_simbson_smooth(self, strength=1):
        if self.gen_params == False:
            self.calc_gen_params()
        if self.non_simb_ws_flag == False:
            self.calculate_non_simb_ws()

        new_v = non_simbson_smooth_NB(self.st_p.values, self.g_p.non_simb_ws,
                                      self.g_p.nbr_values, strength=strength)
        new_st = st_gen(st_p=self.st_p, data_crs=self.data_crs, g_p=self.g_p)
        new_st.st_p.values = new_v
        new_st.upd_st()
        return new_st
    


    @njit
    def ang_dif(ang1, ang2):
        if ang2 - ang1 > 3.14159:
            return ang2 - ang1 - 3.14159 * 2
        elif ang1 - ang2 > 3.14159:
            return ang1 - ang2 - 3.14159 * 2
        else:
            return np.abs(ang2 - ang1)
    


    def export_to_ply(self, export_file_name: str, export_crs_str: Union[str, None] = None):
        if type(export_crs_str) == type(None):
            export_crs = self.data_crs
            proj_x, proj_y = self.lons, self.lats
        else:
            export_crs = CRS.from_user_input(export_crs_str)
            transformer = Transformer.from_crs(self.data_crs, export_crs)
            proj_x, proj_y = zip(*list(transformer.itransform(list(zip(self.lats, self.lons)), radians=True)))

        vertex = np.array(list(zip(proj_x, proj_y, self.values)), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        face = self.simplices.copy()
        face.dtype = [('vertex_indices', 'i4', (3,))]
        el1 = PlyElement.describe(vertex, 'vertex')
        el2 = PlyElement.describe(face.flatten(), 'face', val_types={'vertex_indices': 'i4'}, len_types={'vertex_indices': 'u4'})
        PlyData([el1, el2], text=True).write(export_file_name)

    def createPlanarTriangulation(self, t_crs_str, break_meredian_b=True):
        trs_to_del = []
        t_crs = CRS.from_user_input(t_crs_str)

        if break_meredian_b:
            par_n = 0
            for i, par in enumerate(t_crs.coordinate_operation.params):
                if par.name == 'Longitude of natural origin':
                    par_n = i
                    break
            break_meredian = (t_crs.coordinate_operation.params[par_n].value - 180) / 180 * 3.14159
            for tr_i, tr in enumerate(self.st_p.trs):
                trs_to_del_flag = False
                lon1, lon2, is180cross = ang_order(self.st_p.lons[tr[0]], self.st_p.lons[tr[1]])
                if not is180cross or break_meredian > 0:
                    if (lon1 - break_meredian) * (lon2 - break_meredian) <= 0:
                        trs_to_del_flag = True
                else:
                    if (lon1 - break_meredian - 3.14159 * 2) * (lon2 - break_meredian - 3.14159 * 2) <= 0:
                        trs_to_del_flag = True
                lon1, lon2, is180cross = ang_order(self.st_p.lons[tr[1]], self.st_p.lons[tr[2]])
                if not is180cross or break_meredian > 0:
                    if (lon1 - break_meredian) * (lon2 - break_meredian) <= 0:
                        trs_to_del_flag = True
                else:
                    if (lon1 - break_meredian - 3.14159 * 2) * (lon2 - break_meredian - 3.14159 * 2) <= 0:
                        trs_to_del_flag = True
                if trs_to_del_flag:
                    trs_to_del.append(tr_i)
        transformer = Transformer.from_crs(self.data_crs, t_crs)
        proj_x, proj_y = zip(*list(transformer.itransform(list(zip(self.lats, self.lons)), radians=True)))

        # print(self.orig_lats)
        orig_x, orig_y = zip(*list(transformer.itransform(list(zip(self.orig_lats, self.orig_lons)), radians=True)))
        t_p = t_params(np.array(proj_x), np.array(proj_y), self.values, self.lst, self.lptr, self.lend, self.simplices)
        self.t = t_gen(x=orig_x, y=orig_y, t_p=t_p, data_crs=t_crs_str)
        for i in trs_to_del:
            deleteTriangleNB_arr2(i, self.t.t_p)
        self.t.upd_st()

