"""



"""


from scipy.interpolate import SmoothSphereBivariateSpline,LSQSphereBivariateSpline
# from maihr_core.maihr_model import get_inputdata_maihr

from scipy import signal
import time
# import fatiando
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from toolbox.utils import tqdm_joblib
from tqdm import tqdm
import pandas as pd
from scipy import interpolate
from scipy.interpolate import griddata
# from fatiando import gridder
from toolbox.geo import Inverse_weighted_interpolation



def xr_SmoothSphereBivariateSpline_sparse_to_grid(ds,
                                                  lat_grid=None,
                                                  lon_grid=None,
                                                  res=0.25,  # grau
                                                  s=None,
                                                  lat_90=False,
                                                  lon_180=False,
                                                    s_factor=1,
                                                  n_process=20,
                                                  virtual_mean_border=False,
                                                  smooth_bivariate=False):


    ds = ds.copy()
    ds = ds.dropna('time',how='all')

    assert 'latitude' in list(ds.coords), 'must have latitude as coords'
    assert 'longitude' in list(ds.coords), 'must have latitude as coords'
    assert 'time' in list(ds.coords), 'must have time as dim'
    assert 'points' in list(ds.coords), 'must have points as dim'

    # U = U.transpose('latitude','longitude')

    ###############
    # Project to scipy function domain
    ###############
    if lat_90:
        ds = ds.assign_coords(latitude=('points', (ds.latitude.values + 90) * np.pi / 180))
    else:
        print('latitude are between 0 and 180 degree')
    if lon_180:
        ds = ds.assign_coords(longitude=('points', (ds.longitude.values + 180) * np.pi / 180))
    else:
        print('longitude are between 0 and 360 degree')


    ##############################
    # New grid position
    #############################
    if lat_grid is None:
        latmin_rad = ds.latitude.min()
        latmax_rad = ds.latitude.max()
        lonmin_rad = ds.longitude.min()
        lonmax_rad = ds.longitude.max()

        res_radian = res*np.pi/180 # radian
        step_lat_rad = (latmax_rad - latmin_rad) / res_radian
        step_lon_rad = (lonmax_rad - lonmin_rad) / res_radian
        lat_grid = np.linspace(latmin_rad, latmax_rad, step_lat_rad)
        lon_grid = np.linspace(lonmin_rad, lonmax_rad, step_lon_rad)

    positions_x, positions_y = np.meshgrid(lon_grid,lat_grid)


    ##############
    # Virtual mean border to ensure extrapolation stability
    ##############
    if virtual_mean_border:

        nb_points_border = 3
        nb_points_along_border = 3

        lon_right  = np.linspace(lonmax_rad, lonmax_rad+(nb_points_border*res_radian), nb_points_border)
        lon_left = np.linspace(lonmin_rad-(nb_points_border*res_radian), lonmin_rad, nb_points_border)

        lat_top  = np.linspace(latmax_rad, latmax_rad+(nb_points_border*res_radian), nb_points_border)
        lat_bottom = np.linspace(latmin_rad-(nb_points_border*res_radian), latmin_rad, nb_points_border)

        position_right_x, position_right_y = np.meshgrid(lon_right,np.linspace(latmin_rad, latmax_rad, nb_points_along_border))
        position_left_x,position_left_y = np.meshgrid(lon_left,np.linspace(latmin_rad, latmax_rad, nb_points_along_border))
        position_bottom_x, position_bottom_y = np.meshgrid(np.linspace(lonmin_rad, lonmax_rad, nb_points_along_border),lat_bottom)
        position_top_x,position_top_y = np.meshgrid(np.linspace(lonmin_rad, lonmax_rad, nb_points_along_border),lat_top)

        ds_position_right = xr.DataArray(dims=['points','time'],coords={'points':range(position_right_x.size),
                                                                  'longitude':('points',position_right_x.ravel()),
                                                                  'latitude':('points',position_right_y.ravel())
                                                                        ,'time':ds.time.values})
        ds_position_left = xr.DataArray(dims=['points','time'],coords={'points':range(position_left_x.size),
                                                                  'longitude':('points',position_left_x.ravel()),
                                                                  'latitude':('points',position_left_y.ravel()),'time':ds.time.values})
        ds_position_top = xr.DataArray(dims=['points','time'],coords={'points':range(position_top_x.size),
                                                                  'longitude':('points',position_top_x.ravel()),
                                                                  'latitude':('points',position_top_y.ravel()),'time':ds.time.values})
        ds_position_bottom = xr.DataArray(dims=['points','time'],coords={'points':range(position_bottom_x.size),
                                                                  'longitude':('points',position_bottom_x.ravel()),
                                                                  'latitude':('points',position_bottom_y.ravel()),'time':ds.time.values})

        # ds_position_right = ds_position_right.expand_dims('time')
        ds_position_right = ds_position_right.fillna(ds.mean('points'))
        ds_position_left = ds_position_left.fillna(ds.mean('points'))
        ds_position_top = ds_position_top.fillna(ds.mean('points'))
        ds_position_bottom = ds_position_bottom.fillna(ds.mean('points'))

        ds = xr.concat([ds,ds_position_left,ds_position_right,ds_position_top,ds_position_bottom],dim='points')


    ##############
    # Apply scipy function 
    ###############
    def apply_spline_func(name, element,s,s_factor):

        d = element.dropna('points', how='all')

        nb_test = 30
        delta_increase= 0.15
        s_factores = np.linspace(s_factor,s_factor+(nb_test*(s_factor*delta_increase)),nb_test)
        i=0
        j=0
        while i<1:
            try:
                if s is None:
                    ss = int(d.std() * len(d.points)) * s_factor
                else:
                    ss = int(s) * s_factor

                interpolator_y = SmoothSphereBivariateSpline(d.latitude.values,
                                                             d.longitude.values,
                                                             d.values,
                                                             s=ss)
                i=1
                print(str(ss))
            except:
                j = j + 1
                s_factor =s_factores[j]
                # print(f'could not compute')

        va = interpolator_y.ev(positions_y.ravel(), positions_x.ravel()).reshape(positions_y.shape)

        ds_interp_rad = xr.DataArray(va,dims=['latitude','longitude'], coords={'latitude':LAT,'longitude':LON})
        ds_interp_rad = ds_interp_rad.assign_coords({'time':name})
        ds_interp_rad = ds_interp_rad.expand_dims('time')
        ds_interp_rad = ds_interp_rad.assign_coords({'s': s})
        ds_interp_rad = ds_interp_rad.expand_dims('s')

        return ds_interp_rad

    print('AppLy spline')
    list_ds_interp_rad = Parallel(n_jobs=n_process, prefer='threads')(delayed(apply_spline_func)(X_name, X_element,s,s_factor) for X_name, X_element in list(ds.groupby('time')))
    print('concat time spline')
    ds_interp_grid = xr.concat(list_ds_interp_rad, dim='time')
    print(ds_interp_grid)
    print('done')
    ###############
    # Reconstruct original domain
    ###############
    if lat_90:
        ds_interp_grid = ds_interp_grid.assign_coords(latitude=(ds_interp_grid.latitude.values * 180 / np.pi) - 90)
    if lon_180:
        ds_interp_grid = ds_interp_grid.assign_coords(longitude=(ds_interp_grid.longitude.values * 180 / np.pi) -180)

    print('done interp grid')
    return ds_interp_grid



def xr_fatiando_interp(ds, lat_grid=None, lon_grid=None, res=None):

    if lat_grid is None:
        latmin_rad = ds.lat.min()
        latmax_rad = ds.lat.max()
        lonmin_rad = ds.lon.min()
        lonmax_rad = ds.lon.max()

        step_lat_rad = (latmax_rad - latmin_rad) / res
        step_lon_rad = (lonmax_rad - lonmin_rad) / res
        lat_grid = np.linspace(latmin_rad, latmax_rad, step_lat_rad)
        lon_grid = np.linspace(lonmin_rad, lonmax_rad, step_lon_rad)
    positions_x, positions_y = np.meshgrid(lon_grid,lat_grid)

    shape = positions_y.shape
    ds = ds.dropna('time',how='all')
    ds_group = ds.groupby('time')

    def interp(d):
        try:
            d = d[1]
            time = d.time
            d = d.dropna('sta_id',how='any')
            xp, yp, cubic = gridder.interp(d.lat.values,
                                           d.lon.values,
                                           d.values, shape,
                                           area = (positions_y[0,0],positions_y[-1,0],positions_x[0,0],positions_x[0,-1]),
                                           algorithm='cubic',
                                           extrapolate=True)
            xp = xp.reshape(shape)
            yp = yp.reshape(shape)
            cubic = cubic.reshape(shape)
            ds_interp = xr.DataArray(cubic,dims=['lat','lon'],coords={'lat':xp[:,0],'lon':yp[0,:]})
            ds_interp = ds_interp.assign_coords({'time':time})
            ds_interp = ds_interp.expand_dims('time')

        except:
            ds_interp = None
        return ds_interp

    list_ds_interp_rad = Parallel(n_jobs=40, prefer='threads')(delayed(interp)(d) for d in tqdm(ds_group))
    list_ds_interp_rad = [ d for d in list_ds_interp_rad if d is not None]
    print('concat interp')
    ds_interp_grid = xr.concat(list_ds_interp_rad, dim='time')
    print('done interp')
    return ds_interp_grid


def xr_idx_interp(ds_y_for_spline, lon, lat,p=2):
    """
    https://rafatieppo.github.io/post/2018_07_27_idw2pyr/

    :param ds_y_for_spline:
    :param lon:
    :param lat:
    :return:
    """

    ds_y_for_spline = ds_y_for_spline.load()
    time= ds_y_for_spline.time
    print('interp idx')
    Lon, Lat = np.meshgrid(lon, lat)
    ds_group = ds_y_for_spline.groupby('time')

    out_shape = Lon.shape

    Lon = Lon.ravel()
    Lat = Lat.ravel()


    def interp_idw(dd,Lon,Lat, out_shape,lat,lon,p):
        # try:
        dd = dd[1].dropna('point', how='all')
        ds_interp = Inverse_weighted_interpolation(dd.lon.values,
                                                   dd.lat.values,
                                                   dd.values,
                                                   Lon,
                                                   Lat,p)

        ds_interp = xr.DataArray(np.array(ds_interp).reshape(out_shape), dims=['lat', 'lon'],
                                 coords={'lon':lon,
                                         'lat':lat})
        # except:
        #     ds_interp=None
        return ds_interp

    list_interped = Parallel(n_jobs=6, prefer='processes')(delayed(interp_idw)(dd,Lon,Lat, out_shape,lat,lon,p) for dd in tqdm(ds_group))
    list_interped = [d for d in list_interped if d is not None]
    print('concat interp')
    ds_interp_grid = xr.concat(list_interped, dim='time')
    ds_interp_grid = ds_interp_grid.assign_coords(time=time)
    print('done interp')
    return ds_interp_grid

def xr_smooth(ds, kernel_size,n_jobs=40):
    kernel = np.ones((kernel_size, kernel_size))
    kernel= kernel/(kernel_size*kernel_size)


    ds_group = ds.groupby('time')

    def smooth(d,kernel):
        d = d[1]
        time = d.time

        grad = signal.convolve2d(d, kernel, boundary='symm', mode='same')
        ds_interp = xr.DataArray(grad, dims=['lat', 'lon'], coords={'lat': d.lat.values, 'lon':d.lon.values})
        ds_interp = ds_interp.assign_coords({'time': time})
        ds_interp = ds_interp.expand_dims('time')

        return ds_interp

    list_ds_interp_rad = Parallel(n_jobs=n_jobs, prefer='processes')(delayed(smooth)(d,kernel) for d in tqdm(ds_group))
    list_ds_interp_rad = [ d for d in list_ds_interp_rad if d is not None]
    print('concat interp')
    ds_interp_grid = xr.concat(list_ds_interp_rad, dim='time')
    print('done interp')
    return ds_interp_grid


if __name__ == '__main__':
    pass