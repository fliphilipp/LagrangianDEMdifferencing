import os
import requests
import datetime
import numpy as np
import xarray
import rasterio as rio
import rioxarray
from rasterio import plot as rioplot
from rasterio import warp
from rasterio.windows import Window
import matplotlib.pylab as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid
from cmcrameri import cm as cmc
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    cdict = {'red': [],'green': [],'blue': [],'alpha': []}
    reg_index = np.linspace(start, stop, 257)
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), 
                             np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap


def inpaint_nans(array):
    valid_mask = ~np.isnan(array)
    coords = np.array(np.nonzero(valid_mask)).T
    values = array[valid_mask]
    it = LinearNDInterpolator(coords, values, fill_value=0)
    filled = it(list(np.ndindex(array.shape))).reshape(array.shape)
    return filled


def load_rema_and_velocity_data(path, fn1, fn2, velocity_data_fn, show_plot=True):
    
    date1 = fn1[15:25]
    date2 = fn2[15:25]

    src1 = rio.open(path+fn1)
    rema1 = src1.read(1)
    height = rema1.shape[0]
    width = rema1.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rio.transform.xy(src1.transform, rows, cols)
    X1 = np.array(xs)
    Y1 = np.array(ys)
    x1 = X1[0,:]
    y1 = Y1[:,0]

    src2 = rio.open(path+fn2)
    rema2 = src2.read(1)
    height = rema2.shape[0]
    width = rema2.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rio.transform.xy(src2.transform, rows, cols)
    X2 = np.array(xs)
    Y2 = np.array(ys)
    x2 = X2[0,:]
    y2 = Y2[:,0]

    # get the time difference
    dateformat = '%Y-%m-%d'
    t1 = datetime.datetime.strptime(date1, dateformat)
    t2 = datetime.datetime.strptime(date2, dateformat)
    dt = (t2-t1).days

    xr = rioxarray.open_rasterio(velocity_data_fn)
    xrange = np.max(x2) - np.min(x2)
    yrange = np.max(y2) - np.min(y2)
    xmin = np.min(x2) - xrange
    xmax = np.max(x2) + xrange
    ymin = np.min(y2) - yrange
    ymax = np.max(y2) + yrange
    v_data = xr.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

    if show_plot:
        plt.rcParams.update({'font.size': 6})
        fig, axs = plt.subplots(figsize=[8,6], dpi=100, ncols=2, nrows=2)
        remaplot1 = axs[0,0].pcolormesh(x1, y1, rema1, cmap=cmc.batlow, shading='auto')
        remaplot2 = axs[0,1].pcolormesh(x2, y2, rema2, cmap=cmc.batlow, shading='auto')
        interp = RegularGridInterpolator((np.flip(y1), x1), np.flipud(rema1), bounds_error=False, fill_value=np.nan)
        pts = np.array(list(zip(np.flip(Y2).flatten(), X2.flatten())))
        rema1_interp = interp(pts).reshape(rema2.shape)
        dh_euler = rema2-rema1_interp
        mp = -np.nanmin(dh_euler) / (np.nanmax(dh_euler)-np.nanmin(dh_euler))
        shifted_cmap = shiftedColorMap(cmc.vik, midpoint=mp, name='centered_shifted_cmap_euler')
        remadiff = axs[1,0].pcolormesh(x2, y2, rema2-rema1_interp, cmap=shifted_cmap, shading='auto')
        v_data.v.plot(ax=axs[1,1])
        n_arr = 10
        XA_grid, YA_grid = np.meshgrid(np.int32(np.round(np.linspace(0,len(v_data.x)-1, n_arr))), 
                             np.int32(np.round(np.linspace(0,len(v_data.y)-1, n_arr))))
        XX, YY = np.meshgrid(v_data.x, v_data.y)
        xdist = np.abs(XX[YA_grid[0,0],XA_grid[0,0]] - XX[YA_grid[0,1],XA_grid[0,1]])
        ydist = np.abs(YY[YA_grid[0,0],XA_grid[0,0]] - YY[YA_grid[1,0],XA_grid[1,0]])
        XA = XA_grid.flatten()
        YA = YA_grid.flatten()
        vx = v_data.vx.values.copy()
        vx = vx.reshape(vx.shape[1:])
        vy = v_data.vy.values.copy()
        vy = vy.reshape(vy.shape[1:])
        xs, ys, us, vs = [], [], [], []
        for i in range(len(XA)):
            xs.append(XX[YA[i],XA[i]])
            ys.append(YY[YA[i],XA[i]])
            us.append(vx[YA[i],XA[i]])
            vs.append(vy[YA[i],XA[i]])
        vs = np.sqrt(np.array(us)**2 + np.array(vs)**2)
        scale_factor = np.max((xdist,ydist)) / np.max(vs) * 0.95
        for i in range(len(xs)):    
            axs[1,1].arrow(xs[i], ys[i], us[i]*scale_factor, vs[i]*scale_factor, head_width=xdist/10)

        handles = [remaplot1, remaplot2, remadiff]
        for i, ax in enumerate(axs.flatten()[:3]): 
            ax.set_aspect('equal')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(handles[i], cax=cax, orientation='vertical')
            cbar.ax.get_yaxis().labelpad = 7
            cbar.ax.set_ylabel('elevation (m)', rotation=270)
            ax.set_xlabel('easting (m)')
            ax.set_ylabel('northing (m)')
        
        axs[0,0].text(0.5, 0.95, 'REMA %s' % date1 ,transform=axs[0,0].transAxes, ha='center', va='center', fontsize=10)
        axs[0,1].text(0.5, 0.95, 'REMA %s' % date2 ,transform=axs[0,1].transAxes, ha='center', va='center', fontsize=10)
        axs[1,0].text(0.5, 0.95, 'simple eulerian difference',transform=axs[1,0].transAxes, ha='center', va='center', fontsize=10)
        fig.tight_layout()
        plt.savefig('plots/data.jpg')
        
    return date1, date2, dt, rema1, x1, y1, rema2, x2, y2, v_data


def get_lagrangian_difference(date1, date2, dt, rema1, x1, y1, rema2, x2, y2, v_data, 
                              t_inc=1, velocity_fill_fn='data/itslive/ANT_G0240_0000.nc', show_plot=True):
    xr_fill = rioxarray.open_rasterio(velocity_fill_fn)
    xrange_fill = np.max(x2) - np.min(x2)
    yrange_fill = np.max(y2) - np.min(y2)
    xmin_fill = np.min(x2) - xrange_fill
    xmax_fill = np.max(x2) + xrange_fill
    ymin_fill = np.min(y2) - yrange_fill
    ymax_fill = np.max(y2) + yrange_fill
    v_data_fill = xr_fill.sel(x=slice(xmin_fill, xmax_fill), y=slice(ymax_fill, ymin_fill))

    x_vel = v_data.x.values
    y_vel = v_data.y.values
    vx = v_data.vx.values.copy()
    vx = vx.reshape(vx.shape[1:])
    vy = v_data.vy.values.copy()
    vy = vy.reshape(vy.shape[1:])

    x_vel_fill = v_data_fill.x.values
    y_vel_fill = v_data_fill.y.values
    vx_fill = v_data_fill.vx.values
    vx_fill = vx_fill.reshape(vx_fill.shape[1:])
    vy_fill = v_data_fill.vy.values
    vy_fill = vy_fill.reshape(vy_fill.shape[1:])

    v = np.sqrt(vx**2 + vy**2)
    vx[v>15e3] = np.nan
    vy[v>15e3] = np.nan
    idx_nan = np.isnan(vx) | np.isnan(vy)

    # fill nans in annual composite with overall Antarctica velocity composite
    XV, YV = np.meshgrid(x_vel, y_vel)
    pts = np.array(list(zip(YV.flatten(), XV.flatten())))
    if np.sum(np.isnan(vx)) > 0: 
        vx_interp = RegularGridInterpolator((np.flip(y_vel_fill), x_vel_fill), np.flipud(vx_fill), 
                                            bounds_error=False, fill_value=np.nan, method='linear')
        vx_fill_interp = vx_interp(pts).reshape(vx.shape)
        vx[np.isnan(vx)] = vx_fill_interp[np.isnan(vx)]
    if np.sum(np.isnan(vy)) > 0: 
        vy_interp = RegularGridInterpolator((np.flip(y_vel_fill), x_vel_fill), np.flipud(vy_fill), 
                                        bounds_error=False, fill_value=np.nan, method='linear')
        vy_fill_interp = vy_interp(pts).reshape(vy.shape)
        vy[np.isnan(vy)] = vy_fill_interp[np.isnan(vy)]

    # now fill any remaining nan values with interpolation
    vx = inpaint_nans(vx)
    vy = inpaint_nans(vy)
    v = np.sqrt(vx**2 + vy**2)

    # advect earlier DEM forward
    N_delt = round(abs(dt/t_inc))
    XS, YS = np.meshgrid(x1, y1)
    xf_ = XS.flatten()
    yf_ = YS.flatten()
    pts_start = np.array(list(zip(yf_, xf_)))
    pts = pts_start.copy()
    interpolator_u = RegularGridInterpolator((np.flip(y_vel), x_vel), np.flipud(vx), 
                                             bounds_error=False, fill_value=np.nan, method='linear')
    interpolator_v = RegularGridInterpolator((np.flip(y_vel), x_vel), np.flipud(vy), 
                                             bounds_error=False, fill_value=np.nan, method='linear')
    for i in np.arange(1,N_delt+1):
        interpu = interpolator_u(pts)
        interpv = interpolator_v(pts)
        pts[:,1] += interpu*t_inc/365.25
        pts[:,0] += interpv*t_inc/365.25
        print('time step: %i / %i (%i days of %i)' % (i, N_delt, i*t_inc, dt), end='\r')

    # interpolate to the later DEM grid points
    XE = pts[:,1].reshape(XS.shape)
    YE = pts[:,0].reshape(YS.shape)
    X2, Y2 = np.meshgrid(x2, y2)
    rema1_lagrangian = griddata(np.fliplr(pts), rema1.flatten(), (X2, Y2), method='cubic')
    dh = rema2 - rema1_lagrangian
    
    if show_plot:
        plt.rcParams.update({'font.size': 8})
        fig, (ax1,ax2) = plt.subplots(figsize=[9,4], dpi=100, ncols=2)
        advected_plot = ax1.pcolormesh(x2, y2, rema1_lagrangian, shading='auto')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(advected_plot, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('elevation (m)', rotation=270, fontsize=10)
        n_arr = 10
        YA, XA = np.meshgrid(np.int32(np.round(np.linspace(0,XE.shape[0]-1, n_arr))), np.int32(np.round(np.linspace(0,XE.shape[1]-1, n_arr))))
        XA = XA.flatten()
        YA = YA.flatten()
        for i in range(len(XA)):
            xs = XS[YA[i],XA[i]]
            ys = YS[YA[i],XA[i]]
            xe = XE[YA[i],XA[i]]
            ye = YE[YA[i],XA[i]]
            ax1.arrow(xs, ys, xe-xs, ye-ys, head_width=60)
        ax1.set_aspect('equal')
        ax1.set_title('ice-advected earlier DEM')
        ax1.set_xlabel('easting of later DEM (m)')
        ax1.set_ylabel('northing of later DEM (m)')
        mp = -np.nanmin(dh) / (np.nanmax(dh)-np.nanmin(dh))
        shifted_cmap = shiftedColorMap(cmc.vik, midpoint=mp, name='centered_shifted_cmap')
        dhplot = ax2.pcolormesh(x2, y2, dh, cmap=shifted_cmap, shading='auto')
        ax2.set_aspect('equal')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(dhplot, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('elevation change (m)', rotation=270, fontsize=10)
        ax2.set_title('lagrangian differencing')
        ax2.set_xlabel('easting of later DEM (m)')
        ax2.set_ylabel('northing of later DEM (m)')
        fig.tight_layout()
        plt.savefig('plots/lagrangian.jpg')
    
    return dh