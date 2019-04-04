import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import Angle
from idc_lib.idc_functions import save_fits_gz
from idc_lib.gal_data import gal_data

dir_SFR = 'data/z0mgs_SFR/'
path_SFR = dir_SFR + 'IC342_SFR.FUV.W4_SPIRE500.fits.gz'
path_SFRm = dir_SFR + 'IC342_SFR.mask_SPIRE500.fits.gz'
fns = {'HI': 'data/EveryTHINGS/IC342.spire500.HI.fits.gz',
       'H2': 'data/PHANGS/IC342.spire500.CO.fits.gz',
       'SFR': path_SFR,
       'SFRm': path_SFRm,
       'dust': 'Projects/KINGFISH18_SPIRE500/IC0342/' +
       'IC0342_dust.surface.density_BE.beta=2.0_.fits.gz',
       'radius': 'data/KINGFISH_DR3_RGD/IC0342_radius.arcsec.fits.gz'}


class PointMover:
    def __init__(self, left_map, right_map):
        self.left_map = left_map
        self.right_map = right_map
        self.fig, self.ax = plt.subplots(1, 2, figsize=(8, 3))
        self.ax[0].set_title('click somewhere on the map')
        self.records = []
        #
        # Some map. Replace it with your own.
        #
        cax = self.ax[0].imshow(self.left_map, origin='lower',
                                cmap='inferno', norm=LogNorm())
        cax = self.ax[1].imshow(self.right_map, origin='lower',
                                cmap='inferno', norm=LogNorm())
        plt.colorbar(cax, ax=self.ax[0])
        #
        # Create the clicking point
        #
        self.point, = self.ax[0].plot([0], [0], 'c*')
        plt.show()
        #
        # Link the class to clicking event
        #
        self.cid = \
            self.point.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        #
        # Grab clicking event information
        #
        # print('click', event)
        if event.inaxes != self.point.axes:
            return
        j, i = int(round(event.xdata)), int(round(event.ydata))
        #
        # Update pointer position in the first axis
        #
        self.point.set_data(j, i)
        self.point.figure.canvas.draw()
        #
        # Clear the other axes. Avoid nan / inf problem here if you want.
        #
        if not np.isfinite(self.left_map[i, j]):
            return
        #
        # Update the other axes. Replace it with what you want.
        #
        print([i, j])


def manual_bkg_removal():
    # 1) Get the clicking template
    # 2) Load current mask. Create one if not existing
    SFR, hdr = fits.getdata(path_SFR, header=True)
    mask = np.zeros_like(SFR, dtype=bool)
    r = [[240, 179],
         [273, 184],
         [213, 166],
         [206, 194],
         [204, 198],
         [235, 247],
         [184, 234],
         [311, 228],
         [300, 142],
         [262, 109],
         [184, 143],
         [133, 224],
         [133, 251],
         [149, 265],
         [262, 162],
         [254, 195],
         [274, 149]]
    for coord in r:
        i, j = coord
        for p in range(SFR.shape[0]):
            for q in range(SFR.shape[1]):
                if ((p-i)**2 + (q-j)**2) < 3**2:
                    mask[p, q] = True
    SFR_masked = np.copy(SFR)
    SFR_masked[mask] = np.nan
    mask = mask.astype(int)
    # save_fits_gz(dir_SFR + 'IC342_SFR.mask_SPIRE500.fits', mask, hdr)
    # 3) Plot two panels: with and without masking
    p = PointMover(SFR, SFR_masked)
    # 4) Save the clicked points
    # 5) Generate small circles and update the mask from the list
    # 6) iterate
    pass


def radial_profile(image, radius_map, rbin=21, rlim=[0, None]):
    if rlim[1] is None:
        rlim[1] = np.nanmax(radius_map)
    bounds = np.linspace(rlim[0], rlim[1], rbin)
    rs = (bounds[:-1] + bounds[1:]) / 2
    profile = np.empty_like(rs)
    for i in range(rbin - 1):
        mask = (bounds[i] < radius_map) * (radius_map <= bounds[i + 1])
        profile[i] = np.nanmean(image[mask])
    return rs, profile


def radial_profiles():
    # 0) use dictionary to save things
    map_ = {}
    profile = pd.DataFrame()
    list_load = ['HI', 'H2', 'SFR', 'SFRm', 'dust', 'radius']
    list_avg = ['HI', 'H2', 'gas', 'SFR', 'dust']
    list_plot = ['HI', 'H2', 'H2-to-SFR', 'SFR', 'dust', 'DGR']
    # 1-1) Load HI, H2, fH2, total gas, SFR, SFR mask, dust, radius
    for p in list_load:
        fn = fns[p]
        map_[p] = fits.getdata(fn)
    # 1-2) Refine loaded files; Load R25
    # HI: nothing
    # H2: alpha_CO
    map_['H2'] *= 4.35
    # total gas: generate
    map_['gas'] = 1.36 * map_['HI'] + map_['H2']
    # SFR: mask
    map_['SFRm'] = map_['SFRm'].astype(bool)
    map_['SFR'][map_['SFRm']] = np.nan
    del map_['SFRm']
    # dust: dimension
    map_['dust'] = map_['dust'][0]
    # radius: kpc and r25
    temp = gal_data('IC342')
    r25_arcsec = Angle(temp['R25_DEG'][0] * u.deg).arcsec
    map_['radius_r25'] = map_['radius'] / r25_arcsec
    dist_mpc = temp['DIST_MPC'][0]
    map_['radius_kpc'] = map_['radius'] * dist_mpc * 1000 * \
        Angle(1 * u.arcsec).rad
    print('Distance:', dist_mpc, 'mpc')
    print('R25:', r25_arcsec, 'arcsec')
    print('R25:', r25_arcsec * dist_mpc * 1000 *
          Angle(1 * u.arcsec).rad, 'kpc')
    # 1-3) Normalize eveything to a selected radial ring
    mask = (0.160 < map_['radius_r25']) * (map_['radius_r25'] <= 0.360)
    for p in list_avg:
        temp = np.nanmean(map_[p][mask])
        map_[p] /= temp
    # 2) Generate radial profiles with the same radial bins
    # 2-1) radial_profile()
    for p in list_avg:
        profile['radius'], profile[p] = \
            radial_profile(map_[p], map_['radius_kpc'], 11, [0.5, 3.5])
    # 2-2) derived quantities: fH2, DGR
    profile['fH2'] = profile['H2'] / profile['gas']
    profile['DGR'] = profile['dust'] / profile['gas']
    profile['H2-to-SFR'] = profile['H2'] / profile['SFR']
    # 3) Plot everything with radius
    fig, ax = plt.subplots()
    for p in list_plot:
        ax.plot(profile['radius'], profile[p], label=p)
    ax.legend(fontsize=20)
    ax.set_xlabel('Radius (kpc)', fontsize=20)
    ax.set_ylabel('Normalized Value', fontsize=20)
    ax.set_yscale('log')


def azimuthal_dependence():
    # 0) use dictionary to save things
    map_ = {}
    list_load = ['HI', 'H2', 'SFR', 'SFRm', 'dust', 'radius']
    list_avg = ['HI', 'H2', 'gas', 'SFR', 'dust']
    # 1-1) Load HI, H2, fH2, total gas, SFR, SFR mask, dust, radius
    for p in list_load:
        fn = fns[p]
        map_[p] = fits.getdata(fn)
    # 1-2) Refine loaded files; Load R25
    # HI: nothing
    # H2: alpha_CO
    map_['H2'] *= 4.35
    # total gas: generate
    map_['gas'] = 1.36 * map_['HI'] + map_['H2']
    # SFR: mask
    map_['SFRm'] = map_['SFRm'].astype(bool)
    map_['SFR'][map_['SFRm']] = np.nan
    del map_['SFRm']
    # dust: dimension
    map_['dust'] = map_['dust'][0]
    # radius: kpc and r25
    temp = gal_data('IC342')
    r25_arcsec = Angle(temp['R25_DEG'][0] * u.deg).arcsec
    map_['radius_r25'] = map_['radius'] / r25_arcsec
    dist_mpc = temp['DIST_MPC'][0]
    map_['radius_kpc'] = map_['radius'] * dist_mpc * 1000 * \
        Angle(1 * u.arcsec).rad
    print('Distance:', dist_mpc, 'mpc')
    print('R25:', r25_arcsec, 'arcsec')
    print('R25:', r25_arcsec * dist_mpc * 1000 *
          Angle(1 * u.arcsec).rad, 'kpc')
    # 1-3) Normalize eveything to a selected radial ring
    mask = (0.160 < map_['radius_r25']) * (map_['radius_r25'] <= 0.360)
    for p in list_avg:
        temp = np.nanmean(map_[p][mask])
        map_[p] /= temp
    # 3) Plot how SFR trace CO in that radial ring
    fig, ax = plt.subplots()
    ax.scatter(map_['SFR'][mask], map_['H2'][mask], s=1)
    max_ = np.nanmax([map_['SFR'][mask].max(), map_['H2'][mask].max()])
    ax.plot([0, max_], [0, max_], label='x=y', color='k')
    ax.legend()
    ax.set_xlabel(r'Normalized SFR', fontsize=20)
    ax.set_ylabel(r'Normalized H$_2$', fontsize=20)


manual_bkg_removal()
# azimuthal_dependence()
