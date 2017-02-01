from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import numpy as np
from time import clock
import matplotlib.pyplot as plt
from h5py import File

import astropy.units as u
from astropy.constants import c
from astro_idchiang.external import voronoi_2d_binning_m
from astro_idchiang import imshowid

# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
const = 2.0891E-4
kappa160 = 9.6 * np.pi # fitting uncertainty = 1.3
                       # Calibration uncertainty = 2.5
                       # 01/13/2017: pi facor added from erratum
WDC = 2900 # Wien's displacement constant (um*K)

# Column density to mass surface density M_sun/pc**2
col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value

THINGS_Limit = 1.0E18 # HERACLES_LIMIT: heracles*2 > things

FWHM = {'SPIRE_500': 36.09, 'SPIRE_350': 24.88, 'SPIRE_250': 18.15, 
        'Gauss_25': 25, 'PACS_160': 11.18, 'PACS_100': 7.04, 
        'HERACLES': 13}
fwhm_sp500 = FWHM['SPIRE_500'] * u.arcsec.to(u.rad) # in rad

# Calibration error of PACS_100, PACS_160, SPIRE_250, SPIRE_350, SPIRE_500
# For extended source
calerr_matrix2 = np.array([0.10,0.10,0.08,0.08,0.08]) ** 2

# Number of fitting parameters
ndim = 2


def fit_dust_density(name='NGC3198', nwalkers=20, nsteps=150):
    """
    Inputs:
        df: <pandas DataFrame>
            DataFrame contains map information for name
        name: <str>
            Object name to be calculated.
        nwalkers: <int>
            Number of 'walkers' in the mcmc algorithm
        nsteps: <int>
            Number of steps in the mcm algorithm
    Outputs (file):
        name_popt: <numpy array>
            Optimized parameters
        name_perr: <numpy array>
            Error of optimized parameters
    """
    targetSN = 5
    # Dust density in Solar Mass / pc^2
    # kappa_lambda in cm^2 / g
    # SED in MJy / sr        
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        total_gas = np.array(grp['Total_gas'])
        sed = np.array(grp['Herschel_SED'])
        bkgerr = np.array(grp['Herschel_bkgerr'])
        diskmask = np.array(grp['Diskmask'])
        D = float(np.array(grp['Galaxy_distance']))
        INCL = float(np.array(grp['INCL']))
        dp_radius = np.array(grp['DP_RADIUS'])
        # THINGS_Limit = np.array(grp['THINGS_LIMIT'])

    binmap = np.full_like(sed[:, :, 0], np.nan, dtype=int)
    # Voronoi binning
    # d --> diskmasked, len() = sum(diskmask);
    # b --> binned, len() = number of binned area
    print("Start binning " + name + "...")
    tic = clock()
    signal_d = np.min(np.abs(sed[diskmask] / bkgerr), axis=1)
    noise_d = np.ones(signal_d.shape)
    x_d, y_d = np.meshgrid(range(sed.shape[1]), range(sed.shape[0]))
    x_d, y_d = x_d[diskmask], y_d[diskmask]
    # Dividing into layers
    
    fwhm_radius = fwhm_sp500 * D * 1E3 / np.cos(INCL * np.pi / 180)
    nlayers = int(np.nanmax(dp_radius) // fwhm_radius)
    masks = []
    masks.append(dp_radius < fwhm_radius)
    for i in range(1, nlayers - 1):
        masks.append((dp_radius >= i * fwhm_radius) * 
                     (dp_radius < (i + 1) * fwhm_radius))
    masks.append(dp_radius >= (nlayers - 1) * fwhm_radius)
    ##### test image: original layers #####

    image_test1 = np.full_like(dp_radius, np.nan)
    for i in range(nlayers):
        image_test1[masks[i]] = i
    #####

    
    for i in range(nlayers - 1, -1, -1):
        judgement = np.abs(np.sum(signal_d[masks[i][diskmask]])) / np.sqrt(len(masks[i][diskmask]))
        if judgement < targetSN:
            if i > 0:
                masks[i - 1] += masks[i]
                del masks[i]
            else:
                masks[0] += masks[1]
                del masks[1]
    nlayers = len(masks)
    ##### test image: combined layers #####

    image_test2 = np.full_like(dp_radius, np.nan)
    for i in range(nlayers):
        image_test2[masks[i]] = i

    #######################################
    """ Modify radial bins here """
    masks = [masks[i][diskmask] for i in range(nlayers)]
    max_binNum = 0
    binNum = np.full_like(signal_d, np.nan)
    for i in range(nlayers):
        x_l, y_l, signal_l, noise_l = x_d[masks[i]], y_d[masks[i]], \
                                      signal_d[masks[i]], noise_d[masks[i]]
        if np.min(signal_l) > targetSN:
            binNum_l = np.arange(len(signal_l))
        else:
            binNum_l, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
                voronoi_2d_binning_m(x_l, y_l, signal_l, noise_l, targetSN, 
                                     pixelsize=1, plot=False, quiet=True)
        binNum_l += max_binNum
        max_binNum = np.max(binNum_l)
        binNum[masks[i]] = binNum_l

    for i in range(len(signal_d)):
        binmap[y_d[i], x_d[i]] = binNum[i]
    binmap = binmap.astype(float)
    binmap[binmap < 0] = np.nan
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    
    image_test0 = np.abs(sed / bkgerr)
    image_test0 = np.min(image_test0, axis=2)
    plt.figure()
    plt.subplot(221)
    imshowid(np.log10(image_test0))
    plt.title('Worst log(SNR) Herschel map')
    plt.subplot(222)
    plt.imshow(image_test1, origin='lower')
    plt.title('First cuts')
    plt.subplot(223)
    plt.imshow(image_test2, origin='lower')
    plt.title('SNR-limited cuts')
    plt.subplot(224)
    plt.imshow(np.sin(binmap), origin='lower')
    plt.title('Voronoi bin map')
    plt.suptitle(name, size=24)