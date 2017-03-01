from __future__ import absolute_import, division, print_function, \
                       unicode_literals
from time import clock
# import emcee
from h5py import File
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import c, h, k_B
import corner
from . import idc_voronoi, gal_data
range = xrange

# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
const = 2.0891E-4
kappa160 = 9.6 * np.pi
# fitting uncertainty = 1.3, Calibration uncertainty = 2.5
# 01/13/2017: pi facor added from erratum
WDC = 2900  # Wien's displacement constant (um*K)

# Column density to mass surface density M_sun/pc**2
col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value
H2HaHe = 1.36

THINGS_Limit = 1.0E18  # HERACLES_LIMIT: heracles*2 > things

FWHM = {'SPIRE_500': 36.09, 'SPIRE_350': 24.88, 'SPIRE_250': 18.15,
        'Gauss_25': 25, 'PACS_160': 11.18, 'PACS_100': 7.04,
        'HERACLES': 13}
fwhm_sp500 = FWHM['SPIRE_500'] * u.arcsec.to(u.rad)  # in rad

# Calibration error of PACS_100, PACS_160, SPIRE_250, SPIRE_350, SPIRE_500
# For extended source
calerr_matrix2 = np.array([0.10, 0.10, 0.08, 0.08, 0.08]) ** 2

# Number of fitting parameters
ndim = 2


def testing(t=1, surveys=['SPIRE_500', 'SPIRE_350', 'SPIRE_250', 'PACS_160',
                          'PACS_100']):
    RSRF = pd.DataFrame()
    if t:
        surveys = ['SPIRE_500']
    for survey in surveys:
        if survey in ['SPIRE_500', 'SPIRE_350', 'SPIRE_250']:
            filename = 'data/Gordon_RSRF/' + survey.replace('_', '') + \
                       '_resp_ext.dat'
            data = np.loadtxt(filename)
            print(data)
            data.columns = ['Wavelength', 'RSRF']
            print(data['Wavelength'])
        elif survey in ['PACS_160', 'PACS_100']:
            filename = 'data/Gordon_RSRF/' + survey.replace('_', '') + \
                       '_resp.dat'
