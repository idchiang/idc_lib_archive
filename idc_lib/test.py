from __future__ import absolute_import, division, print_function, \
                       unicode_literals
# import emcee
import numpy as np
import pandas as pd
from time import clock
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import c, h, k_B
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


# Probability functions & model functions for fitting (internal)
def _B(T, freq=nu):
    """Return blackbody SED of temperature T(with unit) in MJy"""
    with np.errstate(over='ignore'):
        return (2 * h * freq**3 / c**2 / (np.exp(h * freq / k_B / T) - 1)
                ).to(u.Jy).value * 1E-6


def _model(wl, sigma, T, freq=nu):
    """Return fitted SED in MJy"""
    return const * kappa160 * (160.0 / wl)**2 * sigma * _B(T * u.K, freq)


def _sigma0(wl, SL, T):
    """Generate the inital guess of dust surface density"""
    return SL * (wl / 160)**2 / const / kappa160 / \
        _B(T * u.K, (c / wl / u.um).to(u.Hz))


def _lnlike(theta, x, y, inv_sigma2):
    """Probability function for fitting"""
    sigma, T = theta
    model = _model(x, sigma, T)
    if np.sum(np.isinf(inv_sigma2)):
        return -np.inf
    else:
        return -0.5 * (np.sum((y - model)**2 * inv_sigma2))


def _lnprior(theta):
    """Probability function for fitting"""
    sigma, T = theta
    if np.log10(sigma) < 3 and 0 < T < 50:
        return 0
    return -np.inf


def _lnprob(theta, x, y, inv_sigma2):
    """Probability function for fitting"""
    lp = _lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _lnlike(theta, x, y, inv_sigma2)


def testing():
    print("Initializing grid parameters...")
    """ Grid parameters """
    logsigma_step = 0.005
    min_logsigma = -5.
    max_logsigma = 3.
    T_step = 0.05
    min_T = T_step
    max_T = 50.
    logsigmas = np.arange(min_logsigma, max_logsigma, logsigma_step)
    Ts = np.arange(min_T, max_T, T_step)
    logsigmas, Ts = np.meshgrid(logsigmas, Ts)
    models = np.zeros([Ts.shape[0], Ts.shape[1], len(wl)])
    """ new codes """
    ##
    print("Constructing control model...")
    models0 = np.zeros([Ts.shape[0], Ts.shape[1], len(wl)])
    for i in range(len(wl)):
        models0[:, :, i] = _model(wl[i], 10**logsigmas, Ts, nu[i])
    ##
    print("Constructing PACS RSRF model...")
    tic = clock()
    pacs_rsrf = pd.read_csv("data/RSRF/PACS_RSRF.csv")
    pacs_wl = pacs_rsrf['Wavelength'].values
    pacs_nu = (c / pacs_wl / u.um).to(u.Hz)
    pacs_100 = pacs_rsrf['PACS_100'].values
    pacs_160 = pacs_rsrf['PACS_160'].values
    pacs_dlambda = pacs_rsrf['dlambda'].values
    del pacs_rsrf
    #
    pacs_models = np.zeros([Ts.shape[0], Ts.shape[1], len(pacs_wl)])
    for i in range(len(pacs_wl)):
        pacs_models[:, :, i] = _model(pacs_wl[i], 10**logsigmas, Ts,
                                      pacs_nu[i])
    del pacs_nu
    models[:, :, 0] = np.sum(pacs_models * pacs_dlambda * pacs_100) / \
        np.sum(pacs_dlambda * pacs_100 * pacs_wl / wl[0])
    models[:, :, 1] = np.sum(pacs_models * pacs_dlambda * pacs_160) / \
        np.sum(pacs_dlambda * pacs_160 * pacs_wl / wl[1])
    #
    del pacs_wl, pacs_100, pacs_160, pacs_dlambda, pacs_models
    ##
    print("Constructing SPIRE RSRF model...")
    spire_rsrf = pd.read_csv("data/RSRF/SPIRE_RSRF.csv")
    spire_wl = spire_rsrf['Wavelength'].values
    spire_nu = (c / spire_wl / u.um).to(u.Hz)
    spire_250 = spire_rsrf['SPIRE_250'].values
    spire_350 = spire_rsrf['SPIRE_350'].values
    spire_500 = spire_rsrf['SPIRE_500'].values
    spire_dlambda = spire_rsrf['dlambda'].values
    del spire_rsrf
    #
    spire_models = np.zeros([Ts.shape[0], Ts.shape[1], len(spire_wl)])
    for i in range(len(spire_wl)):
        spire_models[:, :, i] = _model(spire_wl[i], 10**logsigmas, Ts,
                                       spire_nu[i])
    del spire_nu
    models[:, :, 2] = np.sum(spire_models * spire_dlambda * spire_250) / \
        np.sum(spire_dlambda * spire_250 * spire_wl / wl[2])
    models[:, :, 3] = np.sum(spire_models * spire_dlambda * spire_350) / \
        np.sum(spire_dlambda * spire_350 * spire_wl / wl[3])
    models[:, :, 4] = np.sum(spire_models * spire_dlambda * spire_500) / \
        np.sum(spire_dlambda * spire_500 * spire_wl / wl[4])
    #
    del spire_wl, spire_250, spire_350, spire_500, spire_dlambda, spire_models
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
