from __future__ import absolute_import, division, print_function, \
                       unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.colors import LogNorm
from h5py import File
from astropy.constants import c, h, k_B
from idc_lib import gal_data
range = xrange

"""
My target: 1323
My target: 1325
My target: 1326
My target: 2169
My target: 2195
"""
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


def sb(name='NGC5457', bins=30, off=-22.5, cmap0='gist_heat', dr25=0.025):
    # name = 'NGC3198'
    # bins = 10
    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        logs_d = np.array(grp.get('Dust_surface_density_log'))  # in log
        serr = np.array(grp.get('Dust_surface_density_err_dex'))  # in dex
        topt = np.array(grp.get('Dust_temperature'))
        terr = np.array(grp.get('Dust_temperature_err'))
        total_gas = np.array(grp.get('Total_gas'))
        sed = np.array(grp.get('Herschel_SED'))
        bkgerr = np.array(grp.get('Herschel_binned_bkg'))
        binmap = np.array(grp.get('Binmap'))
        radiusmap = np.array(grp.get('Radius_map'))  # kpc
        D = float(np.array(grp['Galaxy_distance']))
        logsigmas = np.array(grp.get('logsigmas'))
        # readme = np.array(grp.get('readme'))

    nanmask = np.isnan(total_gas)
    total_gas[nanmask] = -1.
    total_gas[np.less_equal(total_gas, 0)] = np.nan
    xcoord, ycoord = np.arange(logs_d.shape[0]), np.arange(logs_d.shape[1])
    xcoord, ycoord = np.meshgrid(xcoord, ycoord)
    full_wl = np.linspace(100, 500, 200)
    full_nu = (c / full_wl / u.um).to(u.Hz)

    if len(logsigmas.shape) == 2:
        logsigmas = logsigmas[0]
    #
    pdfs = pd.read_csv('output/' + name + '_pdf.csv', index_col=0)
    pdfs.index = pdfs.index.astype(int)

    lnprob = np.full_like(logs_d, np.nan)
    binlist = np.unique(binmap)
    for bin_ in binlist:
        mask = binmap == bin_
        calerr2 = calerr_matrix2 * sed[mask][0]**2
        inv_sigma2 = 1 / (bkgerr[mask][0]**2 + calerr2)
        lnprob[mask] = _lnprob([10**logs_d[mask][0], topt[mask][0]], wl,
                               sed[mask][0], inv_sigma2)
        x = lnprob[mask][0]
        if -22.5 > x > -30:
            s = logs_d[mask][0]
            t = topt[mask][0]
            xx = xcoord[mask][0]
            yy = ycoord[mask][0]
            full_model = _model(full_wl, 10**s, t, full_nu)
            csp = np.cumsum(pdfs.iloc[bin_])[:-1]
            csp = np.append(0, csp / csp[-1])
            sss = np.interp([0.16, 0.5, 0.84], csp, logsigmas)
            print('Target bin:', bin_)
            print('(', xx, ',', yy, ')')
            print('log(Sd):', s, '; serr:', serr[mask][0])
            print('T:', t, '; terr:', terr[mask][0])
            print('chi^2:', x)
            fig, ax = plt.subplots()
            ax.scatter(wl, sed[mask][0], label='Data')
            ax.plot(full_wl, full_model, label='Fitting')
            ax.legend()
            ax.set_xlabel(r'Wavelength ($\mu m$)', size=16)
            ax.set_ylabel(r'SED ($MJysr^{-1}$)', size=16)
            ax.set_title('(' + str(xx) + ', ' + str(yy) +
                         r'), $\log(\Sigma_d)=$' + str(s) + r', $T=$' +
                         str(t), size=20)
            fig.savefig('output/' + name + 'bin_' + str(bin_) + '.png')
            fig, ax = plt.subplots()
            mask = (pdfs.iloc[bin_] > pdfs.iloc[bin_].max() / 1000).values
            smin = logsigmas[mask].min()
            smax = logsigmas[mask].max()
            ax.plot(logsigmas, pdfs.iloc[bin_], 'PDF')
            ax.plot([sss[0]]*len(pdfs.iloc[bin_]), pdfs.iloc[bin_], label='16')
            ax.plot([sss[1]]*len(pdfs.iloc[bin_]), pdfs.iloc[bin_], label='50')
            ax.plot([sss[2]]*len(pdfs.iloc[bin_]), pdfs.iloc[bin_], label='84')
            ax.set_xlim([smin, smax])
            ax.legend()
            ax.set_xlabel(r'$\log(\Sigma_d)$', size=16)
            ax.set_ylabel(r'PDF', size=16)
            ax.set_title('(' + str(xx) + ', ' + str(yy) +
                         r'), $\log(\Sigma_d)=$' + str(s) + r', $T=$' +
                         str(t), size=20)
            fig.savefig('output/' + name + 'bin_' + str(bin_) + 'PDF.png')
    plt.close('all')
