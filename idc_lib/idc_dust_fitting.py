from time import clock
# import emcee
from h5py import File
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import astropy.units as u
from astropy.constants import c, h, k_B
# import corner
from . import idc_voronoi, gal_data


# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
const = 2.0891E-4
kappa160 = 9.6 * np.pi
# fitting uncertainty = 1.3, Calibration uncertainty = 2.5
# 01/13/2017: pi facor added from erratum
WDC = 2900  # Wien's displacement constant (um*K)

solar_oxygen_bundance = 8.69  # (O/H)_\odot, ZB12

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
cali_mat2 = np.array([[0.1**2 + 0.02**2, 0.1**2, 0, 0, 0],
                      [0.1**2, 0.1**2 + 0.02**2, 0, 0, 0],
                      [0, 0, 0.08**2 + 0.015**2, 0.08**2, 0.08**2],
                      [0, 0, 0.08**2, 0.08**2 + 0.015**2, 0.08**2],
                      [0, 0, 0.08**2, 0.08**2, 0.08**2 + 0.015**2]])
# Calibration error for non-covariance matrix mode
calerr_matrix2 = np.array([0.10, 0.10, 0.08, 0.08, 0.08]) ** 2


# Probability functions & model functions for fitting (internal)
def _B(T, freq=nu):
    """Return blackbody SED of temperature T(with unit) in MJy"""
    with np.errstate(over='ignore'):
        return (2 * h * freq**3 / c**2 / (np.exp(h * freq / k_B / T) - 1)
                ).to(u.Jy).value * 1E-6


def _model(wl, sigma, T, beta, freq=nu):
    """Return fitted SED in MJy"""
    return const * kappa160 * (160.0 / wl)**beta * \
        sigma * _B(T * u.K, freq)


def radial_map_gen(radiusmap, rbins, binvalues, zeromask):
    n = len(rbins) - 1
    assert len(zeromask) == n
    assert np.sum(zeromask) == len(binvalues)
    assert n == (len(binvalues) + np.sum(~zeromask))
    result = np.empty_like(radiusmap, dtype=float)
    j = 0
    for i in range(n):
        if zeromask[i]:
            with np.errstate(invalid='ignore'):
                mask = (radiusmap >= rbins[i]) * (radiusmap < rbins[i + 1])
            result[mask] = binvalues[j]
            j += 1
    return result

"""
# Reminders of MCMC fitting

def _sigma0(wl, SL, T):
    # Generate the inital guess of dust surface density
    return SL * (wl / 160)**const_beta / const / kappa160 / \
        _B(T * u.K, (c / wl / u.um).to(u.Hz))


def _lnlike(sigma, T, wl, obs, inv_sigma2, freq=nu):
    # Probability function for fitting
    model = _model(wl, sigma, T, freq)
    if np.sum(np.isinf(inv_sigma2)):
        return -np.inf
    else:
        return -0.5 * (np.sum((obs - model)**2 * inv_sigma2))


def _lnprior(theta):
    # Probability function for fitting
    sigma, T = theta
    if np.log10(sigma) < 3 and 0 < T < 50:
        return 0
    return -np.inf


def _lnprob(theta, x, y, inv_sigma2):
    # Probability function for fitting
    lp = _lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _lnlike(theta, x, y, inv_sigma2)
"""


def fit_dust_density(name, cov_mode=True, fixed_beta=False, beta_f=2.0):
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
    ndim = 2 if fixed_beta else 3

    if cov_mode is None:
        print('COV mode? (1 for COV, 0 for non-COV)')
        cov_mode = bool(int(input()))
    # Dust density in Solar Mass / pc^2
    # kappa_lambda in cm^2 / g
    # SED in MJy / sr
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        total_gas = np.array(grp['Total_gas'])
        sed = np.array(grp['Herschel_SED'])
        sed_unc = np.array(grp['Herschel_SED_unc'])
        bkgcov = np.array(grp['Herschel_bkgcov'])
        diskmask = np.array(grp['Diskmask'])
        glx_ctr = np.array(grp['Galaxy_center'])
        D = float(np.array(grp['Galaxy_distance']))
        INCL = float(np.array(grp['INCL']))
        PA = float(np.array(grp['PA']))
        PS = np.array(grp['PS'])
        dp_radius = np.array(grp['DP_RADIUS'])
        # THINGS_Limit = np.array(grp['THINGS_LIMIT'])

    binmap = np.full_like(sed[:, :, 0], np.nan, dtype=int)
    # Voronoi binning
    # d --> diskmasked, len() = sum(diskmask);
    # b --> binned, len() = number of binned area
    print("Start binning " + name + "...")
    tic = clock()
    noise4snr = np.array([np.sqrt(bkgcov[i, i]) for i in range(5)])
    signal_d = np.min(np.abs(sed[diskmask] / noise4snr), axis=1)
    noise_d = np.ones(signal_d.shape)
    x_d, y_d = np.meshgrid(range(sed.shape[1]), range(sed.shape[0]))
    x_d, y_d = x_d[diskmask], y_d[diskmask]
    # Dividing into layers
    judgement = np.abs(np.sum(signal_d)) / np.sqrt(len(signal_d))
    if judgement < targetSN:
        print(name, 'is having just too low overall SNR. Will not fit')

    fwhm_radius = fwhm_sp500 * D * 1E3 / np.cos(INCL * np.pi / 180)
    nlayers = int(np.nanmax(dp_radius) // fwhm_radius)
    masks = []
    with np.errstate(invalid='ignore'):
        masks.append(dp_radius < fwhm_radius)
        for i in range(1, nlayers - 1):
            masks.append((dp_radius >= i * fwhm_radius) *
                         (dp_radius < (i + 1) * fwhm_radius))
        masks.append(dp_radius >= (nlayers - 1) * fwhm_radius)
    masks = [masks[i][diskmask] for i in range(nlayers)]
    # test image: original layers
    """
    image_test1 = np.full_like(dp_radius, np.nan)
    for i in range(nlayers):
        image_test1[masks[i]] = i
    """
    #######################################

    for i in range(nlayers - 1, -1, -1):
        judgement = np.abs(np.sum(signal_d[masks[i]])) / np.sqrt(len(masks[i]))
        if judgement < targetSN:
            if i > 0:
                masks[i - 1] += masks[i]
                del masks[i]
            else:
                masks[0] += masks[1]
                del masks[1]
    nlayers = len(masks)
    # test image: combined layers #
    """
    image_test2 = np.full_like(dp_radius, np.nan)
    for i in range(nlayers):
        image_test2[masks[i]] = i
    plt.figure()
    plt.subplot(221)
    imshowid(np.log10(total_gas))
    plt.title('Total gas')
    plt.subplot(222)
    imshowid(dp_radius)
    plt.title('Deprojected radius map')
    plt.subplot(223)
    imshowid(image_test1)
    plt.title('Original cuts')
    plt.subplot(224)
    imshowid(image_test2)
    plt.title('SNR combined cuts')
    """
    #######################################
    """ Modify radial bins here """
    max_binNum = 0
    binNum = np.full_like(signal_d, np.nan)
    for i in range(nlayers):
        x_l, y_l, signal_l, noise_l = x_d[masks[i]], y_d[masks[i]], \
                                      signal_d[masks[i]], noise_d[masks[i]]
        if np.min(signal_l) > targetSN:
            binNum_l = np.arange(len(signal_l))
        else:
            binNum_l, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
                idc_voronoi.voronoi_m(x_l, y_l, signal_l, noise_l, targetSN,
                                      pixelsize=1, plot=False, quiet=True)
        binNum_l += max_binNum
        max_binNum = np.max(binNum_l)
        binNum[masks[i]] = binNum_l

    for i in range(len(signal_d)):
        binmap[y_d[i], x_d[i]] = binNum[i]
    binlist = np.unique(binNum)
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")

    print("Generating grid...")
    """ Grid parameters """
    logsigma_step = 0.0025
    min_logsigma = -4.
    max_logsigma = 1.
    T_step = 0.5
    min_T = 5.
    max_T = 50.
    beta_step = 0.1
    min_beta = -1.0
    max_beta = 4.0
    logsigmas = np.arange(min_logsigma, max_logsigma, logsigma_step)
    logsigmas_untouched = np.arange(min_logsigma, max_logsigma, logsigma_step)
    Ts = np.arange(min_T, max_T, T_step)
    Ts_untouched = np.arange(min_T, max_T, T_step)
    betas = np.arange(min_beta, max_beta, beta_step)
    betas_untouched = np.arange(min_beta, max_beta, beta_step)
    if fixed_beta:
        logsigmas, Ts = np.meshgrid(logsigmas, Ts)
    else:
        logsigmas, Ts, betas = np.meshgrid(logsigmas, Ts, betas)
    #
    try:
        if fixed_beta:
            fn = 'output/rsrf_models_b_' + str(beta_f) + '.h5'
            with File(fn, 'r') as hf:
                models = np.array(hf['models_fb'])
        else:
            with File('output/rsrf_models.h5', 'r') as hf:
                models = np.array(hf['models'])
    except IOError:
        if fixed_beta:
            models = np.zeros([Ts.shape[0], Ts.shape[1], 5])
        else:
            models = np.zeros([Ts.shape[0], Ts.shape[1], Ts.shape[2], 5])
        # Applying RSRFs to generate fake-observed models
        print("Constructing PACS RSRF model...")
        tic = clock()
        pacs_rsrf = pd.read_csv("data/RSRF/PACS_RSRF.csv")
        pacs_wl = pacs_rsrf['Wavelength'].values
        pacs_nu = (c / pacs_wl / u.um).to(u.Hz)
        pacs100dnu = pacs_rsrf['PACS_100'].values * pacs_rsrf['dnu'].values[0]
        pacs160dnu = pacs_rsrf['PACS_160'].values * pacs_rsrf['dnu'].values[0]
        del pacs_rsrf
        #
        if fixed_beta:
            pacs_models = np.zeros([Ts.shape[0], Ts.shape[1], len(pacs_wl)])
            for i in range(len(pacs_wl)):
                pacs_models[:, :, i] = _model(pacs_wl[i], 10**logsigmas, Ts,
                                              beta_f, pacs_nu[i])
            del pacs_nu
            models[:, :, 0] = np.sum(pacs_models * pacs100dnu, axis=ndim) / \
                np.sum(pacs100dnu * pacs_wl / wl[0])
            models[:, :, 1] = np.sum(pacs_models * pacs160dnu, axis=ndim) / \
                np.sum(pacs160dnu * pacs_wl / wl[1])
        else:
            pacs_models = np.zeros([Ts.shape[0], Ts.shape[1], Ts.shape[2],
                                   len(pacs_wl)])
            for i in range(len(pacs_wl)):
                pacs_models[:, :, :, i] = _model(pacs_wl[i], 10**logsigmas, Ts,
                                                 betas, pacs_nu[i])
            del pacs_nu
            models[:, :, :, 0] = np.sum(pacs_models * pacs100dnu, axis=ndim) /\
                np.sum(pacs100dnu * pacs_wl / wl[0])
            models[:, :, :, 1] = np.sum(pacs_models * pacs160dnu, axis=ndim) /\
                np.sum(pacs160dnu * pacs_wl / wl[1])
        #
        del pacs_wl, pacs100dnu, pacs160dnu, pacs_models
        ##
        print("Constructing SPIRE RSRF model...")
        spire_rsrf = pd.read_csv("data/RSRF/SPIRE_RSRF.csv")
        spire_wl = spire_rsrf['Wavelength'].values
        spire_nu = (c / spire_wl / u.um).to(u.Hz)
        spire250dnu = spire_rsrf['SPIRE_250'].values * \
            spire_rsrf['dnu'].values[0]
        spire350dnu = spire_rsrf['SPIRE_350'].values * \
            spire_rsrf['dnu'].values[0]
        spire500dnu = spire_rsrf['SPIRE_500'].values * \
            spire_rsrf['dnu'].values[0]
        del spire_rsrf
        #
        if fixed_beta:
            spire_models = np.zeros([Ts.shape[0], Ts.shape[1], len(spire_wl)])
            for i in range(len(spire_wl)):
                spire_models[:, :, i] = _model(spire_wl[i], 10**logsigmas, Ts,
                                               beta_f, spire_nu[i])
            del spire_nu
            models[:, :, 2] = np.sum(spire_models * spire250dnu, axis=ndim) / \
                np.sum(spire250dnu * spire_wl / wl[2])
            models[:, :, 3] = np.sum(spire_models * spire350dnu, axis=ndim) / \
                np.sum(spire350dnu * spire_wl / wl[3])
            models[:, :, 4] = np.sum(spire_models * spire500dnu, axis=ndim) / \
                np.sum(spire500dnu * spire_wl / wl[4])
        else:
            spire_models = np.zeros([Ts.shape[0], Ts.shape[1], Ts.shape[2],
                                     len(spire_wl)])
            for i in range(len(spire_wl)):
                spire_models[:, :, :, i] = _model(spire_wl[i], 10**logsigmas,
                                                  Ts, betas, spire_nu[i])
            del spire_nu
            models[:, :, :, 2] = np.sum(spire_models * spire250dnu,
                                        axis=ndim) / \
                np.sum(spire250dnu * spire_wl / wl[2])
            models[:, :, :, 3] = np.sum(spire_models * spire350dnu,
                                        axis=ndim) / \
                np.sum(spire350dnu * spire_wl / wl[3])
            models[:, :, :, 4] = np.sum(spire_models * spire500dnu,
                                        axis=ndim) / \
                np.sum(spire500dnu * spire_wl / wl[4])
        #
        del spire_wl, spire250dnu, spire350dnu, spire500dnu
        del spire_models
        print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
        if fixed_beta:
            fn = 'output/rsrf_models_b_' + str(beta_f) + '.h5'
            with File(fn, 'a') as hf:
                hf.create_dataset('models_fb', data=models)
        else:
            with File('output/rsrf_models.h5', 'a') as hf:
                hf.create_dataset('models', data=models)

    # RSRF models with fixed beta and T values
    logsigmas_f = np.arange(min_logsigma, max_logsigma, logsigma_step)
    Ts_f = np.array([12.0, 15.0, 18.0])
    logsigmas_f, Ts_f = np.meshgrid(logsigmas_f, Ts_f)
    try:
        with File('output/rsrf_models_f.h5', 'r') as hf:
            models_f = np.array(hf['models'])
    except IOError:
        models_f = np.zeros([Ts_f.shape[0], Ts_f.shape[1], 5])
        # Applying RSRFs to generate fake-observed models
        print("Constructing PACS RSRF model for fixed T and beta...")
        tic = clock()
        pacs_rsrf = pd.read_csv("data/RSRF/PACS_RSRF.csv")
        pacs_wl = pacs_rsrf['Wavelength'].values
        pacs_nu = (c / pacs_wl / u.um).to(u.Hz)
        pacs100dnu = pacs_rsrf['PACS_100'].values * pacs_rsrf['dnu'].values[0]
        pacs160dnu = pacs_rsrf['PACS_160'].values * pacs_rsrf['dnu'].values[0]
        del pacs_rsrf
        #
        pacs_models = np.zeros([Ts_f.shape[0], Ts_f.shape[1], len(pacs_wl)])
        for i in range(len(pacs_wl)):
            pacs_models[:, :, i] = _model(pacs_wl[i], 10**logsigmas_f, Ts_f,
                                          beta_f, pacs_nu[i])
        del pacs_nu
        models_f[:, :, 0] = np.sum(pacs_models * pacs100dnu, axis=2) / \
            np.sum(pacs100dnu * pacs_wl / wl[0])
        models_f[:, :, 1] = np.sum(pacs_models * pacs160dnu, axis=2) / \
            np.sum(pacs160dnu * pacs_wl / wl[1])
        #
        del pacs_wl, pacs100dnu, pacs160dnu, pacs_models
        ##
        print("Constructing SPIRE RSRF model...")
        spire_rsrf = pd.read_csv("data/RSRF/SPIRE_RSRF.csv")
        spire_wl = spire_rsrf['Wavelength'].values
        spire_nu = (c / spire_wl / u.um).to(u.Hz)
        spire250dnu = spire_rsrf['SPIRE_250'].values * \
            spire_rsrf['dnu'].values[0]
        spire350dnu = spire_rsrf['SPIRE_350'].values * \
            spire_rsrf['dnu'].values[0]
        spire500dnu = spire_rsrf['SPIRE_500'].values * \
            spire_rsrf['dnu'].values[0]
        del spire_rsrf
        #
        spire_models = np.zeros([Ts_f.shape[0], Ts_f.shape[1], len(spire_wl)])
        for i in range(len(spire_wl)):
            spire_models[:, :, i] = _model(spire_wl[i], 10**logsigmas_f,
                                           Ts_f, beta_f, spire_nu[i])
        del spire_nu
        models_f[:, :, 2] = np.sum(spire_models * spire250dnu, axis=2) / \
            np.sum(spire250dnu * spire_wl / wl[2])
        models_f[:, :, 3] = np.sum(spire_models * spire350dnu, axis=2) / \
            np.sum(spire350dnu * spire_wl / wl[3])
        models_f[:, :, 4] = np.sum(spire_models * spire500dnu, axis=2) / \
            np.sum(spire500dnu * spire_wl / wl[4])
        #
        del spire_wl, spire250dnu, spire350dnu, spire500dnu
        del spire_models
        print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
        with File('output/rsrf_models_f.h5', 'a') as hf:
            hf.create_dataset('models', data=models_f)
    """
    Start fitting
    """
    print("Start fitting", name, "dust surface density...")
    tic = clock()
    p = 0
    sopt, serr, topt, terr, bopt, berr = [], [], [], [], [], []
    b_model = []
    f_b_model, f_sopt, f_serr = [[], [], []], [[], [], []], [[], [], []]
    gas_avg, sed_avg, radius_avg, cov_n1s, pdfs = [], [], [], [], []
    t_pdfs, b_pdfs = [], []
    inv_sigma2s = []
    # results = [] # array for saving all the raw chains
    for i in range(len(binlist)):
        if (i + 1) / len(binlist) > p:
            print('Step', (i + 1), '/', str(len(binlist)) + '.',
                  "Elapsed time:", round(clock()-tic, 3), "s.")
            p += 0.1
        """ Binning everything """
        bin_ = (binmap == binlist[i])
        # total_gas weighted radius / total gas
        radius_avg.append(np.nansum(dp_radius[bin_] * total_gas[bin_]) /
                          np.nansum(total_gas[bin_]))
        gas_avg.append(np.nanmean(total_gas[bin_]))
        # mean sed
        sed_avg.append(np.nanmean(sed[bin_], axis=0))
        # Mean uncertainty
        unc2_avg = np.mean(sed_unc[bin_]**2, axis=0)
        unc2_avg[np.isnan(unc2_avg)] = 0
        if cov_mode:
            # bkg covariance matrix
            bkgcov_avg = bkgcov / np.sum(bin_)
            # uncertainty diagonal matrix
            unc2cov_avg = np.identity(5) * unc2_avg
            # calibration error covariance matrix
            sed_vec = sed_avg[-1].reshape(1, 5)
            calcov = sed_vec.T * cali_mat2 * sed_vec
            # Finally everything for covariance matrix is here...
            cov_n1 = np.linalg.inv(bkgcov_avg + unc2cov_avg + calcov)
            cov_n1s.append(cov_n1)
            """ Grid fitting """
            # sed_avg[i].shape = (5)
            # models.shape = (len(logsigmas), len(Ts), 5)
            diff = models - sed_avg[-1]
            temp_matrix = np.empty_like(diff)
            if fixed_beta:
                for j in range(5):
                    temp_matrix[:, :, j] = np.sum(diff * cov_n1[:, j],
                                                  axis=ndim)
            else:
                for j in range(5):
                    temp_matrix[:, :, :, j] = np.sum(diff * cov_n1[:, j],
                                                     axis=ndim)
            chi2 = np.sum(temp_matrix * diff, axis=ndim)
        else:
            # Non-covariance matrix mode (old fashion)
            # 1D bkgerr
            bkg2_avg = (bkgcov / np.sum(bin_)).diagonal()
            # calibration error covariance matrix
            calerr2 = calerr_matrix2 * sed_avg[-1]**2
            # Finally everything for variance is here...
            inv_sigma2 = 1 / (bkg2_avg + calerr2 + unc2_avg)
            inv_sigma2s.append(inv_sigma2)
            """ Grid fitting """
            chi2 = (np.sum((sed_avg[i] - models)**2 * inv_sigma2, axis=ndim))
        """ Find the (s, t) that gives Maximum likelihood """
        temp = chi2.argmin()
        if fixed_beta:
            tempa = temp // (chi2.shape[1])
            tempb = temp % chi2.shape[1]
            s_ML = logsigmas[tempa, tempb]
            t_ML = Ts[tempa, tempb]
            b_ML = beta_f
            b_model.append(models[tempa, tempb])
        else:
            tempa = temp // (chi2.shape[1] * chi2.shape[2])
            temp = temp % (chi2.shape[1] * chi2.shape[2])
            tempb = temp // chi2.shape[2]
            tempc = temp % chi2.shape[2]
            s_ML = logsigmas[tempa, tempb, tempc]
            t_ML = Ts[tempa, tempb, tempc]
            b_ML = betas[tempa, tempb, tempc]
            b_model.append(models[tempa, tempb, tempc])
        """ Show map """
        # plt.figure()
        # imshowid(np.log10(-lnprobs))

        """ Continue saving """
        pr = np.exp(-0.5 * chi2)
        mask = chi2 < np.nanmin(chi2) + 12
        logsigmas_cp, Ts_cp, pr_cp = \
            logsigmas[mask], Ts[mask], pr[mask]
        #
        ids = np.argsort(logsigmas_cp)
        logsigmas_cp = logsigmas_cp[ids]
        prs = pr_cp[ids]
        csp = np.cumsum(prs)[:-1]
        csp = np.append(0, csp / csp[-1])
        sss = np.interp([0.16, 0.5, 0.84], csp, logsigmas_cp).tolist()
        #
        idT = np.argsort(Ts_cp)
        Ts_cp = Ts_cp[idT]
        prT = pr_cp[idT]
        csp = np.cumsum(prT)[:-1]
        csp = np.append(0, csp / csp[-1])
        sst = np.interp([0.16, 0.5, 0.84], csp, Ts_cp).tolist()
        #
        if fixed_beta:
            ssb = [beta_f] * 3
        else:
            betas_cp = betas[mask]
            idb = np.argsort(betas_cp)
            betas_cp = betas_cp[idb]
            prb = pr_cp[idb]
            csp = np.cumsum(prb)[:-1]
            csp = np.append(0, csp / csp[-1])
            ssb = np.interp([0.16, 0.5, 0.84], csp, betas_cp).tolist()
        """ Saving to results """
        sss[1], sst[1], ssb[1] = s_ML, t_ML, b_ML
        sopt.append(sss[1])
        topt.append(sst[1])
        bopt.append(ssb[1])
        serr.append(max(sss[2]-sss[1], sss[1]-sss[0]))
        terr.append(max(sst[2]-sst[1], sst[1]-sst[0]))
        berr.append(max(ssb[2]-ssb[1], ssb[1]-ssb[0]))
        """ New: saving PDF """
        if fixed_beta:
            pdf = np.sum(pr, axis=(0))
            pdfs.append(pdf / np.sum(pdf))
            pdf = np.sum(pr, axis=(1))
            t_pdfs.append(pdf / np.sum(pdf))
        else:
            pdf = np.sum(pr, axis=(0, 2))
            pdfs.append(pdf / np.sum(pdf))
            pdf = np.sum(pr, axis=(1, 2))
            t_pdfs.append(pdf / np.sum(pdf))
            pdf = np.sum(pr, axis=(0, 1))
            b_pdfs.append(pdf / np.sum(pdf))
        if cov_mode:
            for j in range(len(Ts_f)):
                diff = models_f[j] - sed_avg[-1]
                temp_matrix = np.empty_like(diff)
                for k in range(5):
                    temp_matrix[:, k] = np.sum(diff * cov_n1[:, k], axis=1)
                chi2 = np.sum(temp_matrix * diff, axis=1)
                temp = chi2.argmin()
                s_ML = logsigmas_f[0, temp]
                f_b_model[j].append(models_f[j, temp])
                """ Continue saving """
                pr = np.exp(-0.5 * chi2)
                mask = chi2 < np.nanmin(chi2) + 12
                logsigmas_cp, pr_cp = logsigmas_f[0][mask], pr[mask]
                #
                ids = np.argsort(logsigmas_cp)
                logsigmas_cp = logsigmas_cp[ids]
                prs = pr_cp[ids]
                csp = np.cumsum(prs)[:-1]
                csp = np.append(0, csp / csp[-1])
                sss = np.interp([0.16, 0.5, 0.84], csp, logsigmas_cp).tolist()
                """ Saving to results """
                sss[1] = s_ML
                f_sopt[j].append(sss[1])
                f_serr[j].append(max(sss[2]-sss[1], sss[1]-sss[0]))

    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    # Saving to h5 file
    # Total_gas and dust in M_sun/pc**2
    # Temperature in K
    # SED in MJy/sr
    # D in Mpc
    # Galaxy_distance in Mpc
    # Galaxy_center in pixel [y, x]
    # INCL, PA in degrees
    # PS in arcsec
    if fixed_beta:
        fn = '_dust_data_fb.h5'
    else:
        fn = '_dust_data.h5' if cov_mode else '_ncdust_data.h5'
    with File('output/' + name + fn, 'a') as hf:
        hf.create_dataset('Total_gas', data=gas_avg)
        hf.create_dataset('Herschel_SED', data=sed_avg)
        hf.create_dataset('Dust_surface_density_log', data=sopt)
        # sopt in log scale (search sss)
        hf.create_dataset('Dust_surface_density_err_dex', data=serr)
        # serr in dex
        hf.create_dataset('Dust_temperature', data=topt)
        hf.create_dataset('Dust_temperature_err', data=terr)
        hf.create_dataset('beta', data=bopt)
        hf.create_dataset('beta_err', data=berr)
        hf.create_dataset('Herschel_covariance_matrix', data=cov_n1s)
        hf.create_dataset('Herschel_variance', data=inv_sigma2s)
        hf.create_dataset('Binmap', data=binmap)
        hf.create_dataset('Binlist', data=binlist)
        hf.create_dataset('Galaxy_distance', data=D)
        hf.create_dataset('Galaxy_center', data=glx_ctr)
        hf.create_dataset('INCL', data=INCL)
        hf.create_dataset('PA', data=PA)
        hf.create_dataset('PS', data=PS)
        hf.create_dataset('Radius_avg', data=radius_avg)  # kpc
        hf.create_dataset('logsigmas', data=logsigmas_untouched)
        hf.create_dataset('Ts', data=Ts_untouched)
        hf.create_dataset('betas', data=betas_untouched)
        hf.create_dataset('PDF', data=pdfs)
        hf.create_dataset('PDF_T', data=t_pdfs)
        hf.create_dataset('PDF_B', data=b_pdfs)
        hf.create_dataset('Best_fit_model', data=b_model)
        hf.create_dataset('Fixed_temperatures', data=Ts_f[:, 0])
        hf.create_dataset('Fixed_beta', data=beta_f)
        hf.create_dataset('Fixed_best_fit_model', data=f_b_model)
        hf.create_dataset('Fixed_best_fit_sopt', data=f_sopt)
        hf.create_dataset('Fixed_best_fit_serr', data=f_serr)
    print("Datasets saved.")


def read_dust_file(name='NGC5457', bins=30, off=45., cmap0='gist_heat',
                   dr25=0.025, ncmode=False, cmap2='seismic',
                   fixed_beta=False):
    plt.close('all')
    plt.ioff()
    if fixed_beta:
        fn = 'output/' + name + '_dust_data_fb.h5'
    else:
        fn = 'output/' + name + '_dust_data.h5'
    with File(fn, 'r') as hf:
        alogs_d = np.array(hf['Dust_surface_density_log'])  # in log
        aserr = np.array(hf['Dust_surface_density_err_dex'])  # in dex
        atopt = np.array(hf['Dust_temperature'])
        aterr = np.array(hf['Dust_temperature_err'])
        abopt = np.array(hf['beta'])
        aberr = np.array(hf['beta_err'])
        agas = np.array(hf['Total_gas'])
        ased = np.array(hf['Herschel_SED'])
        acov_n1 = np.array(hf['Herschel_covariance_matrix'])
        binmap = np.array(hf['Binmap'])
        binlist = np.array(hf['Binlist'])
        aradius = np.array(hf['Radius_avg'])  # kpc
        D = float(np.array(hf['Galaxy_distance']))
        logsigmas = np.array(hf['logsigmas'])
        Ts = np.array(hf['Ts'])
        apdfs = np.array(hf['PDF'])
        t_apdfs = np.array(hf['PDF_T'])
        amodel = np.array(hf['Best_fit_model'])
        Ts_f = np.array(hf['Fixed_temperatures'])
        beta_f = float(np.array(hf['Fixed_beta']))
        f_amodel = np.array(hf['Fixed_best_fit_model'])
        f_asopt = np.array(hf['Fixed_best_fit_sopt'])
        # f_serr = np.array(hf['Fixed_best_fit_serr'])

    if ncmode:
        with File('output/' + name + '_ncdust_data.h5', 'r') as hf:
            ncalogs_d = np.array(hf['Dust_surface_density_log'])  # in log
            ncaserr = np.array(hf['Dust_surface_density_err_dex'])  # in dex
            ncinv_sigma2 = np.array(hf['Herschel_variance'])
            ncbinmap = np.array(hf['Binmap'])
            ncamodel = np.array(hf['Best_fit_model'])
            assert not np.sum(binmap != ncbinmap)
            del ncbinmap

    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        ubthings = np.array(grp['THINGS']) * col2sur * H2HaHe
        ubheracles = np.array(grp['HERACLES'])
        ubradius = np.array(grp['DP_RADIUS'])
        logSFR = np.array(grp['logSFR'])

    if Ts_f.ndim == 2:
        Ts_f = Ts_f[:, 0]
    # Filtering bad fits
    diffs = (ased - amodel).reshape(-1, 5)
    with np.errstate(invalid='ignore'):
        amodel_d_sed = amodel / ased
    achi2 = np.array([np.dot(np.dot(diffs[i].T, acov_n1[i]), diffs[i])
                      for i in range(len(binlist))])
    #
    f_achi2 = np.full_like(f_asopt, np.nan, dtype=float)
    for i in range(len(Ts_f)):
        diffs = (ased - f_amodel[i]).reshape(-1, 5)
        f_achi2[i] = np.array([np.dot(np.dot(diffs[j].T, acov_n1[j]), diffs[j])
                              for j in range(len(binlist))])
    #
    if ncmode:
        ncachi2 = np.sum(diffs**2 * ncinv_sigma2, axis=1)
        # nc results
        diffs = (ased - ncamodel).reshape(-1, 5)
        nccoachi2 = np.array([np.dot(np.dot(diffs[i].T, acov_n1[i]), diffs[i])
                              for i in range(len(binlist))])
        ncncachi2 = np.sum(diffs**2 * ncinv_sigma2, axis=1)
        del ncamodels, diffs, ncinv_sigma2, acov_n1, ncatopt, ncabopt

    # Calculating DGR
    alogs_gas = np.log10(agas)
    alogdgr = alogs_d - alogs_gas
    # Calculating distances
    # D in Mpc. Need r25 in kpc
    R25 = gal_data.gal_data([name]).field('R25_DEG')[0]
    R25 *= (np.pi / 180.) * (D * 1E3)
    aradius /= R25
    ubradius /= R25
    x_ext0, x_ext1, y_ext0, y_ext1 = 0, ubradius.shape[1], 0, ubradius.shape[0]
    ycm = np.nanargmin(ubradius) % x_ext1
    xcm = np.nanargmin(ubradius) % x_ext1
    x_ext1 -= 1
    y_ext1 -= 1
    dis = 50
    if np.isnan(ubradius[ycm, x_ext0]):
        x_ext0 = -((ubradius[ycm, xcm - dis] - ubradius[ycm, xcm]) *
                   (xcm / dis) + ubradius[ycm, xcm])
    else:
        x_ext0 = -ubradius[ycm, x_ext0]
    if np.isnan(ubradius[ycm, x_ext1]):
        x_ext1 = (ubradius[ycm, xcm + dis] - ubradius[ycm, xcm]) * \
            ((x_ext1 - xcm) / dis) + ubradius[ycm, xcm]
    else:
        x_ext1 = ubradius[ycm, x_ext1]
    if np.isnan(ubradius[y_ext0, xcm]):
        y_ext0 = -((ubradius[ycm - dis, xcm] - ubradius[ycm, xcm]) *
                   (ycm / dis) + ubradius[ycm, xcm])
    else:
        y_ext0 = -ubradius[y_ext0, xcm]
    if np.isnan(ubradius[y_ext1, xcm]):
        y_ext1 = (ubradius[ycm + dis, xcm] - ubradius[ycm, xcm]) * \
            ((y_ext1 - ycm) / dis) + ubradius[ycm, xcm]
    else:
        y_ext1 = ubradius[y_ext1, xcm]

    # Constructing maps
    with np.errstate(invalid='ignore', divide='ignore'):
        logs_H2 = np.log10(ubheracles)
    logs_d = np.full_like(binmap, np.nan, dtype=float)
    topt = np.full_like(binmap, np.nan, dtype=float)
    serr = np.full_like(binmap, np.nan, dtype=float)
    terr = np.full_like(binmap, np.nan, dtype=float)
    bopt = np.full_like(binmap, np.nan, dtype=float)
    berr = np.full_like(binmap, np.nan, dtype=float)
    logs_gas = np.full_like(binmap, np.nan, dtype=float)
    chi2 = np.full_like(binmap, np.nan, dtype=float)
    logdgr = np.full_like(binmap, np.nan, dtype=float)
    radius = np.full_like(binmap, np.nan, dtype=float)
    model_d_sed = np.full([binmap.shape[0], binmap.shape[1], 5], np.nan,
                          dtype=float)
    sed = np.full([binmap.shape[0], binmap.shape[1], 5], np.nan, dtype=float)
    f_sopt = np.full([len(Ts_f), binmap.shape[0], binmap.shape[1]], np.nan,
                     dtype=float)
    f_chi2 = np.full([len(Ts_f), binmap.shape[0], binmap.shape[1]], np.nan,
                     dtype=float)

    if ncmode:
        nclogs_d = np.full_like(binmap, np.nan, dtype=float)
        ncserr = np.full_like(binmap, np.nan, dtype=float)
        concchi2 = np.full_like(binmap, np.nan, dtype=float)
        nccochi2 = np.full_like(binmap, np.nan, dtype=float)
        ncncchi2 = np.full_like(binmap, np.nan, dtype=float)
    for i in range(len(binlist)):
        mask = binmap == binlist[i]
        logs_d[mask] = alogs_d[i]
        topt[mask] = atopt[i]
        serr[mask] = aserr[i]
        terr[mask] = aterr[i]
        bopt[mask] = abopt[i]
        berr[mask] = aberr[i]
        logs_gas[mask] = alogs_gas[i]
        chi2[mask] = achi2[i]
        logdgr[mask] = alogdgr[i]
        radius[mask] = aradius[i]
        model_d_sed[mask] = amodel_d_sed[i]
        sed[mask] = ased[i]
        for j in range(len(Ts_f)):
            f_sopt[j][mask] = f_asopt[j][i]
            f_chi2[j][mask] = f_achi2[j][i]
        if ncmode:
            nclogs_d[mask] = ncalogs_d[i]
            ncserr[mask] = ncaserr[i]
            concchi2[mask] = ncachi2[i]
            nccochi2[mask] = nccoachi2[i]
            ncncchi2[mask] = ncncachi2[i]

    if ncmode:
        # Plot these first
        fig, ax = plt.subplots(2, 2, figsize=(18, 14))
        cax = np.empty_like(ax)
        ax[0, 0].set_title(r'$\Sigma_d$ (COV)', size=20)
        cax[0, 0] = ax[0, 0].imshow(logs_d, origin='lower', cmap=cmap0,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        ax[1, 0].set_title(r'$\Sigma_d$ error (COV)', size=20)
        cax[1, 0] = ax[1, 0].imshow(serr, origin='lower', cmap=cmap0,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        ax[0, 1].set_title(r'$\Sigma_d$ (non-COV)', size=20)
        cax[0, 1] = ax[0, 1].imshow(nclogs_d, origin='lower', cmap=cmap0,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        ax[1, 1].set_title(r'$\Sigma_d$ error (non-COV)', size=20)
        cax[1, 1] = ax[1, 1].imshow(ncserr, origin='lower', cmap=cmap0,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        for i in range(2):
            for j in range(2):
                fig.colorbar(cax[i, j], ax=ax[i, j])
                ax[i, j].set_xlabel['r25']
                ax[i, j].set_xlabel['r25']
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
        fig.savefig('output/' + name + '_Sd_conc.png')
        fig.clf()
        #
        fig, ax = plt.subplots(2, 2, figsize=(18, 14))
        cax = np.empty_like(ax)
        ax[0, 0].set_title(r'$\chi^2$ (COV f. + 2D l.f.)', size=20)
        cax[0, 0] = ax[0, 0].imshow(chi2, origin='lower', cmap=cmap0,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        ax[1, 0].set_title(r'$\chi^2$ (COV f. + 1D l.f.)', size=20)
        cax[1, 0] = ax[1, 0].imshow(concchi2, origin='lower', cmap=cmap0,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        ax[0, 1].set_title(r'$\chi^2$ (NC f. + 2D l.f.)', size=20)
        cax[0, 1] = ax[0, 1].imshow(nccochi2, origin='lower', cmap=cmap0,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        ax[1, 1].set_title(r'$\chi^2$ (NC f. + 1D l.f.)', size=20)
        cax[1, 1] = ax[1, 1].imshow(ncncchi2, origin='lower', cmap=cmap0,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        for i in range(2):
            for j in range(2):
                fig.colorbar(cax[i, j], ax=ax[i, j])
                ax[i, j].set_xlabel['r25']
                ax[i, j].set_xlabel['r25']
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
        fig.savefig('output/' + name + '_chi2_conc.png')
        fig.clf()
        del concchi2, ncncchi2, nccochi2, ncachi2, ncncachi2, nccoachi2, \
            nclogs_d, ncserr

    # Plot metallicity vs. DGR map
    plt.close('all')
    mtl = pd.read_csv('output/' + name + '_metal.csv')
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(logdgr, origin='lower', cmap=cmap0)
    ax.scatter(mtl.new_c1, mtl.new_c2, s=100, marker='s', facecolors='none',
               edgecolors='c')
    fig.savefig('output/' + name + '_metallicity_map.png')
    # Plot metallicity vs. DGR
    x, y = np.arange(logdgr.shape[0]), np.arange(logdgr.shape[1])
    x, y = np.meshgrid(x, y)
    mask = ~np.isnan(logdgr)
    points = np.concatenate((x[mask].reshape(-1, 1), y[mask].reshape(-1, 1)),
                            axis=1)
    dgr_d91 = griddata(points, 10**logdgr[mask],
                       (mtl['new_c1'], mtl['new_c2']),
                       method='linear') / 0.0091
    temp_radius = np.empty_like(dgr_d91)
    for i in range(len(dgr_d91)):
        dgr_d91[i] = 10**logdgr[int(mtl['new_c2'].iloc[i]),
                                int(mtl['new_c1'].iloc[i])] / (0.0091 / 1.36)
        temp_radius[i] = ubradius[int(mtl['new_c2'].iloc[i]),
                                  int(mtl['new_c1'].iloc[i])]
    oxygen_abd_rel = 10**(mtl['12+log(O/H)'] - solar_oxygen_bundance)
    oxygen_r25 = mtl['r25']
    #
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.scatter(oxygen_abd_rel, dgr_d91, c='r', s=15, label='Data points')
    ax.plot(oxygen_abd_rel, oxygen_abd_rel, label='x=y')
    ax.plot([10**(8.0 - solar_oxygen_bundance)] * len(dgr_d91), dgr_d91, '-k',
            alpha=0.5, label='8.0')
    ax.plot([10**(8.2 - solar_oxygen_bundance)] * len(dgr_d91), dgr_d91, '-k',
            alpha=0.5, label='8.2')
    ax.set_xlabel(r'$(O/H)/(O/H)_\odot$', size=20)
    ax.set_ylabel(r'$DGR / 0.0067$', size=20)
    ax.legend()
    fig.savefig('output/' + name + '_metallicity_vs.png')
    # DGR & (O/H) vs. Radius
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_yscale('log')
    ax.scatter(oxygen_r25, oxygen_abd_rel, c='r', s=15,
               label=r'$(O/H)/(O/H)_\odot$')
    ax.scatter(temp_radius, dgr_d91, c='b', s=15,
               label=r'$DGR / 0.0067$')
    ax.plot(oxygen_r25, [10**(8.0 - solar_oxygen_bundance)] * len(oxygen_r25),
            '-k', alpha=0.5, label='8.0')
    ax.plot(oxygen_r25, [10**(8.2 - solar_oxygen_bundance)] * len(oxygen_r25),
            '-k', alpha=0.5, label='8.2')
    ax.set_xlabel(r'Radius ($R25$)', size=20)
    ax.set_ylabel(r'$DGR / 0.0067$ or $(O/H)/(O/H)_\odot$', size=20)
    ax.legend()
    fig.savefig('output/' + name + '_metallicity_and.png')
    # DGR & (O/H) vs. Radius twin axis
    boundary_ratio = 1.2
    fig, ax1 = plt.subplots(figsize=(10, 7.5))
    max1 = np.nanmax(oxygen_abd_rel.values) * 5
    max2 = np.nanmax(dgr_d91) * (0.0091 / 1.36)
    min2 = np.nanmin(dgr_d91) * (0.0091 / 1.36)
    min1 = max1 * (min2 / max2)
    ax1.set_yscale('log')
    ax1.scatter(oxygen_r25, oxygen_abd_rel, c='r', s=15,
                label=r'$(O/H)/(O/H)_\odot$')
    ax1.set_xlabel(r'Radius ($R25$)', size=20)
    ax1.set_ylabel(r'$(O/H)/(O/H)_\odot$', size=20, color='r')
    ax1.tick_params('y', colors='r')
    ax1.set_ylim([min1 / boundary_ratio, max1 * boundary_ratio])
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.scatter(temp_radius, dgr_d91 * (0.0091 / 1.36), c='b', s=15,
                label=r'$DGR$')
    ax2.set_ylabel(r'$DGR$', size=20, color='b')
    ax2.tick_params('y', colors='b')
    ax2.set_ylim([min2 / boundary_ratio, max2 * boundary_ratio])
    fig.tight_layout()
    fig.savefig('output/' + name + '_metallicity_twinaxis.png')
    # Fixed beta & temperature fitting results
    fig, ax = plt.subplots(2, len(Ts_f), figsize=(20, 12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    for i in range(len(Ts_f)):
        ax[0, i].set_title(r'$\Sigma_d$ for $T=$' + str(Ts_f[i]) +
                           r' ;$\beta=$' + str(beta_f), size=20)
        cax[0, i] = ax[0, i].imshow(f_sopt[i], origin='lower', cmap=cmap0,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        fig.colorbar(cax[0, i], ax=ax[0, i])
        ax[0, i].set_xlabel('r25')
        ax[0, i].set_ylabel('r25')
        ax[1, i].set_title(r'$\chi^2$', size=20)
        cax[1, i] = ax[1, i].imshow(f_chi2[i], origin='lower', cmap=cmap0,
                                    vmax=60,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        ax[1, i].set_xlabel('r25')
        ax[1, i].set_ylabel('r25')
        fig.colorbar(cax[1, i], ax=ax[1, i])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_fixed_T_beta.png')

    fig, ax = plt.subplots(2, 2, figsize=(20, 12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    ax[0, 0].set_title(r'$\chi^2$ for $T=$' + str(Ts_f[0]) + r' ;$\beta=$' +
                       str(beta_f), size=20)
    cax[0, 0] = ax[0, 0].imshow(f_chi2[0] / 4, origin='lower', cmap=cmap0,
                                vmax=25,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    fig.colorbar(cax[0, 0], ax=ax[0, 0])
    ax[0, 1].set_title(r'$\chi^2$ for $T=$' + str(Ts_f[1]) + r' ;$\beta=$' +
                       str(beta_f), size=20)
    cax[0, 1] = ax[0, 1].imshow(f_chi2[1] / 4, origin='lower', cmap=cmap0,
                                vmax=25,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    fig.colorbar(cax[0, 1], ax=ax[0, 1])
    ax[1, 0].set_title(r'$\chi^2$ for $T=$' + str(Ts_f[2]) + r' ;$\beta=$' +
                       str(beta_f), size=20)
    cax[1, 0] = ax[1, 0].imshow(f_chi2[2] / 4, origin='lower', cmap=cmap0,
                                vmax=25,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    fig.colorbar(cax[1, 0], ax=ax[1, 0])
    ax[1, 1].set_title(r'$\chi^2$ for normal fitting', size=20)
    cax[1, 1] = ax[1, 1].imshow(chi2 / 2, origin='lower', cmap=cmap0, vmax=25,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    fig.colorbar(cax[1, 1], ax=ax[1, 1])
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlabel('r25')
            ax[i, j].set_ylabel('r25')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_fixed_T_beta_chi2.png')

    # Fitting results
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    ax[0, 0].set_title(r'$\Sigma_d$ $(\log_{10}(M_\odot pc^{-2}))$', size=20)
    cax[0, 0] = ax[0, 0].imshow(logs_d, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[1, 0].set_title(r'$\Sigma_d$ error (dex)', size=20)
    cax[1, 0] = ax[1, 0].imshow(serr, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[0, 1].set_title(r'$T_d$ ($K$)', size=20)
    cax[0, 1] = ax[0, 1].imshow(topt, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[1, 1].set_title(r'$T_d$ error ($K$)', size=20)
    cax[1, 1] = ax[1, 1].imshow(terr, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[0, 2].set_title(r'$\beta$', size=20)
    cax[0, 2] = ax[0, 2].imshow(bopt, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[1, 2].set_title(r'$\beta$ error', size=20)
    cax[1, 2] = ax[1, 2].imshow(berr, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    for i in range(2):
        for j in range(3):
            fig.colorbar(cax[i, j], ax=ax[i, j])
            ax[i, j].set_xlabel('r25')
            ax[i, j].set_ylabel('r25')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_Sd_Td.png')
    fig.clf()

    # Best fit models versus observed SEDs
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    cax = np.empty_like(ax)
    titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']
    for i in range(5):
        p, q = i // 3, i % 3
        cax[p, q] = ax[p, q].imshow(model_d_sed[:, :, i], origin='lower',
                                    cmap=cmap2, vmin=0, vmax=2,
                                    extent=[x_ext0, x_ext1, y_ext0, y_ext1])
        fig.colorbar(cax[p, q], ax=ax[p, q])
        ax[p, q].set_xlabel('r25')
        ax[p, q].set_ylabel('r25')
        ax[p, q].set_title(titles[i], size=20)
    fig.tight_layout()
    fig.savefig('output/' + name + '_model_divided_by_SED.png')
    # Beta distribution
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(name, size=28, y=0.995)
    ax[0].set_title(r'$\beta$ vs. $\log(\Sigma_d)$', size=20)
    ax[0].hist2d(alogs_d, abopt, bins=[30, 15], cmap='gist_heat')
    ax[0].set_xlabel(r'$\log(\Sigma_d)$', size=16)
    ax[0].set_ylabel(r'$\beta$', size=16)
    ax[1].set_title(r'$\beta$ vs. $T$', size=20)
    ax[1].hist2d(atopt, abopt, bins=[30, 15], cmap='gist_heat')
    ax[1].set_xlabel(r'$T$', size=16)
    ax[1].set_ylabel(r'$\beta$', size=16)
    ax[2].set_title(r'Histogram of $\beta$', size=20)
    ax[2].hist(abopt, bins=15)
    ax[2].set_xlabel(r'$\beta$', size=16)
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + 'beta.png')
    fig.clf()

    # Total gas & DGR
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    cax[0, 0] = ax[0, 0].imshow(logdgr, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[0, 0].set_title('DGR (log)', size=20)
    cax[0, 1] = ax[0, 1].imshow(logs_gas, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[0, 1].set_title(r'$\Sigma_{gas}$ $(\log_{10}(M_\odot pc^{-2}))$',
                       size=20)
    cax[0, 2] = ax[0, 2].imshow(logs_H2, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[0, 2].set_title(r'HERACLES (log, not binned)', size=20)
    cax[1, 0] = ax[1, 0].imshow(logs_d, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[1, 0].set_title(r'$\Sigma_{d}$ $(\log_{10}(M_\odot pc^{-2}))$', size=20)
    cax[1, 1] = ax[1, 1].imshow(serr, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[1, 1].set_title(r'$\Sigma_d$ error (dex)', size=20)
    cax[1, 2] = ax[1, 2].imshow(chi2, origin='lower', cmap=cmap0,
                                extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    ax[1, 2].set_title(r'$\chi^2$', size=20)
    for i in range(2):
        for j in range(3):
            ax[i, j].set_xlabel('r25')
            ax[i, j].set_ylabel('r25')
            fig.colorbar(cax[i, j], ax=ax[i, j])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_DGR_Sd_GAS.png')
    fig.clf()

    plt.figure()
    plt.imshow(logdgr, alpha=0.8, origin='lower', cmap='gist_heat',
               extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    plt.colorbar()
    plt.imshow(logs_H2, alpha=0.6, origin='lower', cmap='bone',
               extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    plt.xlabel('r25')
    plt.ylabel('r25')
    plt.colorbar()
    plt.title(r'DGR overlay with $H_2$ map', size=24)
    plt.savefig('output/' + name + '_DGR_H2_OL.png')

    # hist2d
    if len(logsigmas.shape) == 2:
        logsigmas = logsigmas[:, 0]
    sigmas = 10**logsigmas
    #
    r, s, w = [], [], []
    for i in range(len(binlist)):
        temp_g = agas[i]
        temp_r = aradius[i]
        mask2 = apdfs[i] > apdfs[i].max() / 1000
        temp_s = sigmas[mask2]
        temp_p = apdfs[i][mask2]
        temp_p /= np.sum(temp_p)
        for j in range(len(temp_s)):
            r.append(temp_r)
            s.append(temp_s[j] / temp_g)
            w.append(temp_g * temp_p[j])
        if i % (len(binlist) // 10) == 0:
            print('Current bin:' + str(i) + ' of ' + str(len(binlist)))
    r, s, w = np.array(r), np.array(s), np.array(w)
    nanmask = np.isnan(r) + np.isnan(s) + np.isnan(w)
    r, s, w = r[~nanmask], s[~nanmask], w[~nanmask]
    rbins = np.linspace(np.min(r), np.max(r), 100)
    sbins = np.logspace(np.min(np.log10(s)), np.max(np.log10(s)), 250)
    counts, _, _ = np.histogram2d(r, s, bins=(rbins, sbins), weights=w)
    counts2, _, _ = np.histogram2d(r, s, bins=(rbins, sbins))
    counts = counts.T
    counts2 = counts2.T
    for i in range(counts.shape[1]):
        if np.sum(counts2[:, i]) > 0:
            counts[:, i] /= np.sum(counts[:, i])
            counts2[:, i] /= np.sum(counts2[:, i])
    plt.close("all")
    """ Generates H2 & HI radial profile """
    mask = ~(np.isnan(ubthings) + np.isnan(ubradius))
    r_HI = ubradius[mask]
    sd_HI = ubthings[mask]
    total_sd, _ = np.histogram(r_HI, rbins, weights=sd_HI)
    count_HI, _ = np.histogram(r_HI, rbins)
    with np.errstate(invalid='ignore'):
        mean_HI = total_sd / count_HI
    #
    mask = ~(np.isnan(ubheracles) + np.isnan(ubradius))
    r_H2 = ubradius[mask]
    sd_H2 = ubheracles[mask]
    total_sd, _ = np.histogram(r_H2, rbins, weights=sd_H2)
    count_H2, _ = np.histogram(r_H2, rbins)
    with np.errstate(invalid='ignore'):
        mean_H2 = total_sd / count_H2
    """ Generate avg fixed beta & temperature radial profile """
    f_dgr = 10**(f_sopt - logs_gas)
    mask = ~(np.isnan(f_dgr[0]) + np.isnan(ubradius))
    r_f = ubradius[mask]
    sd_f = f_dgr[:, mask]
    count_f, _ = np.histogram(r_f, rbins)
    total_sd_f = []
    for i in range(len(Ts_f)):
        temp, _ = np.histogram(r_f, rbins, weights=sd_f[i])
        total_sd_f.append(temp)
    total_sd_f = np.array(total_sd_f)
    with np.errstate(invalid='ignore'):
        mean_f = total_sd_f / count_f
    """ Generates logSFR radial profile """
    SFR = 10**logSFR
    mask = ~(np.isnan(SFR) + np.isnan(ubradius))
    r_SFR = ubradius[mask]
    sd_SFR = SFR[mask]
    total_sd, _ = np.histogram(r_SFR, rbins, weights=sd_SFR)
    count_SFR, _ = np.histogram(r_SFR, rbins)
    with np.errstate(invalid='ignore'):
        mean_SFR = total_sd / count_SFR
    #
    sbins2 = (sbins[:-1] + sbins[1:]) / 2
    rbins2 = (rbins[:-1] + rbins[1:]) / 2
    dgr_median = []
    zeromask = np.full(counts.shape[1], True, dtype=bool)
    #
    plt.figure(figsize=(9, 6))
    plt.semilogy(rbins2, mean_SFR, label='SFR')
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'SFR ($M_\odot/yr$)', size=16)
    plt.title('SFR radial profile', size=20)
    plt.savefig('output/' + name + 'hist2d_SFR.png')
    return 0
    #
    for i in range(counts.shape[1]):
        if np.sum(counts[:, i]) > 0:
            mask = counts[:, i] > (np.max(counts[:, i]) / 1000)
            smax = np.max(counts[mask, i])
            smin = np.min(counts[mask, i])
            csp = np.cumsum(counts[:, i])[:-1]
            csp = np.append(0, csp / csp[-1])
            sss = np.interp([0.16, 0.5, 0.84], csp, sbins2)
            fig, ax = plt.subplots(2, 1)
            ax[0].semilogx([sss[0]] * len(counts[:, i]), counts[:, i],
                           label='16')
            ax[0].semilogx([sss[1]] * len(counts[:, i]), counts[:, i],
                           label='50')
            ax[0].semilogx([sss[2]] * len(counts[:, i]), counts[:, i],
                           label='84')
            # dgr_avg.append(np.sum(counts[:, i] * sbins2))
            # ax[0].semilogx([dgr_avg[-1]] * len(counts[:, i]),
            #                counts[:, i], label='Exp')
            ax[0].semilogx(sbins2, counts[:, i], label='PDF')
            ax[0].set_xlim([smin, smax])
            ax[0].legend()

            dgr_median.append(sss[1])

            ax[1].semilogx(sbins2, counts2[:, i], label='Unweighted PDF')
            ax[1].set_xlim([smin, smax])
            ax[1].legend()
            fig.suptitle(str(np.log10(sss[0])) + '; ' + str(np.log10(sss[1])) +
                         '; ' + str(np.log10(sss[2])))
            fig.savefig('output/' + name + '_' + str(i) + '_' +
                        str(rbins2[i]) + '.png')
            plt.close('all')

        else:
            zeromask[i] = False
    #
    dgr_median = np.array(dgr_median)
    cmap = 'Reds'
    c_median = 'c'
    #
    plt.figure(figsize=(10, 7.5))
    plt.pcolormesh(rbins2, sbins2, counts, norm=LogNorm(), cmap=cmap,
                   vmin=1E-3)
    plt.yscale('log')
    plt.colorbar()
    plt.plot(rbins2[zeromask], dgr_median, c_median, label='Median')
    plt.ylim([1E-5, 1E-1])
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'DGR', size=16)
    plt.title('Gas mass weighted DGR PDF', size=20)
    plt.savefig('output/' + name + 'hist2d_plt_pcolormesh.png')
    # hist2d with metallicity
    fig, ax1 = plt.subplots(figsize=(10, 7.5))
    ax1.pcolor(rbins2, sbins2 / 0.01, counts, norm=LogNorm(),
               cmap=cmap, vmin=1E-3)
    ax1.set_ylim([1E-3, 1E1])
    ax1.set_yscale('log')
    ax1.plot(rbins2[zeromask], dgr_median / 0.01, 'g',
             label='DGR Median / 0.01')
    ax1.set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax1.set_title('DGR PDF vs. Metallicity', size=20)
    ax1.set_ylabel('Ratio', size=16)
    ax1.plot(rbins2,
             10**(8.715 - 0.027 * rbins2 * R25 - solar_oxygen_bundance), 'b',
             label='$(O/H) / (O/H)_\odot$')
    ax1.legend()
    # ax1.set_ylabel(r'$(O/H) / (O/H)_\odot$', size=16, color='b')
    fig.savefig('output/' + name + 'hist2d_plt_pcolormesh_M.png')
    #
    plt.figure(figsize=(9, 6))
    plt.semilogy(rbins2, mean_HI, label='HI')
    plt.semilogy(rbins2, mean_H2, label=r'H$_2$')
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'Surface density', size=16)
    plt.title('Gas surface densities', size=20)
    plt.legend()
    plt.savefig('output/' + name + 'hist2d_plt_HIH2.png')
    #
    plt.figure(figsize=(10, 7.5))
    plt.semilogy(rbins2, mean_H2 / mean_HI, label=r'$H_2$ $/$ $HI$')
    plt.semilogy(rbins2, mean_H2 / (mean_H2 + mean_HI),
                 label=r'$H_2$ $/$ $Total$ $gas$')
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'Ratio', size=16)
    plt.title('Gas ratio', size=20)
    plt.legend()
    plt.savefig('output/' + name + 'H2_ratios.png')

    fig, ax = plt.subplots()
    for i in range(len(Ts_f)):
        ax.semilogy(rbins2, mean_f[i], label='T=' + str(Ts_f[i]))
    ax.set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax.set_ylabel(r'$<\Sigma_d>$', size=16)
    ax.set_title(r'Dust surface densities ($\beta=$' + str(beta_f) + ')',
                 size=20)
    ax.legend()
    fig.savefig('output/' + name + 'hist2d_dust_fixed.png')
    #
    dgr_median_map = radial_map_gen(radius, rbins, dgr_median, zeromask)
    plt.figure()
    with np.errstate(invalid='ignore', divide='ignore'):
        plt.imshow(logdgr - np.log10(dgr_median_map), origin='lower',
                   cmap=cmap0, extent=[x_ext0, x_ext1, y_ext0, y_ext1])
    plt.xlabel('r25')
    plt.ylabel('r25')
    plt.colorbar()
    plt.title(r'$\log(DGR/<DGR>_r)$', size=20)
    plt.savefig('output/' + name + 'DGR_residue.png')

    plt.close("all")
    # Generate temperature gradient
    r, t, w = [], [], []
    for i in range(len(binlist)):
        temp_r = aradius[i]
        temp_g = agas[i]
        mask2 = t_apdfs[i] > t_apdfs[i].max() / 1000
        temp_t = Ts[mask2]
        temp_p = t_apdfs[i][mask2]
        temp_p /= np.sum(temp_p)
        for j in range(len(temp_t)):
            r.append(temp_r)
            t.append(temp_t[j])
            w.append(temp_g * temp_p[j])
        if i % (len(binlist) // 10) == 0:
            print('Current bin:' + str(i) + ' of ' + str(len(binlist)))
    r, t, w = np.array(r), np.array(t), np.array(w)
    nanmask = np.isnan(r) + np.isnan(t) + np.isnan(w)
    r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
    rbins = np.linspace(np.min(r), np.max(r), 100)
    tbins = np.linspace(np.min(t), np.max(t), 50)
    counts, _, _ = np.histogram2d(r, t, bins=(rbins, tbins), weights=w)
    counts2, _, _ = np.histogram2d(r, t, bins=(rbins, tbins))
    counts = counts.T
    counts2 = counts2.T
    for i in range(counts.shape[1]):
        if np.sum(counts2[:, i]) > 0:
            counts[:, i] /= np.sum(counts[:, i])
            counts2[:, i] /= np.sum(counts2[:, i])
    tbins2 = (tbins[:-1] + tbins[1:]) / 2
    rbins2 = (rbins[:-1] + rbins[1:]) / 2
    t_median = []
    zeromask = np.full(counts.shape[1], True, dtype=bool)
    #
    for i in range(counts.shape[1]):
        if np.sum(counts[:, i]) > 0:
            mask = counts[:, i] > (np.max(counts[:, i]) / 1000)
            csp = np.cumsum(counts[:, i])[:-1]
            csp = np.append(0, csp / csp[-1])
            sst = np.interp([0.16, 0.5, 0.84], csp, tbins2)
            t_median.append(sst[1])
        else:
            zeromask[i] = False
    #
    t_median = np.array(t_median)
    cmap = 'Reds'
    c_median = 'c'
    #
    plt.figure(figsize=(10, 7.5))
    plt.pcolormesh(rbins2, tbins2, counts, norm=LogNorm(), cmap=cmap,
                   vmin=1E-3)
    plt.colorbar()
    plt.plot(rbins2[zeromask], t_median, c_median, label='Median')
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'Temperature (K)', size=16)
    plt.title('Gas mass weighted Temperature PDF', size=20)
    fn = 'hist2d_T_pcolormesh_fb.png' if fixed_beta else \
        'hist2d_T_pcolormesh.png'
    plt.savefig('output/' + name + fn)
    plt.close('all')


def plot_single_pixel(name='NGC5457', cmap0='gist_heat'):
    plt.close('all')
    plt.ioff()
    with File('output/' + name + '_dust_data.h5', 'r') as hf:
        ased = np.array(hf['Herschel_SED'])
        acov_n1 = np.array(hf['Herschel_covariance_matrix'])
        binmap = np.array(hf['Binmap'])
        binlist = np.array(hf['Binlist'])
        logsigmas = np.array(hf['logsigmas'])
        apdfs = np.array(hf['PDF'])
        amodel = np.array(hf['Best_fit_model'])
        Ts_f = np.array(hf['Fixed_temperatures'])
        f_amodel = np.array(hf['Fixed_best_fit_model'])

    if Ts_f.ndim == 2:
        Ts_f = Ts_f[:, 0]

    print("Input x coordinate:")
    x = int(input())
    print("Input y coordinate:")
    y = int(input())
    for i in range(len(binlist)):
        mask = binmap == binlist[i]
        if mask[y, x]:
            pn = name + '_bin-' + str(i) + '_x-' + str(x) + '_y-' + str(y)
            plt.imshow(mask, origin='lower')
            plt.savefig('output/' + pn)
            diff = (ased[i] - amodel[i]).reshape(-1, 5)
            chi2 = round(np.dot(np.dot(diff, acov_n1[i]), diff.T)[0, 0], 3)
            # SED vs. fitting
            fig, ax = plt.subplots()
            ax.plot(wl, ased[i], label='Observed')
            ax.plot(wl, amodel[i], label=r'Model, $\chi^2=$' +
                    str(chi2))
            for j in range(len(Ts_f)):
                diff = (ased[i] - f_amodel[j][i]).reshape(-1, 5)
                chi2 = round(np.dot(np.dot(diff, acov_n1[i]), diff.T)[0, 0], 3)
                ax.plot(wl, f_amodel[j][i], label=r'$T=$' +
                        str(int(Ts_f[j])) + r'Model, $\chi^2=$' + str(chi2))
            ax.legend()
            fig.savefig('output/' + pn + '_model_fitting.png')
            # PDF
            pdf = apdfs[i]
            mask = pdf > pdf.max() / 1000
            pdf, logsigmas = pdf[mask], logsigmas[mask]
            fig, ax = plt.subplots()
            ax.plot(logsigmas, pdf)
            ax.set_xlabel(r'$\Sigma_d$')
            ax.set_ylabel('PDF')
            fig.savefig('output/' + pn + '_PDF.png')
            plt.close("all")

"""
def vs_KINGFISH(name='NGC5457', targetSNR=10, dr25=0.025):
    df = pd.DataFrame()

    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        with np.errstate(invalid='ignore'):
            df['dust_fit'] = \
                10**(np.array(grp['Dust_surface_density_log']).flatten())
        serr = np.array(grp['Dust_surface_density_err_dex']).flatten()
        D = float(np.array(grp['Galaxy_distance']))
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        df['dust_kf'] = np.array(grp['KINGFISH']).flatten() / 1E6
        df['snr_kf'] = df['dust_kf'] / \
            np.array(grp['KINGFISH_unc']).flatten() * 1E6
        df['total_gas'] = np.array(grp['Total_gas']).flatten()
        radius = np.array(grp['DP_RADIUS']).flatten()

    R25 = gal_data([name]).field('R25_DEG')[0]
    R25 *= (np.pi / 180.) * (D * 1E3)
    df['r25'] = radius / R25
    ##
    with np.errstate(invalid='ignore', divide='ignore'):
        df['snr_fit'] = 1 / (10**serr - 1.0)
        df['dgr_fit'] = df['dust_fit'] / df['total_gas']
        df['dgr_kf'] = df['dust_kf'] / df['total_gas']
    del radius, serr, R25, D

    # Building a binned KINGFISH map for easier radial profile plotting
    mask = np.isnan(df['snr_kf']) | np.isnan(df['total_gas']) | \
        np.isnan(df['snr_fit']) | np.isnan(df['r25'])
    df = df[~mask]
    assert np.sum(np.isnan(df.values)) == 0

    # Redistributing data to rings
    nlayers = int(np.max(df['r25']) // dr25)
    masks = [(df['r25'] < dr25)]
    for i in range(1, nlayers - 1):
        masks.append((df['r25'] >= i * dr25) & (df['r25'] < (i + 1) * dr25))
    masks.append(df['r25'] >= (nlayers - 1) * dr25)
    masks = np.array(masks)
    r_ri, log_dgr_fit_ri, log_dgr_kf_ri, snr_fit_ri, snr_kf_ri = \
        np.empty(nlayers), np.empty(nlayers), np.empty(nlayers), \
        np.empty(nlayers), np.empty(nlayers)
    for i in range(nlayers):
        mask = masks[i]
        masses = df['total_gas'][mask] / np.nansum(df['total_gas'][mask])
        r_ri[i] = np.sum(df['r25'][mask] * masses)
        with np.errstate(invalid='ignore'):
            log_dgr_fit_ri[i] = np.log10(np.sum(df['dgr_fit'][mask] * masses))
            log_dgr_kf_ri[i] = np.log10(np.sum(df['dgr_kf'][mask] * masses))
        snr_fit_ri[i] = np.sum(df['snr_fit'][mask] * masses)
        snr_kf_ri[i] = np.sum(df['snr_kf'][mask] * masses)

    # Total gas & DGR
    fig, ax = plt.subplots(2, 1, figsize=(10, 14))
    fig.suptitle(name, size=28, y=0.995)
    ax[0].plot(r_ri, log_dgr_fit_ri, label='This work')
    ax[0].plot(r_ri, log_dgr_kf_ri, label='KINGFISH')
    ax[0].set_xlim([np.min(r_ri), np.max(r_ri)])
    ax[0].set_ylabel('Dust-to-gas ratio (log scale)', size=16)
    ax[0].legend(fontsize=16)

    ax[1].plot(r_ri, snr_fit_ri, label='This work')
    ax[1].plot(r_ri, snr_kf_ri, label='KINGFISH')
    ax[1].plot(r_ri, [3]*len(r_ri), 'k')
    ax[1].set_xlim([np.min(r_ri), np.max(r_ri)])
    ax[1].set_ylabel('Mean fitting SNR', size=16)
    ax[1].set_xlabel('r25', size=16)
    ax[1].legend(fontsize=16)
    fig.tight_layout()
    fig.savefig('output/' + name + '_vs_KINGFISH.png')
    fig.clf()

    plt.close("all")
"""


"""
def reject_outliers(data, sig=2.):
    data = data[~np.isnan(data)]
    plt.figure()
    stdev = np.std(data)
    print stdev
    mean = np.mean(data)
    data_temp = data[np.abs(data-mean) < (sig * stdev)]
    plt.subplot(131)
    plt.hist(data_temp, bins=50)
    err1 = np.std(data_temp)

    d = np.abs(data - np.median(data))
    plt.subplot(132)
    plt.hist(d, bins=50)
    mad = np.median(d)
    print mad
    data_temp = data[d < sig * mad]
    plt.subplot(133)
    plt.hist(data_temp, bins=50)
    err2 = np.std(data_temp)
    print err1, err2
"""
