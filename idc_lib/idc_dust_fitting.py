from time import clock
# import emcee
from h5py import File
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import c, h, k_B
# import corner
from . import idc_voronoi


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

THINGS_Limit = 1.0E17  # HERACLES_LIMIT: heracles*2 > things

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
def B(T, freq=nu):
    """Return blackbody SED of temperature T(with unit) in MJy"""
    with np.errstate(over='ignore'):
        return (2 * h * freq**3 / c**2 / (np.exp(h * freq / k_B / T) - 1)
                ).to(u.Jy).value * 1E-6


def model(wl, sigma, T, beta, freq=nu):
    """Return fitted SED in MJy"""
    return const * kappa160 * (160.0 / wl)**beta * \
        sigma * B(T * u.K, freq)
"""
# Remainder of MCMC fitting

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


def fit_dust_density(name, cov_mode=True, fixed_beta=True, beta_f=2.0):
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
        total_gas = np.array(grp['TOTAL_GAS'])
        sed = np.array(grp['HERSCHEL_011111'])
        sed_unc = np.array(grp['HERSCHEL_011111_UNCMAP'])
        bkgcov = np.array(grp['HERSCHEL_011111_BKGCOV'])
        diskmask = np.array(grp['HERSCHEL_011111_DISKMASK'])
        D = float(np.array(grp['DIST_MPC']))
        cosINCL = float(np.array(grp['cosINCL']))
        dp_radius = np.array(grp['RADIUS_KPC'])

    binmap = np.full_like(total_gas, np.nan, dtype=int)
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

    fwhm_radius = fwhm_sp500 * D * 1E3 / cosINCL
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
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    print("Reading/Generating fitting grid...")
    tic = clock()
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
        print(" --Constructing PACS RSRF model...")
        ttic = clock()
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
                pacs_models[:, :, i] = model(pacs_wl[i], 10**logsigmas, Ts,
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
                pacs_models[:, :, :, i] = model(pacs_wl[i], 10**logsigmas, Ts,
                                                betas, pacs_nu[i])
            del pacs_nu
            models[:, :, :, 0] = np.sum(pacs_models * pacs100dnu, axis=ndim) /\
                np.sum(pacs100dnu * pacs_wl / wl[0])
            models[:, :, :, 1] = np.sum(pacs_models * pacs160dnu, axis=ndim) /\
                np.sum(pacs160dnu * pacs_wl / wl[1])
        #
        del pacs_wl, pacs100dnu, pacs160dnu, pacs_models
        print("   --Done. Elapsed time:", round(clock()-ttic, 3), "s.\n")
        ##
        print(" --Constructing SPIRE RSRF model...")
        ttic = clock()
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
                spire_models[:, :, i] = model(spire_wl[i], 10**logsigmas, Ts,
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
                spire_models[:, :, :, i] = model(spire_wl[i], 10**logsigmas,
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
        print("   --Done. Elapsed time:", round(clock()-ttic, 3), "s.\n")
        if fixed_beta:
            fn = 'output/rsrf_models_b_' + str(beta_f) + '.h5'
            with File(fn, 'a') as hf:
                hf.create_dataset('models_fb', data=models)
        else:
            with File('output/rsrf_models.h5', 'a') as hf:
                hf.create_dataset('models', data=models)
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

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
            pacs_models[:, :, i] = model(pacs_wl[i], 10**logsigmas_f, Ts_f,
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
            spire_models[:, :, i] = model(spire_wl[i], 10**logsigmas_f,
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
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.")
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
            print(' --Step', (i + 1), '/', str(len(binlist)) + '.',
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

    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.")
    # Saving to h5 file
    # Total_gas and dust in M_sun/pc**2
    # Temperature in K
    # SED in MJy/sr
    # D in Mpc
    # Galaxy_distance in Mpc
    # Galaxy_center in pixel [y, x]
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
        hf.create_dataset('Radius_avg', data=radius_avg)  # kpc
        hf.create_dataset('logsigmas', data=logsigmas_untouched)
        hf.create_dataset('Ts', data=Ts_untouched)
        hf.create_dataset('betas', data=betas_untouched)
        hf.create_dataset('PDF', data=pdfs)
        hf.create_dataset('PDF_T', data=t_pdfs)
        hf.create_dataset('PDF_B', data=b_pdfs)
        hf.create_dataset('Best_fit_model', data=b_model)
    print("Datasets saved.")
