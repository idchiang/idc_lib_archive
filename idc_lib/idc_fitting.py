from time import clock
from h5py import File
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import astropy.units as u
from astropy.constants import c
from sklearn import linear_model
# from scipy.stats.stats import pearsonr
# import corner
from .idc_voronoi import voronoi_m
from .idc_functions import map2bin, SEMBB, BEMBB, WD
plt.ioff()

# Grid parameters
logsigma_step = 0.025
min_logsigma = -4.
max_logsigma = 1.
T_step = 0.5
min_T = 5.
max_T = 50.
beta_step = 0.1
min_beta = -1.0
max_beta = 4.0
beta2_step = 0.25
min_beta2 = -1.0
max_beta2 = 4.0
lambda_c_step = 25
min_lambda_c = 50
max_lambda_c = 600
WDfrac_step = 0.002
min_WDfrac = 0.0
max_WDfrac = 0.05

# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)

FWHM = {'SPIRE_500': 36.09, 'SPIRE_350': 24.88, 'SPIRE_250': 18.15,
        'Gauss_25': 25, 'PACS_160': 11.18, 'PACS_100': 7.04,
        'HERACLES': 13}
fwhm_sp500 = FWHM['SPIRE_500'] * u.arcsec.to(u.rad)  # in rad

# Calibration error of PACS_100, PACS_160, SPIRE_250, SPIRE_350, SPIRE_500
# For extended source
PAU = 10.0 / 100.0  # PACS Absolute Uncertainty
PRU = 2.0 / 100.0  # PACS Relative Uncertainty
SAU = 8.0 / 100.0  # SPIRE Absolute Uncertainty
SRU = 1.5 / 100.0  # SPIRE Relative Uncertainty
cali_mat2 = np.array([[PAU + PRU, PAU, 0, 0, 0],
                      [PAU, PAU + PRU, 0, 0, 0],
                      [0, 0, SAU + SRU, SAU, SAU],
                      [0, 0, SAU, SAU + SRU, SAU],
                      [0, 0, SAU, SAU, SAU + SRU]])**2
# Calibration error for non-covariance matrix mode
calerr_matrix2 = np.array([PAU + PRU, PAU + PRU, SAU + SRU, SAU + SRU,
                           SAU + SRU]) ** 2
del PAU, PRU, SAU, SRU


def half_integer(num):
    return round(num * 2) / 2


def first_decimal(num):
    return round(num, 1)


def cal_err(masked_dist, masked_pr, ML=None):
    idx = np.argsort(masked_dist)
    sorted_dist = masked_dist[idx]
    sorted_pr = masked_pr[idx]
    csp = np.cumsum(sorted_pr)[:-1]
    csp = np.append(0, csp / csp[-1])
    results = np.interp([0.16, 0.5, 0.84], csp, sorted_dist).tolist()
    if ML is not None:
        results[1] = ML
    return max(results[2] - results[1], results[1] - results[0])


def normalize_pdf(pdf):
    while pdf.ndim > 1:
        pdf = np.sum(pdf, axis=-1)
    return pdf / np.sum(pdf)


def fit_DataY(DataX, DataY, DataY_err, err_bdr=0.3, rbin=51, dbin=250):
    print('Calculating predicted parameter...')
    tic = clock()
    nanmask = np.isnan(np.sum(DataY, axis=1) + DataY_err +
                       np.sum(DataX, axis=1))
    mask = ((DataY_err / DataY.reshape(-1)) < err_bdr).reshape(-1) * ~nanmask
    sigma_n2 = (DataY_err / DataY.reshape(-1))**(-2)
    print('Total number of bins:', mask.size)
    print('Total number of good fits:', mask.sum())
    regr = linear_model.LinearRegression()
    regr.fit(DataX[mask], DataY[mask], sigma_n2[mask])
    coef_ = np.array(regr.coef_[0] + regr.intercept_)
    DataX[nanmask] = np.zeros(DataX.shape[1])
    DataY_pred = regr.predict(DataX)
    DataY_pred[nanmask] = np.nan
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
    return DataY_pred.reshape(-1), coef_


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


def fit_dust_density(name='NGC5457', cov_mode=True, beta_f=2.0,
                     method_abbr='FB', del_model=False):
    """
    Should have unique names for all the methods I am currently using:
        EF, FB, FBPT, PB, BEMFB
    """
    targetSN = 5
    ndims = {'EF': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4, 'FBWD': 3,
             'BEMFBFL': 3}
    try:
        ndims[method_abbr]
    except KeyError:
        raise KeyError("The input method \"" + method_abbr +
                       "\" is not supported yet!")
    if cov_mode is None:
        print('COV mode? (1 for COV, 0 for non-COV)')
        cov_mode = bool(int(input()))
    print('################################################')
    print('   ' + name + '-' + method_abbr + ' fitting')
    print('################################################')
    # Dust density in Solar Mass / pc^2
    # kappa_lambda in cm^2 / g
    # SED in MJy / sr
    with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
        grp = hf['Regrid']
        total_gas = grp['TOTAL_GAS'].value
        sed = grp['HERSCHEL_011111'].value
        sed_unc = grp['HERSCHEL_011111_UNCMAP'].value
        bkgcov = grp['HERSCHEL_011111_BKGCOV'].value
        diskmask = grp['HERSCHEL_011111_DISKMASK'].value
        nwl = len(bkgcov)
        D = grp['DIST_MPC'].value
        cosINCL = grp['cosINCL'].value
        dp_radius = grp['RADIUS_KPC'].value

    binmap = np.full_like(total_gas, np.nan, dtype=int)
    # Voronoi binning
    # d --> diskmasked, len() = sum(diskmask);
    # b --> binned, len() = number of binned area
    try:
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Bin']
            binlist = grp['BINLIST'].value
            binmap = grp['BINMAP'].value
            gas_avg = grp['GAS_AVG'].value
            sed_avg = grp['Herschel_SED'].value
            cov_n1s = grp['Herschel_covariance_matrix'].value
            inv_sigma2s = grp['Herschel_variance'].value
            radius_avg = grp['Radius_avg'].value
    except KeyError:
        print("Start binning " + name + "...")
        tic = clock()
        noise4snr = np.array([np.sqrt(bkgcov[i, i]) for i in range(nwl)])
        diskmask *= ~np.isnan(np.sum(sed, axis=2))
        temp_snr = sed[diskmask] / noise4snr
        signal_d = np.array([np.min(temp_snr[i]) for i in
                             range(len(temp_snr))])
        signal_d[signal_d < 0] = 0
        del temp_snr
        noise_d = np.ones(signal_d.shape)
        x_d, y_d = np.meshgrid(range(sed.shape[1]), range(sed.shape[0]))
        x_d, y_d = x_d[diskmask], y_d[diskmask]
        # Dividing into layers
        judgement = np.sum(signal_d) / np.sqrt(len(signal_d))
        if judgement < targetSN:
            print(name, 'is having just too small overall SNR. Will not fit')
        fwhm_radius = fwhm_sp500 * D * 1E3 / cosINCL
        nlayers = int(np.nanmax(dp_radius) // fwhm_radius)
        masks = []
        with np.errstate(invalid='ignore'):
            masks.append(dp_radius < fwhm_radius)
            for i in range(1, nlayers - 1):
                masks.append((dp_radius >= i * fwhm_radius) *
                             (dp_radius < (i + 1) * fwhm_radius))
            masks.append(dp_radius >= (nlayers - 1) * fwhm_radius)
        # test image: original layers
        testimage1 = np.full_like(dp_radius, np.nan)
        for i in range(nlayers):
            testimage1[masks[i]] = np.sin(i)
        testimage1[~diskmask] = np.nan
        #
        for i in range(nlayers - 1, -1, -1):
            judgement = np.sum(signal_d[masks[i][diskmask]]) / \
                np.sqrt(len(masks[i][diskmask]))
            if judgement < targetSN:
                if i > 0:
                    masks[i - 1] += masks[i]
                    del masks[i]
                else:
                    masks[0] += masks[1]
                    del masks[1]
        nlayers = len(masks)
        # test image: combined layers #
        testimage2 = np.full_like(dp_radius, np.nan)
        for i in range(nlayers):
            testimage2[masks[i]] = np.sin(i)
        testimage2[~diskmask] = np.nan
        #######################################
        masks = [masks[i][diskmask] for i in range(nlayers)]
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
                    voronoi_m(x_l, y_l, signal_l, noise_l, targetSN,
                              pixelsize=1, plot=False, quiet=True)
            binNum_l += max_binNum
            max_binNum = np.max(binNum_l) + 1
            binNum[masks[i]] = binNum_l

        for i in range(len(signal_d)):
            binmap[y_d[i], x_d[i]] = binNum[i]
        binlist = np.unique(binNum)
        temp_snr = sed / noise4snr
        testimage0 = np.empty_like(testimage1, dtype=float)
        for i in range(sed.shape[0]):
            for j in range(sed.shape[1]):
                testimage0[i, j] = \
                    temp_snr[i, j][np.argmin(np.abs(temp_snr[i, j]))]
        testimage0[~diskmask] = np.nan
        testimage3 = np.sin(binmap)
        testimage3[~diskmask] = np.nan
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        cax = ax[0, 0].imshow(testimage0, origin='lower', cmap='gist_heat',
                              vmax=5)
        fig.colorbar(cax, ax=ax[0, 0])
        ax[0, 0].set_title('Worst SNR')
        ax[0, 1].imshow(testimage1, origin='lower', cmap='jet')
        ax[0, 1].set_title('Original radial bins')
        ax[1, 0].imshow(testimage2, origin='lower', cmap='jet')
        ax[1, 0].set_title('Final radial bins')
        ax[1, 1].imshow(testimage3, origin='lower', cmap='jet')
        ax[1, 1].set_title('Final bins')
        fig.tight_layout()
        pp = PdfPages('output/_Voronnoi' + name + '.pdf')
        pp.savefig(fig)
        pp.close()
        pp = PdfPages('hdf5_MBBDust/' + name + '_Voronnoi.pdf')
        pp.savefig(fig)
        pp.close()
        plt.close('all')
        del testimage0, testimage1, testimage2, testimage3
        radius_avg, gas_avg, sed_avg, cov_n1s, inv_sigma2s = [], [], [], [], []
        for b in binlist:
            bin_ = binmap == b
            radius_avg.append(np.nansum(dp_radius[bin_] * total_gas[bin_]) /
                              np.nansum(total_gas[bin_]))
            gas_avg.append(np.nanmean(total_gas[bin_]))
            # mean sed
            sed_avg.append(np.nanmean(sed[bin_], axis=0))
            unc2_avg = np.mean(sed_unc[bin_]**2, axis=0)
            unc2_avg[np.isnan(unc2_avg)] = 0
            # Covariance mode
            # bkg Covariance matrix
            bkgcov_avg = bkgcov / np.sum(bin_)
            # uncertainty diagonal matrix
            unc2cov_avg = np.identity(nwl) * unc2_avg
            # calibration error covariance matrix
            sed_vec = sed_avg[-1].reshape(1, nwl)
            calcov = sed_vec.T * cali_mat2 * sed_vec
            # Finally everything for covariance matrix is here...
            cov_n1 = np.linalg.inv(bkgcov_avg + unc2cov_avg + calcov)
            cov_n1s.append(cov_n1)
            # Non-covariance matrix mode (old fashion)
            # 1D bkgerr
            bkg2_avg = (bkgcov / np.sum(bin_)).diagonal()
            # calibration error covariance matrix
            calerr2 = calerr_matrix2 * sed_avg[-1]**2
            # Finally everything for variance is here...
            inv_sigma2s.append(1 / (bkg2_avg + calerr2 + unc2_avg))
        with File('hdf5_MBBDust/' + name + '.h5', 'a') as hf:
            grp = hf.create_group('Bin')
            grp['BINLIST'] = binlist
            grp['BINMAP'] = binmap
            grp['GAS_AVG'] = gas_avg
            grp['Herschel_SED'] = sed_avg
            grp['Herschel_covariance_matrix'] = cov_n1s
            grp['Herschel_variance'] = inv_sigma2s
            grp['Radius_avg'] = radius_avg  # kpc
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
        print("Reading/Generating fitting grid...")
    """ Loading Data for prediction """
    if method_abbr in ['FBPT', 'PB']:
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Fitting_results']
            if method_abbr in ['PB']:
                try:
                    subgrp = grp['EF']
                except KeyError:
                    fit_dust_density(name, cov_mode, beta_f, 'EF')
                    subgrp = grp['EF']
                DataX = radius_avg.reshape(-1, 1)
                DataY = subgrp['beta'].value.reshape(-1, 1)
                DataY_err = subgrp['beta_err'].value
                beta_pred, coef_ = fit_DataY(DataX, DataY, DataY_err)
                beta_pred = np.array(list(map(first_decimal, beta_pred)))
            if method_abbr in ['FBPT']:
                try:
                    subgrp = grp['FB']
                except KeyError:
                    fit_dust_density(name, cov_mode, beta_f, 'FB')
                    subgrp = grp['FB']
                DataY = subgrp['Dust_temperature'].value.reshape(-1, 1)
                DataY_err = subgrp['Dust_temperature_err'].value
                grp = hf['Regrid']
                with np.errstate(invalid='ignore'):
                    logSFR = \
                        np.log10(map2bin(grp['SFR'].value, binlist, binmap))
                    logSMSD = \
                        np.log10(map2bin(grp['SMSD'].value, binlist, binmap))
                alogSFR = np.array([logSFR[binmap == binlist[i]][0] for i in
                                    range(len(binlist))])
                alogSMSD = np.array([logSMSD[binmap == binlist[i]][0] for i in
                                     range(len(binlist))])
                DataX = np.array([alogSFR, alogSMSD]).T
                del logSFR, logSMSD, alogSFR, alogSMSD
                t_pred, coef_ = fit_DataY(DataX, DataY, DataY_err)
                t_pred = np.array(list(map(half_integer, t_pred)))
        del DataX, DataY, DataY_err
    #
    tic = clock()
    """ Grid parameters """
    """
    ndims = {'EF': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4}
    'EF': np.meshgrid(logsigmas, Ts, betas)
    'FB': np.meshgrid(logsigmas, Ts)
    'BEMFB': np.meshgrid(logsigmas, Ts, lambda_cs, beta2s)
    """
    logsigmas = np.arange(min_logsigma, max_logsigma, logsigma_step)
    logsigmas_1d = np.arange(min_logsigma, max_logsigma, logsigma_step)
    Ts = np.arange(min_T, max_T, T_step)
    Ts_1d = np.arange(min_T, max_T, T_step)
    betas = beta_f
    if method_abbr == 'EF':
        betas_1d = np.arange(min_beta, max_beta, beta_step)
        logsigmas, Ts, betas = np.meshgrid(logsigmas, Ts, betas_1d)
    elif method_abbr == 'FB':
        logsigmas, Ts = np.meshgrid(logsigmas, Ts)
    elif method_abbr == 'FBPT':
        Ts_1d = np.unique(t_pred)
        logsigmas, Ts = np.meshgrid(logsigmas, Ts_1d)
    elif method_abbr == 'PB':
        betas_1d = np.unique(beta_pred)
        logsigmas, betas, Ts = np.meshgrid(logsigmas, betas_1d, Ts)
    elif method_abbr in ['BEMFB', 'BEMFBFL']:
        beta2s_1d = np.arange(min_beta2, max_beta2, beta2_step)
        lambda_cs_1d = np.arange(min_lambda_c, max_lambda_c, lambda_c_step)
        logsigmas, Ts, lambda_cs, beta2s = \
            np.meshgrid(logsigmas, Ts, lambda_cs_1d, beta2s_1d)
    elif method_abbr == 'FBWD':
        WDfracs_1d = np.arange(min_WDfrac, max_WDfrac, WDfrac_step)
        logsigmas, Ts, WDfracs = np.meshgrid(logsigmas, Ts, WDfracs_1d)
    #
    if del_model:
        with File('hdf5_MBBDust/Models.h5', 'a') as hf:
            del hf[method_abbr]
    try:
        if method_abbr == 'BEMFBFL':
            with File('hdf5_MBBDust/Models.h5', 'r') as hf:
                models_ = hf['BEMFB'].value
        else:
            with File('hdf5_MBBDust/Models.h5', 'r') as hf:
                models_ = hf[method_abbr].value
    except (OSError, KeyError):
        models_ = np.zeros(list(Ts.shape) + [5])
        # Applying RSRFs to generate fake-observed models
        print(" --Constructing PACS RSRF model...")
        ttic = clock()
        pacs_rsrf = pd.read_csv("data/RSRF/PACS_RSRF.csv")
        pacs_wl = pacs_rsrf['Wavelength'].values
        pacss = [pacs_rsrf['PACS_100'].values,
                 pacs_rsrf['PACS_160'].values]
        del pacs_rsrf
        #
        pacs_models = np.zeros(list(Ts.shape) + [len(pacs_wl)])
        if method_abbr in ['BEMFB', 'BEMFBFL']:
            for i in range(len(pacs_wl)):
                pacs_models[..., i] = BEMBB(pacs_wl[i], 10**logsigmas, Ts,
                                            betas, lambda_cs, beta2s)
        elif method_abbr in ['FBWD']:
            for i in range(len(pacs_wl)):
                pacs_models[..., i] = WD(pacs_wl[i], 10**logsigmas, Ts,
                                         betas, WDfracs)
        else:
            for i in range(len(pacs_wl)):
                pacs_models[..., i] = SEMBB(pacs_wl[i], 10**logsigmas, Ts,
                                            betas)
        for i in range(0, 2):
            models_[..., i] = \
                np.sum(pacs_models * pacss[i], axis=-1) / \
                np.sum(pacss[i] * pacs_wl / wl[i])
        del pacs_wl, pacss, pacs_models
        print("   --Done. Elapsed time:", round(clock()-ttic, 3), "s.\n")
        ##
        print(" --Constructing SPIRE RSRF model...")
        ttic = clock()
        spire_rsrf = pd.read_csv("data/RSRF/SPIRE_RSRF.csv")
        spire_wl = spire_rsrf['Wavelength'].values
        spires = [[], [], spire_rsrf['SPIRE_250'].values,
                  spire_rsrf['SPIRE_350'].values,
                  spire_rsrf['SPIRE_500'].values]
        del spire_rsrf
        #
        spire_models = np.zeros(list(Ts.shape) + [len(spire_wl)])
        if method_abbr in ['BEMFB', 'BEMFBFL']:
            for i in range(len(spire_wl)):
                spire_models[..., i] = BEMBB(spire_wl[i], 10**logsigmas,
                                             Ts, betas, lambda_cs, beta2s)
        elif method_abbr in ['FBWD']:
            for i in range(len(spire_wl)):
                spire_models[..., i] = WD(spire_wl[i], 10**logsigmas,
                                          Ts, betas, WDfracs)
        else:
            for i in range(len(spire_wl)):
                spire_models[..., i] = SEMBB(spire_wl[i], 10**logsigmas,
                                             Ts, betas)
        for i in range(2, 5):
            models_[..., i] = \
                np.sum(spire_models * spires[i], axis=-1) / \
                np.sum(spires[i] * spire_wl / wl[i])
        del spire_wl, spires, spire_models
        if method_abbr == 'BEMFBFL':
            with File('hdf5_MBBDust/Models.h5', 'a') as hf:
                hf['BEMFB'] = models_
        else:
            with File('hdf5_MBBDust/Models.h5', 'a') as hf:
                hf[method_abbr] = models_
    """
    Start fitting
    """
    print("Start fitting", name, "dust surface density...")
    tic = clock()
    progress = 0
    sopt, serr, topt, terr, bopt, berr = [], [], [], [], [], []
    pdfs, t_pdfs, b_pdfs = [], [], []
    b2opt, b2err, lcopt, lcerr = [], [], [], []
    b2_pdfs, lc_pdfs = [], []
    Wfopt, Wferr, Wf_pdfs = [], [], []
    achi2, ased_fit = [], []
    if method_abbr in ['FBPT']:
        logsigmas = logsigmas[Ts_1d == Ts_1d[0]][0]
    elif method_abbr in ['PB']:
        logsigmas = logsigmas[betas_1d == betas_1d[0]][0]
        Ts = Ts[betas_1d == betas_1d[0]][0]
    elif method_abbr in ['BEMFBFL']:
        lambda_cf = 300.0
        models = models_[:, :, lambda_cs_1d == lambda_cf, :]
        logsigmas = logsigmas[:, :, lambda_cs_1d == lambda_cf, :]
        Ts = Ts[:, :, lambda_cs_1d == lambda_cf, :]
        beta2s = beta2s[:, :, lambda_cs_1d == lambda_cf, :]
        print(models.shape)
        del models_
    else:
        models = models_
        del models_
    for i in range(len(binlist)):
        if (i + 1) / len(binlist) > progress:
            print(' --Step', (i + 1), '/', str(len(binlist)) + '.',
                  "Elapsed time:", round(clock()-tic, 3), "s.")
            progress += 0.1
        """ Binning everything """
        if method_abbr in ['FBPT']:
            models = models_[Ts_1d == t_pred[i]][0]
        elif method_abbr in ['PB']:
            models = models_[betas_1d == beta_pred[i]][0]
        temp_matrix = np.empty_like(models)
        bin_ = (binmap == binlist[i])
        if cov_mode:
            diff = models - sed_avg[i]
            for j in range(nwl):
                temp_matrix[..., j] = np.sum(diff * cov_n1s[i][:, j], axis=-1)
            chi2 = np.sum(temp_matrix * diff, axis=-1)
        else:
            chi2 = np.sum((sed_avg[i] - models)**2 * inv_sigma2s[i], axis=-1)
        """ Find the (s, t) that gives Maximum likelihood """
        am_idx = np.unravel_index(chi2.argmin(), chi2.shape)
        achi2.append(np.nanmin(chi2))
        ased_fit.append(models[am_idx])
        """
        method_abbrs = {'EF': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4}
        Special care in \Sigma_D: FBPT(1d)
        Special care in Temperature: FBPT
        Special care in \beta: PB(b_ML), EF(err)
        """
        """ Probability and mask """
        mask = chi2 < achi2[-1] + 6.0
        pr = np.exp(-0.5 * chi2)
        pr_cp = pr[mask]
        """ \Sigma_D """
        s_ML = logsigmas[am_idx]
        sopt.append(s_ML)
        serr.append(cal_err(logsigmas[mask], pr_cp, s_ML))
        if method_abbr in ['FBPT', 'PB']:
            pdfs.append(normalize_pdf(pr))
        else:
            pdfs.append(normalize_pdf(np.sum(pr, axis=0)))
        """ T_D """
        if method_abbr in ['FBPT']:
            topt.append(t_pred[i])
            terr.append(0.0)
        else:
            t_ML = Ts[am_idx]
            topt.append(t_ML)
            terr.append(cal_err(Ts[mask], pr_cp, t_ML))
            if method_abbr in ['PB']:
                t_pdfs.append(normalize_pdf(np.sum(pr, axis=0)))
            else:
                t_pdfs.append(normalize_pdf(np.sum(pr, axis=1)))
        """ \beta """
        if method_abbr in ['EF']:
            b_ML = betas[am_idx]
        elif method_abbr in ['PB']:
            b_ML = beta_pred[i]
        else:
            b_ML = betas
        bopt.append(b_ML)
        if method_abbr in ['EF']:
            berr.append(cal_err(betas[mask], pr_cp, b_ML))
            b_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1))))
        else:
            berr.append(0.0)
        """ BEMBB model related """
        if method_abbr in ['BEMFB', 'BEMFBFL']:
            """ \lambda_c """
            if method_abbr == 'BEMFB':
                lc_ML = lambda_cs[am_idx]
                lcopt.append(lc_ML)
                lcerr.append(cal_err(lambda_cs[mask], pr_cp, lc_ML))
                lc_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1))))
            elif method_abbr == 'BEMFBFL':
                lcopt.append(lambda_cf)
                lcerr.append(0)
            """ \beta2 """
            b2_ML = beta2s[am_idx]
            b2opt.append(b2_ML)
            b2err.append(cal_err(beta2s[mask], pr_cp, b2_ML))
            b2_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1, 2))))
        """ Warm Dust related """
        if method_abbr in ['FBWD']:
            """ WDfrac """
            Wf_ML = WDfracs[am_idx]
            Wfopt.append(Wf_ML)
            Wferr.append(cal_err(WDfracs[mask], pr_cp, Wf_ML))
            Wf_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1))))
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.")
    # Saving to h5 file
    # Total_gas and dust in M_sun/pc**2
    # Temperature in K
    # SED in MJy/sr
    # D in Mpc
    # Galaxy_distance in Mpc
    # Galaxy_center in pixel [y, x]
    with File('hdf5_MBBDust/' + name + '.h5', 'a') as hf:
        grp = hf.require_group('Fitting_results')
        try:
            del grp[method_abbr]
        except KeyError:
            pass
        subgrp = grp.create_group(method_abbr)
        subgrp['Dust_surface_density_log'] = sopt
        subgrp['Dust_surface_density_err_dex'] = serr  # serr in dex
        subgrp['Dust_temperature'] = topt
        subgrp['Dust_temperature_err'] = terr
        subgrp['beta'] = bopt
        subgrp['beta_err'] = berr
        subgrp['logsigmas'] = logsigmas_1d
        subgrp['PDF'] = pdfs
        subgrp['Chi2'] = achi2
        subgrp['Best_fit_sed'] = ased_fit
        subgrp['Ts'] = Ts_1d
        if method_abbr in ['EF', 'FB', 'BEMFB', 'PB', 'FBWD', 'BEMFBFL']:
            subgrp['PDF_T'] = t_pdfs
        if method_abbr in ['EF']:
            subgrp['PDF_B'] = b_pdfs
            subgrp['betas'] = betas_1d
        elif method_abbr in ['PB']:
            subgrp['betas'] = betas_1d
        if method_abbr in ['FBPT', 'PB']:
            subgrp['coef_'] = coef_
        if method_abbr in ['BEMFB', 'BEMFBFL']:
            subgrp['beta2s'] = beta2s_1d
            subgrp['Critical_wavelength'] = lcopt
            subgrp['Critical_wavelength_err'] = lcerr
            if method_abbr == 'BEMFB':
                subgrp['PDF_lc'] = lc_pdfs
                subgrp['lambda_cs'] = lambda_cs_1d
            subgrp['beta2'] = b2opt
            subgrp['beta2_err'] = b2err
            subgrp['PDF_b2'] = b2_pdfs
        if method_abbr in ['FBWD']:
            subgrp['WDfracs'] = WDfracs_1d
            subgrp['WDfrac'] = Wfopt
            subgrp['WDfrac_err'] = Wferr
            subgrp['PDF_Wf'] = Wf_pdfs
    print("Datasets saved.")
