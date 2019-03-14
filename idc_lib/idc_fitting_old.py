from time import clock, ctime
from h5py import File
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import astropy.units as u
from astropy.constants import c, N_A
# from corner import corner
from scipy.stats import pearsonr
# from scipy.stats.stats import pearsonr
from .idc_voronoi import voronoi_m
from .idc_functions import map2bin, SEMBB, BEMBB, WD, PowerLaw, B_fast
from .idc_functions import WDT, MWT, Umax
from .z0mg_RSRF import z0mg_RSRF
plt.ioff()
my_beta_f = 2.0

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
alpha_step = 0.1  # Remember to avoid alpha==1
min_alpha = 1.1
max_alpha = 3.0
loggamma_step = 0.2
min_loggamma = -4
max_loggamma = 0
logUmin_step = 0.1
min_logUmin = -2.
max_logUmin = 1.5

# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])

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

ndims = {'SE': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4, 'WD': 3,
         'BE': 3, 'PL': 4}
parallel_rounds = {'SE': 3, 'FB': 1, 'BE': 3, 'WD': 3, 'PL': 12}


def half_integer(num):
    return round(num * 2) / 2


def first_decimal(num):
    return round(num, 1)


def cal_err(masked_dist, masked_pr, ML=None):
    idx = np.argsort(masked_dist)
    sorted_dist = masked_dist[idx]
    sorted_pr = masked_pr[idx]
    csp = np.cumsum(sorted_pr)
    csp = csp / csp[-1]
    results = np.interp([0.16, 0.5, 0.84], csp, sorted_dist).tolist()
    if ML is not None:
        results[1] = ML
    return max(results[2] - results[1], results[1] - results[0])


def normalize_pdf(pdf):
    while pdf.ndim > 1:
        pdf = np.sum(pdf, axis=-1)
    return pdf / np.sum(pdf)


def fit_DataY(DataX, DataY, DataY_err, err_bdr=0.3, quiet=False):
    print('Calculating predicted parameter... (' + ctime() + ')')
    tic = clock()
    nanmask = np.isnan(DataY + DataY_err + DataX)
    mask = ((DataY_err / DataY) < err_bdr) * ~nanmask
    if not quiet:
        print('Number of nan points:', np.sum(nanmask))
        print('Number of negative yerr:', np.sum(DataY_err < 0))
        print('Number of zero yerr:', np.sum(DataY_err == 0))
        print('Number of available data:', mask.sum())
        print('Correlation between DataX and DataY:',
              pearsonr(DataX[mask], DataY[mask])[0])
    popt, pcov = np.polyfit(x=DataX, y=DataY, deg=1, w=1/DataY_err, cov=True)
    perr = np.sqrt(np.diag(pcov))
    if not quiet:
        print('Fitting coef:', popt)
        print('Fitting err :', perr)
    DataX[nanmask] = 0.0
    DataY_pred = DataX * popt[0] + popt[1]
    DataY_pred[nanmask] = np.nan
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
    return DataY_pred, popt


def cal_and_print_err(var, idx, mask, pr_cp, varname):
    ML = var[idx]
    err = cal_err(var[mask], pr_cp, ML)
    print('Best fit', varname + ':', ML)
    print('    ' + ' ' * len(varname) + 'error:', err)


def exp_and_error(var, pr, varname=None, quiet=True):
    Z = np.sum(pr)
    expected = np.sum(var * pr) / Z
    err = cal_err(var.flatten(), pr.flatten(), expected)
    if not quiet:
        print('Expected', varname + ':', expected)
        print('    ' + ' ' * len(varname) + 'error:', err)
    return expected, err


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


def fit_dust_density(name='NGC5457', cov_mode=True, beta_f=my_beta_f,
                     lambda_c_f=300.0, method_abbr='FB', del_model=False,
                     fake=False, nop=5, targetSN=5):
    """
    Should have unique names for all the methods I am currently using:
        EF, FB, BE, WD, PL
    """
    try:
        with File('hdf5_MBBDust/Calibration.h5', 'r') as hf:
            grp = hf[method_abbr]
            kappa160 = grp['kappa160'].value
    except KeyError:
        print('This method is not calibrated yet!! Starting calibration...')
        kappa_calibration(method_abbr, beta_f=beta_f, lambda_c_f=lambda_c_f,
                          nop=nop)
        with File('hdf5_MBBDust/Calibration.h5', 'r') as hf:
            grp = hf[method_abbr]
            kappa160 = grp['kappa160'].value
    if cov_mode is None:
        print('COV mode? (1 for COV, 0 for non-COV)')
        cov_mode = bool(int(input()))
    print('################################################')
    print('   ' + name + '-' + method_abbr + ' fitting (' + ctime() + ')')
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
        ps = grp['SPIRE_500_PS'].value
        if fake:
            grp = hf['Fake']
            subgrp = grp['SED']
            sed = subgrp[method_abbr].value
    ps = np.mean(ps)
    resolution_element = np.pi * (FWHM['SPIRE_500'] / 2)**2 / ps**2
    binmap = np.full_like(total_gas, np.nan, dtype=int)
    # Voronoi binning
    # d --> diskmasked, len() = sum(diskmask);
    # b --> binned, len() = number of binned area
    try:
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Bin']
            if fake:
                grp1 = hf['Fake']
                grp2 = grp1['Bin']
                grp = grp2[method_abbr]
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
        """
        testimage1 = np.full_like(dp_radius, np.nan)
        for i in range(nlayers):
            testimage1[masks[i]] = np.sin(i)
        testimage1[~diskmask] = np.nan
        """
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
        """
        testimage2 = np.full_like(dp_radius, np.nan)
        for i in range(nlayers):
            testimage2[masks[i]] = np.sin(i)
        testimage2[~diskmask] = np.nan
        """
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
        """
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
        if fake:
            fn = '_FAKE_Voronnoi' + name + '_' + method_abbr + '.pdf'
        else:
            fn = '_Voronnoi' + name + '.pdf'
        pp = PdfPages('output/' + fn)
        pp.savefig(fig)
        pp.close()
        pp = PdfPages('hdf5_MBBDust/' + fn)
        pp.savefig(fig)
        pp.close()
        plt.close('all')
        del testimage0, testimage1, testimage2, testimage3
        """
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
            num_res = np.sum(bin_) / resolution_element
            if num_res > 1:
                bkgcov_avg = bkgcov / num_res
            else:
                bkgcov_avg = bkgcov
            # uncertainty diagonal matrix
            # unc2cov_avg = np.identity(nwl) * unc2_avg
            # calibration error covariance matrix
            sed_vec = sed_avg[-1].reshape(1, nwl)
            calcov = sed_vec.T * cali_mat2 * sed_vec
            # Finally everything for covariance matrix is here...
            # cov_n1 = np.linalg.inv(bkgcov_avg + unc2cov_avg + calcov)
            cov_n1 = np.linalg.inv(bkgcov_avg + calcov)
            cov_n1s.append(cov_n1)
            # Non-covariance matrix mode (old fashion)
            # 1D bkgerr
            bkg2_avg = (bkgcov / np.sum(bin_)).diagonal()
            # calibration error covariance matrix
            calerr2 = calerr_matrix2 * sed_avg[-1]**2
            # Finally everything for variance is here...
            inv_sigma2s.append(1 / (bkg2_avg + calerr2 + unc2_avg))
        with File('hdf5_MBBDust/' + name + '.h5', 'a') as hf:
            grp = hf.require_group('Bin')
            if fake:
                grp1 = hf['Fake']
                grp2 = grp1.require_group('Bin')
                grp = grp2.require_group(method_abbr)
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
                    subgrp = grp['SE']
                except KeyError:
                    fit_dust_density(name, cov_mode, beta_f, 'SE')
                    subgrp = grp['SE']
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
    ndims = {'SE': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4}
    'SE': np.meshgrid(logsigmas, Ts, betas)
    'FB': np.meshgrid(logsigmas, Ts)
    'BEMFB': np.meshgrid(logsigmas, Ts, lambda_cs, beta2s)
    """
    logsigmas_1d = np.arange(min_logsigma, max_logsigma, logsigma_step)
    Ts_1d = np.arange(min_T, max_T, T_step)
    betas = beta_f
    if method_abbr == 'PL':
        alphas_1d = np.arange(min_alpha, max_alpha, alpha_step)
        loggammas_1d = np.arange(min_loggamma, max_loggamma, loggamma_step)
        logUmins_1d = np.arange(min_logUmin, max_logUmin, logUmin_step)
        logsigmas, alphas, loggammas, logUmins = \
            np.meshgrid(logsigmas_1d, alphas_1d, loggammas_1d, logUmins_1d)
    else:
        betas = beta_f
    if method_abbr == 'SE':
        betas_1d = np.arange(min_beta, max_beta, beta_step)
        logsigmas, Ts, betas = np.meshgrid(logsigmas_1d, Ts_1d, betas_1d)
    elif method_abbr == 'FB':
        logsigmas, Ts = np.meshgrid(logsigmas_1d, Ts_1d)
    elif method_abbr == 'FBPT':
        Ts_1d = np.unique(t_pred)
        logsigmas, Ts = np.meshgrid(logsigmas_1d, Ts_1d)
    elif method_abbr == 'PB':
        betas_1d = np.unique(beta_pred)
        logsigmas, betas, Ts = np.meshgrid(logsigmas_1d, betas_1d, Ts_1d)
    elif method_abbr == 'BEMFB':
        beta2s_1d = np.arange(min_beta2, max_beta2, beta2_step)
        lambda_cs_1d = np.arange(min_lambda_c, max_lambda_c, lambda_c_step)
        logsigmas, Ts, lambda_cs, beta2s = \
            np.meshgrid(logsigmas_1d, Ts_1d, lambda_cs_1d, beta2s_1d)
    elif method_abbr == 'BE':
        beta2s_1d = np.arange(min_beta2, max_beta2, beta2_step)
        logsigmas, Ts, beta2s = \
            np.meshgrid(logsigmas_1d, Ts_1d, beta2s_1d)
        lambda_cs = np.full(Ts.shape, lambda_c_f)
    elif method_abbr == 'WD':
        WDfracs_1d = np.arange(min_WDfrac, max_WDfrac, WDfrac_step)
        logsigmas, Ts, WDfracs = np.meshgrid(logsigmas_1d, Ts_1d, WDfracs_1d)
    #
    if method_abbr in ['WD', 'PL']:
        Teff_bins = np.append(Ts_1d, Ts_1d[-1] + T_step)
        Teff_bins -= T_step / 2.
        if method_abbr == 'WD':
            Teffs = Ts * (1 - WDfracs) + WDT * WDfracs
        elif method_abbr == 'PL':
            Umins = 10**logUmins
            gammas = 10**loggammas
            Ubar = np.empty_like(Umins)
            mask = alphas == 2
            Ubar[mask] = gammas[mask] * \
                (alphas[mask] - 1) / (alphas[mask] - 2) * \
                (Umax**(2 - alphas[mask]) - Umins[mask]**(2 - alphas[mask])) /\
                (Umax**(1 - alphas[mask]) - Umins[mask]**(1 - alphas[mask])) +\
                (1 - gammas[mask]) * Umins[mask]
            Ubar[~mask] = Umins[~mask] * \
                ((1 - gammas[~mask]) + gammas[~mask] *
                 np.log(Umax / Umins[~mask]) / (1 - Umins[~mask] / Umax))
            del mask, Umins, gammas
            Teffs = MWT * Ubar**(1 / 6.)
            del Ubar
    #
    if del_model:
        with File('hdf5_MBBDust/Models.h5', 'a') as hf:
            try:
                del hf[method_abbr]
            except KeyError:
                pass
    try:
        with File('hdf5_MBBDust/Models.h5', 'r') as hf:
            models_ = hf[method_abbr].value
    except (OSError, KeyError):
        if method_abbr in ['BEMFB', 'BE']:
            def fitting_model(wl):
                return BEMBB(wl, 10**logsigmas, Ts, betas, lambda_cs, beta2s,
                             kappa160=kappa160)
        elif method_abbr in ['WD']:
            def fitting_model(wl):
                return WD(wl, 10**logsigmas, Ts, betas, WDfracs,
                          kappa160=kappa160)
        elif method_abbr in ['PL']:
            def fitting_model(wl):
                return PowerLaw(wl, 10**logsigmas, alphas, 10**loggammas,
                                logUmins, beta=betas, kappa160=kappa160)
        else:
            def fitting_model(wl):
                return SEMBB(wl, 10**logsigmas, Ts, betas,
                             kappa160=kappa160)

        def split_herschel(ri, r_, rounds, _wl, wlr, output):
            tic = clock()
            rw = ri + r_ * nop
            lenwls = wlr[rw + 1] - wlr[rw]
            last_time = clock()
            result = np.zeros(list(logsigmas.shape) + [lenwls])
            print("   --process", ri, "starts... (" + ctime() + ") (round",
                  (r_ + 1), "of", str(rounds) + ")")
            for i in range(lenwls):
                result[..., i] = fitting_model(_wl[i + wlr[rw]])
                current_time = clock()
                # print progress every 10 mins
                if current_time > last_time + 600.:
                    last_time = current_time
                    print("     --process", ri,
                          str(round(100. * (i + 1) / lenwls, 1)) +
                          "% Done. (round", (r_ + 1), "of", str(rounds) + ")")
            output.put((ri, result))
            print("   --process", ri, "Done. Elapsed time:",
                  round(clock()-tic, 3), "s. (" + ctime() + ")")

        models_ = np.zeros(list(logsigmas.shape) + [5])
        timeout = 1e-6
        # Applying RSRFs to generate fake-observed models
        instrs = ['pacs', 'spire']
        rounds = parallel_rounds[method_abbr]
        for instr in range(2):
            print(" --Constructing", instrs[instr], "RSRF model... (" +
                  ctime() + ")")
            ttic = clock()
            _rsrf = pd.read_csv("data/RSRF/" + instrs[instr] + "_rsrf.csv")
            _wl = _rsrf['Wavelength'].values
            h_models = np.zeros(list(logsigmas.shape) + [len(_wl)])
            wlr = [int(ri * len(_wl) / float(nop * rounds)) for ri in
                   range(nop * rounds + 1)]
            if instr == 0:
                rsps = [_rsrf['pacs100'].values,
                        _rsrf['pacs160'].values]
                range_ = range(0, 2)
            elif instr == 1:
                rsps = [[], [], _rsrf['spire250'].values,
                        _rsrf['spire350'].values,
                        _rsrf['spire500'].values]
                range_ = range(2, 5)
            del _rsrf
            # Parallel code
            for r_ in range(rounds):
                print("\n   --" + method_abbr, instrs[instr] + ":Round",
                      (r_ + 1), "of", rounds, '\n')
                q = mp.Queue()
                processes = [mp.Process(target=split_herschel,
                             args=(ri, r_, rounds, _wl, wlr, q))
                             for ri in range(nop)]
                for p in processes:
                    p.start()
                for p in processes:
                    p.join(timeout)
                for p in processes:
                    ri, result = q.get()
                    print("     --Got result from process", ri)
                    rw = ri + r_ * nop
                    h_models[..., wlr[rw]:wlr[rw+1]] = result
                    del ri, result
                del processes, q
            # Parallel code ends
            print("   --Calculating response function integrals")
            for i in range_:
                models_[..., i] = \
                    np.sum(h_models * rsps[i], axis=-1) / \
                    np.sum(rsps[i] * _wl / wl[i])
            del _wl, rsps, h_models, range_
            print("   --Done. Elapsed time:", round(clock()-ttic, 3), "s.\n")
        """
        # version before parallel computing
        models_ = np.zeros(list(logsigmas.shape) + [5])
        # Applying RSRFs to generate fake-observed models
        print(" --Constructing PACS RSRF model... (" + ctime() + ')')
        ttic = clock()
        pacs_rsrf = pd.read_csv("data/RSRF/PACS_RSRF.csv")
        pacs_wl = pacs_rsrf['Wavelength'].values
        pacss = [pacs_rsrf['PACS_100'].values,
                 pacs_rsrf['PACS_160'].values]
        del pacs_rsrf
        #
        pacs_models = np.zeros(list(logsigmas.shape) + [len(pacs_wl)])
        if method_abbr in ['BEMFB', 'BE']:
            for i in range(len(pacs_wl)):
                pacs_models[..., i] = BEMBB(pacs_wl[i], 10**logsigmas, Ts,
                                            betas, lambda_cs, beta2s,
                                            kappa160=kappa160)
        elif method_abbr in ['WD']:
            for i in range(len(pacs_wl)):
                pacs_models[..., i] = WD(pacs_wl[i], 10**logsigmas, Ts,
                                         betas, WDfracs, kappa160=kappa160)
        elif method_abbr in ['PL']:
            for i in range(len(pacs_wl)):
                pacs_models[..., i] = PowerLaw(pacs_wl[i], 10**logsigmas,
                                               alphas, 10**loggammas, logUmins,
                                               kappa160=kappa160)
        else:
            for i in range(len(pacs_wl)):
                pacs_models[..., i] = SEMBB(pacs_wl[i], 10**logsigmas, Ts,
                                            betas, kappa160=kappa160)
        for i in range(0, 2):
            models_[..., i] = \
                np.sum(pacs_models * pacss[i], axis=-1) / \
                np.sum(pacss[i] * pacs_wl / wl[i])
        del pacs_wl, pacss, pacs_models
        print("   --Done. Elapsed time:", round(clock()-ttic, 3), "s.\n")
        ##
        print(" --Constructing SPIRE RSRF model... (" + ctime() + ')')
        ttic = clock()
        spire_rsrf = pd.read_csv("data/RSRF/SPIRE_RSRF.csv")
        spire_wl = spire_rsrf['Wavelength'].values
        spires = [[], [], spire_rsrf['SPIRE_250'].values,
                  spire_rsrf['SPIRE_350'].values,
                  spire_rsrf['SPIRE_500'].values]
        del spire_rsrf
        #
        spire_models = np.zeros(list(logsigmas.shape) + [len(spire_wl)])
        if method_abbr in ['BEMFB', 'BE']:
            for i in range(len(spire_wl)):
                spire_models[..., i] = BEMBB(spire_wl[i], 10**logsigmas,
                                             Ts, betas, lambda_cs, beta2s,
                                             kappa160=kappa160)
        elif method_abbr in ['WD']:
            for i in range(len(spire_wl)):
                spire_models[..., i] = WD(spire_wl[i], 10**logsigmas,
                                          Ts, betas, WDfracs,
                                          kappa160=kappa160)
        elif method_abbr in ['PL']:
            for i in range(len(spire_wl)):
                spire_models[..., i] = PowerLaw(spire_wl[i], 10**logsigmas,
                                                alphas, 10**loggammas,
                                                logUmins, kappa160=kappa160)
        else:
            for i in range(len(spire_wl)):
                spire_models[..., i] = SEMBB(spire_wl[i], 10**logsigmas,
                                             Ts, betas, kappa160=kappa160)
        for i in range(2, 5):
            models_[..., i] = \
                np.sum(spire_models * spires[i], axis=-1) / \
                np.sum(spires[i] * spire_wl / wl[i])
        del spire_wl, spires, spire_models
        """
        with File('hdf5_MBBDust/Models.h5', 'a') as hf:
            hf[method_abbr] = models_
    """
    Start fitting
    """
    print("Start fitting", name, "dust surface density... (" + ctime() + ')')
    tic = clock()
    progress = 0
    sopt, serr, topt, terr, bopt, berr = [], [], [], [], [], []
    pdfs, t_pdfs, b_pdfs = [], [], []
    b2opt, b2err, lcopt, lcerr = [], [], [], []
    b2_pdfs, lc_pdfs = [], []
    Wfopt, Wferr, Wf_pdfs = [], [], []
    achi2, ased_fit = [], []
    gopt, gerr, g_pdfs = [], [], []
    aopt, aerr, a_pdfs = [], [], []
    uopt, uerr, u_pdfs = [], [], []
    teff_pdfs = []
    if method_abbr in ['FBPT']:
        logsigmas = logsigmas[Ts_1d == Ts_1d[0]][0]
    elif method_abbr in ['PB']:
        logsigmas = logsigmas[betas_1d == betas_1d[0]][0]
        Ts = Ts[betas_1d == betas_1d[0]][0]
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
        del am_idx
        """
        method_abbrs = {'SE': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4}
        Special care in Sigma_D: FBPT(1d)
        Special care in Temperature: FBPT
        Special care in beta: PB(b_ML), EF(err)
        """
        """ Probability and mask """
        pr = np.exp(-0.5 * chi2)
        """ Sigma_D """
        sexp, s_err = exp_and_error(logsigmas, pr)
        sopt.append(sexp)
        serr.append(s_err)
        if method_abbr in ['FBPT', 'PB']:
            pdfs.append(normalize_pdf(pr))
        else:
            pdfs.append(normalize_pdf(np.sum(pr, axis=0)))
        """ T_D """
        if method_abbr in ['FBPT']:
            topt.append(t_pred[i])
            terr.append(0.0)
        elif method_abbr in ['PL']:
            pass
        else:
            texp, t_err = exp_and_error(Ts, pr)
            topt.append(texp)
            terr.append(t_err)
            if method_abbr in ['PB']:
                t_pdfs.append(normalize_pdf(np.sum(pr, axis=0)))
            else:
                t_pdfs.append(normalize_pdf(np.sum(pr, axis=1)))
        """ Effective temperature for WD and PL """
        if method_abbr in ['WD', 'PL']:
            teff_pdf = np.histogram(Teffs, bins=Teff_bins, weights=pr)[0]
            teff_pdfs.append(teff_pdf / np.sum(teff_pdf))
        """ \beta """
        if method_abbr not in ['PL']:
            if method_abbr in ['SE']:
                bexp, b_err = exp_and_error(betas, pr)
            elif method_abbr in ['PB']:
                bexp = beta_pred[i]
            else:
                bexp = betas
            bopt.append(bexp)
            if method_abbr in ['SE']:
                berr.append(b_err)
                b_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1))))
            else:
                berr.append(0.0)
        """ BEMBB model related """
        if method_abbr in ['BEMFB', 'BE']:
            """ lambda_c """
            if method_abbr == 'BEMFB':
                lcexp, lc_err = exp_and_error(lambda_cs, pr)
                lcopt.append(lcexp)
                lcerr.append(lc_err)
                lc_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1))))
            elif method_abbr == 'BE':
                lcopt.append(lambda_c_f)
                lcerr.append(0)
            """ \beta2 """
            b2exp, b2_err = exp_and_error(beta2s, pr)
            b2opt.append(b2exp)
            b2err.append(b2_err)
            b2_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1, 2))))
        """ Warm Dust related """
        if method_abbr in ['WD']:
            """ WDfrac """
            Wfexp, Wf_err = exp_and_error(WDfracs, pr)
            Wfopt.append(Wfexp)
            Wferr.append(Wf_err)
            Wf_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1))))
        """ U distribution related """
        if method_abbr in ['PL']:
            """ alpha """
            aexp, a_err = exp_and_error(alphas, pr)
            aopt.append(aexp)
            aerr.append(a_err)
            a_pdfs.append(normalize_pdf(np.sum(pr, axis=1)))
            """ gamma """
            gexp, g_err = exp_and_error(loggammas, pr)
            gopt.append(gexp)
            gerr.append(g_err)
            g_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1))))
            """ Umin """
            uexp, u_err = exp_and_error(logUmins, pr)
            uopt.append(uexp)
            uerr.append(u_err)
            u_pdfs.append(normalize_pdf(np.sum(pr, axis=(0, 1, 2))))
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
        if fake:
            grp1 = hf['Fake']
            grp = grp1.require_group('Fitting_results')
        try:
            del grp[method_abbr]
        except KeyError:
            pass
        subgrp = grp.create_group(method_abbr)
        subgrp['Dust_surface_density_log'] = sopt
        subgrp['Dust_surface_density_err_dex'] = serr  # serr in dex
        if method_abbr not in ['PL']:
            subgrp['Dust_temperature'] = topt
            subgrp['Dust_temperature_err'] = terr
            subgrp['beta'] = bopt
            subgrp['beta_err'] = berr
        subgrp['logsigmas'] = logsigmas_1d
        subgrp['PDF'] = pdfs
        subgrp['Chi2'] = achi2
        subgrp['Best_fit_sed'] = ased_fit
        if method_abbr not in ['PL']:
            subgrp['Ts'] = Ts_1d
        if method_abbr in ['SE', 'FB', 'BEMFB', 'PB', 'WD', 'BE']:
            subgrp['PDF_T'] = t_pdfs
        if method_abbr in ['WD', 'PL']:
            subgrp['PDF_Teff'] = teff_pdfs
            subgrp['Teff_bins'] = Teff_bins
        if method_abbr in ['SE']:
            subgrp['PDF_B'] = b_pdfs
            subgrp['betas'] = betas_1d
        elif method_abbr in ['PB']:
            subgrp['betas'] = betas_1d
        if method_abbr in ['FBPT', 'PB']:
            subgrp['coef_'] = coef_
        if method_abbr in ['BEMFB', 'BE']:
            subgrp['beta2s'] = beta2s_1d
            subgrp['Critical_wavelength'] = lcopt
            subgrp['Critical_wavelength_err'] = lcerr
            if method_abbr == 'BEMFB':
                subgrp['PDF_lc'] = lc_pdfs
                subgrp['lambda_cs'] = lambda_cs_1d
            subgrp['beta2'] = b2opt
            subgrp['beta2_err'] = b2err
            subgrp['PDF_b2'] = b2_pdfs
        if method_abbr in ['WD']:
            subgrp['WDfracs'] = WDfracs_1d
            subgrp['WDfrac'] = Wfopt
            subgrp['WDfrac_err'] = Wferr
            subgrp['PDF_Wf'] = Wf_pdfs
        if method_abbr in ['PL']:
            subgrp['alphas'] = alphas_1d
            subgrp['alpha'] = aopt
            subgrp['alpha_err'] = aerr
            subgrp['PDF_a'] = a_pdfs
            subgrp['loggammas'] = loggammas_1d
            subgrp['loggamma'] = gopt
            subgrp['loggamma_err'] = gerr
            subgrp['PDF_g'] = g_pdfs
            subgrp['logUmins'] = logUmins_1d
            subgrp['logUmin'] = uopt
            subgrp['logUmin_err'] = uerr
            subgrp['PDF_u'] = u_pdfs
    print("Datasets saved.")


def kappa_calibration(method_abbr, beta_f=my_beta_f, lambdac_f=300.0,
                      cov_mode=5, nop=6, quiet=True):
    MWSED = np.array([0.71, 1.53, 1.08, 0.56, 0.25]) * 0.97
    # Correct mode should be 100-Sum_square with Fixen values, or 5
    # Karl's method is 4
    mode_titles = ['100-Sum_Square', 'PS-Sum_Square',
                   '100-Square_Sum', 'PS-Square_Sum',
                   'all-Square_Sum', '100-Sum_Square_F']
    print('################################################')
    print('    Calibrating ' + method_abbr + ' (' + mode_titles[cov_mode] +
          ') (' + ctime() + ')')
    print('################################################')
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
    WDfrac_step = 0.002
    min_WDfrac = 0.0
    max_WDfrac = 0.05
    alpha_step = 0.1  # Remember to avoid alpha==1
    min_alpha = 1.1
    max_alpha = 3.0
    loggamma_step = 0.2
    min_loggamma = -4
    max_loggamma = 0
    logUmin_step = 0.1
    min_logUmin = -2.
    max_logUmin = 1.5
    # Due to memory limit
    if method_abbr == 'PL':
        min_logsigma = -2.
        max_logsigma = 0.
        min_logUmin = -1.
        max_logUmin = 0.
        max_loggamma = -1.
    #
    # MW measurement dataset
    #
    """
    # cov_modes that I am not going to use
    if cov_mode < 2:
        COU = 5.0 / 100.0
        UNU = 2.5 / 100.0
        if cov_mode == 0:  # 100-Sum_Square
            MWcali_mat2 = np.array([[UNU, 0, 0, 0, 0],
                                    [0, COU + UNU, COU, COU, COU],
                                    [0, COU, COU + UNU, COU, COU],
                                    [0, COU, COU, COU + UNU, COU],
                                    [0, COU, COU, COU, COU + UNU]])**2
        else:  # PS-Sum_Square
            MWcali_mat2 = np.array([[COU + UNU, COU, 0, 0, 0],
                                    [COU, COU + UNU, 0, 0, 0],
                                    [0, 0, COU + UNU, COU, COU],
                                    [0, 0, COU, COU + UNU, COU],
                                    [0, 0, COU, COU, COU + UNU]])**2
    elif cov_mode < 4:
        COU = (5.0 / 100.0)**2
        UNU = (2.5 / 100.0)**2
        if cov_mode == 2:  # 100-Square_Sum
            MWcali_mat2 = np.array([[UNU, 0, 0, 0, 0],
                                    [0, COU + UNU, COU, COU, COU],
                                    [0, COU, COU + UNU, COU, COU],
                                    [0, COU, COU, COU + UNU, COU],
                                    [0, COU, COU, COU, COU + UNU]])
        else:  # PS-Square_Sum
            MWcali_mat2 = np.array([[COU + UNU, COU, 0, 0, 0],
                                    [COU, COU + UNU, 0, 0, 0],
                                    [0, 0, COU + UNU, COU, COU],
                                    [0, 0, COU, COU + UNU, COU],
                                    [0, 0, COU, COU, COU + UNU]])
    elif cov_mode == 4:
        COU = (5.0 / 100.0)**2
        UNU = (2.5 / 100.0)**2
        MWcali_mat2 = np.array([[COU + UNU, COU, COU, COU, COU],
                                [COU, COU + UNU, COU, COU, COU],
                                [COU, COU, COU + UNU, COU, COU],
                                [COU, COU, COU, COU + UNU, COU],
                                [COU, COU, COU, COU, COU + UNU]])
    """
    if cov_mode == 5:
        DCOU = 10.0 / 100.0
        DUNU = 1.0 / 100.0
        FCOU = 2.0 / 100.0
        FUNU = 0.5 / 100.0
        MWcali_mat2 = np.array([[DUNU + DCOU, 0, 0, 0, 0],
                                [0, FCOU + FUNU, FCOU, FCOU, FCOU],
                                [0, FCOU, FCOU + FUNU, FCOU, FCOU],
                                [0, FCOU, FCOU, FCOU + FUNU, FCOU],
                                [0, FCOU, FCOU, FCOU, FCOU + FUNU]])**2

    MWSigmaD = (1e20 * 1.0079 * u.g / N_A.value).to(u.M_sun).value * \
        ((1 * u.pc).to(u.cm).value)**2 / 150.
    #
    # Build fitting grid
    #
    for iter_ in range(2):
        logsigmas_1d = np.arange(min_logsigma, max_logsigma, logsigma_step)
        betas = beta_f
        if method_abbr == 'PL':
            alphas_1d = np.arange(min_alpha, max_alpha, alpha_step)
            logUmins_1d = np.arange(min_logUmin, max_logUmin, logUmin_step)
            loggammas_1d = np.arange(min_loggamma, max_loggamma, loggamma_step)
            logsigmas, alphas, loggammas, logUmins = \
                np.meshgrid(logsigmas_1d, alphas_1d, loggammas_1d, logUmins_1d)
        else:
            Ts_1d = np.arange(min_T, max_T, T_step)
        if method_abbr == 'SE':
            betas_1d = np.arange(min_beta, max_beta, beta_step)
            logsigmas, Ts, betas = np.meshgrid(logsigmas_1d, Ts_1d, betas_1d)
        elif method_abbr == 'FB':
            logsigmas, Ts = np.meshgrid(logsigmas_1d, Ts_1d)
        elif method_abbr == 'BE':
            beta2s_1d = np.arange(min_beta2, max_beta2, beta2_step)
            logsigmas, Ts, beta2s = \
                np.meshgrid(logsigmas_1d, Ts_1d, beta2s_1d)
            lambda_cs = np.full(Ts.shape, lambdac_f)
        elif method_abbr == 'WD':
            WDfracs_1d = np.arange(min_WDfrac, max_WDfrac, WDfrac_step)
            logsigmas, Ts, WDfracs = np.meshgrid(logsigmas_1d, Ts_1d,
                                                 WDfracs_1d)
        sigmas = 10**logsigmas
        #
        # Build models
        #
        models_ = np.zeros(list(logsigmas.shape) + [5])
        # Applying RSRFs to generate fake-observed models
        if method_abbr in ['BEMFB', 'BE']:
            def fitting_model(wl):
                return BEMBB(wl, sigmas, Ts, betas, lambda_cs, beta2s,
                             kappa160=1.)
        elif method_abbr in ['WD']:
            def fitting_model(wl):
                return WD(wl, sigmas, Ts, betas, WDfracs,
                          kappa160=1.)
        elif method_abbr in ['PL']:
            def fitting_model(wl):
                return PowerLaw(wl, sigmas, alphas, 10**loggammas,
                                logUmins, beta=betas, kappa160=1.)
        else:
            def fitting_model(wl):
                return SEMBB(wl, sigmas, Ts, betas,
                             kappa160=1.)

        def split_herschel(ri, r_, rounds, _wl, wlr, output):
            tic = clock()
            rw = ri + r_ * nop
            lenwls = wlr[rw + 1] - wlr[rw]
            last_time = clock()
            result = np.zeros(list(logsigmas.shape) + [lenwls])
            if not quiet:
                print("   --process", ri, "starts... (" + ctime() + ") (round",
                      (r_ + 1), "of", str(rounds) + ")")
            for i in range(lenwls):
                result[..., i] = fitting_model(_wl[i + wlr[rw]])
                current_time = clock()
                # print progress every 10 mins
                if (current_time > last_time + 600.) and (not quiet):
                    last_time = current_time
                    print("     --process", ri,
                          str(round(100. * (i + 1) / lenwls, 1)) +
                          "% Done. (round", (r_ + 1), "of", str(rounds) + ")")
            output.put((ri, rw, result))
            if not quiet:
                print("   --process", ri, "Done. Elapsed time:",
                      round(clock()-tic, 3), "s. (" + ctime() + ")")

        models_ = np.zeros(list(logsigmas.shape) + [5])
        timeout = 1e-6
        # Applying RSRFs to generate fake-observed models
        instrs = ['pacs', 'spire']
        rounds = parallel_rounds[method_abbr]
        for instr in range(2):
            if not quiet:
                print(" --Constructing", instrs[instr], "RSRF model... (" +
                      ctime() + ")")
            ttic = clock()
            _rsrf = pd.read_csv("data/RSRF/" + instrs[instr] + "_rsrf.csv")
            _wl = _rsrf['wavelength'].values
            h_models = np.zeros(list(logsigmas.shape) + [len(_wl)])
            wlr = [int(ri * len(_wl) / float(nop * rounds)) for ri in
                   range(nop * rounds + 1)]
            if instr == 0:
                rsps = [_rsrf['pacs100'].values,
                        _rsrf['pacs160'].values]
                range_ = range(0, 2)
            elif instr == 1:
                rsps = [[], [], _rsrf['spire250'].values,
                        _rsrf['spire350'].values,
                        _rsrf['spire500'].values]
                range_ = range(2, 5)
            del _rsrf
            # Parallel code
            for r_ in range(rounds):
                if not quiet:
                    print("\n   --" + method_abbr, instrs[instr] + ":Round",
                          (r_ + 1), "of", rounds, '\n')
                q = mp.Queue()
                processes = [mp.Process(target=split_herschel,
                             args=(ri, r_, rounds, _wl, wlr, q))
                             for ri in range(nop)]
                for p in processes:
                    p.start()
                for p in processes:
                    p.join(timeout)
                for p in processes:
                    ri, rw, result = q.get()
                    if not quiet:
                        print("     --Got result from process", ri)
                    h_models[..., wlr[rw]:wlr[rw+1]] = result
                    del ri, rw, result
                del processes, q, p
            # Parallel code ends
            if not quiet:
                print("   --Calculating response function integrals")
            for i in range_:
                models_[..., i] = \
                    np.sum(h_models * rsps[i], axis=-1) / \
                    np.sum(rsps[i] * _wl / wl[i])
            del _wl, rsps, h_models, range_
            if not quiet:
                print("   --Done. Elapsed time:", round(clock()-ttic, 3),
                      "s.\n")
        #
        # Start fitting
        #
        tic = clock()
        models = models_
        del models_
        temp_matrix = np.empty_like(models)
        diff = models - MWSED
        sed_vec = MWSED.reshape(1, 5)
        yerr = MWSED * np.sqrt(np.diagonal(MWcali_mat2))
        cov_n1 = np.linalg.inv(sed_vec.T * MWcali_mat2 * sed_vec)
        for j in range(5):
            temp_matrix[..., j] = np.sum(diff * cov_n1[:, j], axis=-1)
        chi2 = np.sum(temp_matrix * diff, axis=-1)
        r_chi2 = chi2 / (5. - ndims[method_abbr])
        """ Find the (s, t) that gives Maximum likelihood """
        am_idx = np.unravel_index(chi2.argmin(), chi2.shape)
        """ Probability and mask """
        mask = r_chi2 <= np.nanmin(r_chi2) + 50.
        pr = np.exp(-0.5 * chi2)
        print('\nIteration', str(iter_ + 1))
        print('Best fit r_chi^2:', r_chi2[am_idx])
        """ kappa 160 """
        logkappa160s = logsigmas - np.log10(MWSigmaD)
        logkappa160, logkappa160_err = \
            exp_and_error(logkappa160s, pr, 'logkappa_160')
        kappa160 = 10**logkappa160
        logsigma, _ = exp_and_error(logsigmas, pr, 'logsigmas')
        #
        min_logsigma = logsigma - 0.2
        max_logsigma = logsigma + 0.2
        # All steps
        logsigma_step = 0.002
        T_step = 0.1
        beta_step = 0.02
        beta2_step = 0.02
        WDfrac_step = 0.0005
        alpha_step = 0.01  # Remember to avoid alpha==1
        loggamma_step = 0.1
        logUmin_step = 0.01
        print('Best fit kappa160:', kappa160)
        wl_complete = np.linspace(1, 1000, 1000)
        bands = ['PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', 'SPIRE_500']
        hf = File('hdf5_MBBDust/Calibration_' + str(round(beta_f, 2)) + '.h5',
                  'a')
        try:
            del hf[method_abbr]
        except KeyError:
            pass
        grp = hf.create_group(method_abbr)
        grp['kappa160'] = kappa160
        grp['logkappa160'], grp['logkappa160_err'] = \
            logkappa160, logkappa160_err
        if method_abbr == 'SE':
            samples = np.array([logkappa160s[mask], Ts[mask], betas[mask],
                                r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$T$', r'$\beta$',
                      r'$\tilde{\chi}^2$']
            T, T_err = exp_and_error(Ts, pr, 'T')
            beta, beta_err = exp_and_error(betas, pr, 'beta')
            min_T = T - 1.5
            max_T = T + 1.5
            min_beta = beta - 0.3
            max_beta = beta + 0.3
            grp['T'], grp['T_err'] = T, T_err
            grp['beta'], grp['beta_err'] = beta, beta_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, T, beta,
                                             kappa160=kappa160), bands)
            model_complete = SEMBB(wl_complete, MWSigmaD, T, beta,
                                   kappa160=kappa160)
            gordon_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, 17.2, 1.96,
                                             9.6 * np.pi), bands)
            model_gordon = SEMBB(wl_complete, MWSigmaD, 17.2,
                                 1.96, 9.6 * np.pi)
        elif method_abbr == 'FB':
            samples = np.array([logkappa160s[mask], Ts[mask], r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$T$', r'$\tilde{\chi}^2$']
            T, T_err = exp_and_error(Ts, pr, 'T')
            min_T = T - 1.5
            max_T = T + 1.5
            grp['T'], grp['T_err'] = T, T_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, T, beta_f,
                                             kappa160=kappa160), bands)
            model_complete = SEMBB(wl_complete, MWSigmaD, T, beta_f,
                                   kappa160=kappa160)
            gordon_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, 17.2, 1.96,
                                             9.6 * np.pi), bands)
            model_gordon = SEMBB(wl_complete, MWSigmaD, 17.2,
                                 1.96, 9.6 * np.pi)
        elif method_abbr == 'BE':
            samples = np.array([logkappa160s[mask], Ts[mask], beta2s[mask],
                                r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$T$', r'$\beta_2$',
                      r'$\tilde{\chi}^2$']
            T, T_err = exp_and_error(Ts, pr, 'T')
            beta2, beta2_err = exp_and_error(beta2s, pr, 'beta2')
            min_T = T - 1.5
            max_T = T + 1.5
            min_beta2 = beta2 - 0.3
            max_beta2 = beta2 + 0.3
            grp['T'], grp['T_err'] = T, T_err
            grp['beta2'], grp['beta2_err'] = beta2, beta2_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, BEMBB(wl_complete, MWSigmaD, T, beta_f,
                                             lambdac_f, beta2,
                                             kappa160=kappa160), bands)
            model_complete = BEMBB(wl_complete, MWSigmaD, T, beta_f,
                                   lambdac_f, beta2, kappa160=kappa160)
            e500 = 0.48
            gbeta2 = np.log(e500 + 1) / np.log(294. / 500.) + 2.27
            gordon_integrated = \
                z0mg_RSRF(wl_complete, BEMBB(wl_complete, MWSigmaD, 16.8, 2.27,
                                             294, gbeta2,
                                             11.6 * np.pi), bands)
            model_gordon = BEMBB(wl_complete, MWSigmaD, 16.8, 2.27, 294,
                                 gbeta2, 11.6 * np.pi)
        elif method_abbr == 'WD':
            samples = np.array([logkappa160s[mask], Ts[mask], WDfracs[mask],
                                r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$T$', r'WDfrac',
                      r'$\tilde{\chi}^2$']
            T, T_err = exp_and_error(Ts, pr, 'T')
            WDfrac, WDfrac_err = exp_and_error(WDfracs, pr, 'WDfrac')
            min_T = T - 1.5
            max_T = T + 1.5
            min_WDfrac = 0.0
            max_WDfrac = WDfrac + 0.006
            grp['T'], grp['T_err'] = T, T_err
            grp['WDfrac'], grp['WDfrac_err'] = WDfrac, WDfrac_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, WD(wl_complete, MWSigmaD, T, beta_f,
                                          WDfrac, kappa160=kappa160), bands)
            model_complete = WD(wl_complete, MWSigmaD, T, beta_f, WDfrac,
                                kappa160=kappa160)
            e500 = 0.91
            nu500 = (c / 500 / u.um).to(u.Hz).value
            gWDfrac = e500 * B_fast(15., nu500) / B_fast(6., nu500)
            gordon_integrated = \
                z0mg_RSRF(wl_complete, WD(wl_complete, MWSigmaD, 15., 2.9,
                                          gWDfrac, kappa160=517. * np.pi,
                                          WDT=6.), bands)
            model_gordon = WD(wl_complete, MWSigmaD, 15., 2.9, gWDfrac,
                              kappa160=517. * np.pi, WDT=6.)
        elif method_abbr == 'PL':
            samples = np.array([logkappa160s[mask], loggammas[mask],
                                alphas[mask], logUmins[mask], r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$\log\gamma$', r'$\alpha$',
                      r'\log U_{min}', r'$\tilde{\chi}^2$']
            loggamma, loggamma_err = exp_and_error(loggammas, pr, 'loggamma')
            alpha, alpha_err = exp_and_error(alphas, pr, 'alpha')
            logUmin, logUmin_err = exp_and_error(logUmins, pr, 'logUmin')
            min_alpha = max(alpha - 0.3, 1.1)
            max_alpha = alpha + 0.3
            min_loggamma = loggamma - 0.3
            max_loggamma = min(loggamma + 0.3, 0.)
            min_logUmin = logUmin - 0.1
            max_logUmin = logUmin + 0.1
            grp['loggamma'], grp['loggamma_err'] = loggamma, loggamma_err
            grp['alpha'], grp['alpha_err'] = alpha, alpha_err
            grp['logUmin'], grp['logUmin_err'] = logUmin, logUmin_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, PowerLaw(wl_complete, MWSigmaD, alpha,
                                                10**loggamma, logUmin,
                                                beta=beta_f,
                                                kappa160=kappa160), bands)
            model_complete = PowerLaw(wl_complete, MWSigmaD, alpha,
                                      10**loggamma, logUmin, beta=beta_f,
                                      kappa160=kappa160)
            gordon_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, 17.2, 1.96,
                                             9.6 * np.pi), bands)
            model_gordon = SEMBB(wl_complete, MWSigmaD, 17.2, 1.96,
                                 9.6 * np.pi)
        hf.close()
    #
    del samples, labels
    """
    fig = corner(samples.T, labels=labels, quantities=(0.16, 0.84),
                 show_titles=True, title_kwargs={"fontsize": 12})
    with PdfPages('output/_CALI_' + method_abbr + '_' +
                  mode_titles[cov_mode] + '.pdf') as pp:
        pp.savefig(fig)
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.loglog(wl_complete, model_gordon, label='G14EXP')
    ax.loglog(wl, mode_integrated, 'x', ms=15, label='fitting (int)')
    ax.loglog(wl_complete, model_complete, label='fitting')
    ax.errorbar(wl, MWSED, yerr, label='MWSED')
    ax.loglog(wl, gordon_integrated, 'x', ms=15, label='G14 (int)')
    ax.legend()
    ax.set_ylim(0.03, 3.0)
    ax.set_xlim(80, 1000)
    ax.set_xlabel(r'SED [$MJy\,sr^{-1}\,(10^{20}(H\,Atom)\,cm^{-2})^{-1}$]')
    ax.set_ylabel(r'Wavelength ($\mu m$)')
    with PdfPages('output/_CALI_' + method_abbr + '_' +
                  mode_titles[cov_mode] + '_MODEL.pdf') as pp:
        pp.savefig(fig)
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.")
