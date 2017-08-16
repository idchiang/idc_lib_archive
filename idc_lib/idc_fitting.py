from time import clock
# import emcee
from h5py import File
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import astropy.units as u
from astropy.constants import c, h, k_B
from sklearn import linear_model
# from scipy.stats.stats import pearsonr
# import corner
from . import idc_voronoi
from .idc_plot import map2bin


# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
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


# Probability functions & model functions for fitting (internal)
def B(T, freq=nu):
    """Return blackbody SED of temperature T(with unit) in MJy"""
    with np.errstate(over='ignore'):
        return (2 * h * freq**3 / c**2 / (np.exp(h * freq / k_B / T) - 1)
                ).to(u.Jy).value * 1E-6


def model(wl, sigma, T, beta, freq=nu):
    """Return fitted SED in MJy"""
    const = 2.0891E-4
    kappa160 = 9.6 * np.pi
    return const * kappa160 * (160.0 / wl)**beta * \
        sigma * B(T * u.K, freq)


def half_integer(num):
    return round(num * 2) / 2


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
    plt.ioff()

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
    try:
        with File('output/Voronoi_data.h5', 'r') as hf:
            grp = hf[name]
            binlist = np.array(grp['BINLIST'])
            binmap = np.array(grp['BINMAP'])
            gas_avg = np.array(grp['GAS_AVG'])
            sed_avg = np.array(grp['Herschel_SED'])
            cov_n1s = np.array(grp['Herschel_covariance_matrix'])
            inv_sigma2s = np.array(grp['Herschel_variance'])
            radius_avg = np.array(grp['Radius_avg'])
    except OSError:
        print("Start binning " + name + "...")
        tic = clock()
        noise4snr = np.array([np.sqrt(bkgcov[i, i]) for i in range(5)])
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
                    idc_voronoi.voronoi_m(x_l, y_l, signal_l, noise_l,
                                          targetSN, pixelsize=1, plot=False,
                                          quiet=True)
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
        pp = PdfPages('output/' + name + '_Voronnoi.pdf')
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
            unc2cov_avg = np.identity(5) * unc2_avg
            # calibration error covariance matrix
            sed_vec = sed_avg[-1].reshape(1, 5)
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
        with File('output/Voronoi_data.h5', 'a') as hf:
            grp = hf.create_group(name)
            grp.create_dataset('BINLIST', data=binlist)
            grp.create_dataset('BINMAP', data=binmap)
            grp.create_dataset('GAS_AVG', data=np.array(gas_avg))
            grp.create_dataset('Herschel_SED', data=np.array(sed_avg))
            grp.create_dataset('Herschel_covariance_matrix',
                               data=np.array(cov_n1s))
            grp.create_dataset('Herschel_variance', data=np.array(inv_sigma2s))
            grp.create_dataset('Radius_avg', data=np.array(radius_avg))  # kpc
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
        print("Reading/Generating fitting grid...")

    tic = clock()
    """ Grid parameters """
    logsigma_step = 0.025
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

    """
    Start fitting
    """
    print("Start fitting", name, "dust surface density...")
    tic = clock()
    p = 0
    sopt, serr, topt, terr, bopt, berr = [], [], [], [], [], []
    b_model = []
    pdfs, t_pdfs, b_pdfs = [], [], []
    # results = [] # array for saving all the raw chains
    for i in range(len(binlist)):
        if (i + 1) / len(binlist) > p:
            print(' --Step', (i + 1), '/', str(len(binlist)) + '.',
                  "Elapsed time:", round(clock()-tic, 3), "s.")
            p += 0.1
        """ Binning everything """
        bin_ = (binmap == binlist[i])
        if cov_mode:
            cov_n1 = cov_n1s[i]
            diff = models - sed_avg[i]
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
            """ Grid fitting """
            chi2 = (np.sum((sed_avg[i] - models)**2 * inv_sigma2s[i],
                           axis=ndim))
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
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.")
    # Saving to h5 file
    # Total_gas and dust in M_sun/pc**2
    # Temperature in K
    # SED in MJy/sr
    # D in Mpc
    # Galaxy_distance in Mpc
    # Galaxy_center in pixel [y, x]
    if fixed_beta:
        fn = 'output/Dust_data_fb_'
    else:
        fn = 'output/Dust_data_' if cov_mode else 'output/Dust_data_nc_'
    with File(fn + name + '.h5', 'a') as hf:
        hf.create_dataset('Dust_surface_density_log', data=sopt)
        # sopt in log scale (search sss)
        hf.create_dataset('Dust_surface_density_err_dex', data=serr)
        # serr in dex
        hf.create_dataset('Dust_temperature', data=topt)
        hf.create_dataset('Dust_temperature_err', data=terr)
        hf.create_dataset('beta', data=bopt)
        hf.create_dataset('beta_err', data=berr)
        hf.create_dataset('logsigmas', data=logsigmas_untouched)
        hf.create_dataset('Ts', data=Ts_untouched)
        hf.create_dataset('betas', data=betas_untouched)
        hf.create_dataset('PDF', data=pdfs)
        hf.create_dataset('PDF_T', data=t_pdfs)
        hf.create_dataset('PDF_B', data=b_pdfs)
        hf.create_dataset('Best_fit_model', data=b_model)
    print("Datasets saved.")


def fit_dust_density_Tmap(name='NGC5457', beta_f=2.0, r_bdr=0.8, rbin=51,
                          dbin=250, cmap1='Reds', Tmin=5):
    ndim = 2
    with File('output/Dust_data_fb_' + name + '.h5', 'a') as hf:
        aT = np.array(hf['Dust_temperature'])
    with File('output/Voronoi_data.h5', 'r') as hf:
        grp = hf[name]
        binlist = np.array(grp['BINLIST'])
        binmap = np.array(grp['BINMAP'])
        # gas_avg = np.array(grp['GAS_AVG'])
        ased = np.array(grp['Herschel_SED'])
        acovn1 = np.array(grp['Herschel_covariance_matrix'])
        # inv_sigma2s = np.array(grp['Herschel_variance'])
        aRadius = np.array(grp['Radius_avg'])
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        R25 = float(np.array(grp['R25_KPC']))
        with np.errstate(invalid='ignore', divide='ignore'):
            logSFR = np.log10(map2bin(np.array(grp['SFR']), binlist, binmap))
            logSMSD = np.log10(map2bin(np.array(grp['SMSD']), binlist, binmap))
    alogSFR = np.array([logSFR[binmap == binlist[i]][0] for i in
                        range(len(binlist))])
    alogSMSD = np.array([logSMSD[binmap == binlist[i]][0] for i in
                         range(len(binlist))])
    print('Calculating predicted temperature...')
    tic = clock()
    mask = (aRadius < r_bdr * R25) * (~np.isnan(alogSFR + alogSMSD + aT)) * \
        (~np.isinf(alogSFR + alogSMSD + aT))
    regr = linear_model.LinearRegression()
    DataX = np.array([alogSFR, alogSMSD]).T
    regr.fit(DataX[mask], aT[mask])
    print(regr.coef_, regr.intercept_)
    nanmask = np.isnan(alogSFR + alogSMSD + aT) + \
        np.isinf(alogSFR + alogSMSD + aT)
    DataX[nanmask] = np.array([0, 0])
    aT_pred = regr.predict(DataX)
    aT_pred[aT_pred < Tmin] = Tmin
    aT_pred = np.array(list(map(half_integer, aT_pred)))
    aT_pred[nanmask] = np.nan
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    print("Reading/Generating fitting grid...")
    tic = clock()
    """ Grid parameters """
    logsigma_step = 0.025
    min_logsigma = -4.
    max_logsigma = 1.
    logsigmas_untouched = np.arange(min_logsigma, max_logsigma, logsigma_step)
    Ts_untouched = np.unique(aT_pred)
    logsigmas, Ts = np.meshgrid(logsigmas_untouched, Ts_untouched)
    models = np.zeros([Ts.shape[0], Ts.shape[1], 5])
    print(" --Number of temperature points:", len(Ts_untouched))
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
    pacs_models = np.zeros([Ts.shape[0], Ts.shape[1], len(pacs_wl)])
    for i in range(len(pacs_wl)):
        pacs_models[:, :, i] = model(pacs_wl[i], 10**logsigmas, Ts,
                                     beta_f, pacs_nu[i])
    del pacs_nu
    models[:, :, 0] = np.sum(pacs_models * pacs100dnu, axis=ndim) / \
        np.sum(pacs100dnu * pacs_wl / wl[0])
    models[:, :, 1] = np.sum(pacs_models * pacs160dnu, axis=ndim) / \
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
    spire250dnu = spire_rsrf['SPIRE_250'].values * spire_rsrf['dnu'].values[0]
    spire350dnu = spire_rsrf['SPIRE_350'].values * spire_rsrf['dnu'].values[0]
    spire500dnu = spire_rsrf['SPIRE_500'].values * spire_rsrf['dnu'].values[0]
    del spire_rsrf
    #
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
    #
    del spire_wl, spire250dnu, spire350dnu, spire500dnu
    del spire_models
    print("   --Done. Elapsed time:", round(clock()-ttic, 3), "s.\n")
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
    """
    Start fitting
    """
    print("Start fitting", name, "dust surface density...")
    tic = clock()
    p = 0
    sopt, serr, b_model = [], [], []
    pdfs = np.zeros([len(binlist), len(logsigmas_untouched)], dtype=float)
    # results = [] # array for saving all the raw chains
    for i in range(len(binlist)):
        if (i + 1) / len(binlist) > p:
            print(' --Step', (i + 1), '/', str(len(binlist)) + '.',
                  "Elapsed time:", round(clock()-tic, 3), "s.")
            p += 0.1
        """ Binning everything """
        diff = (models[Ts_untouched == aT_pred[i]] - ased[i]).reshape(-1, 5)
        temp_matrix = np.empty_like(diff)
        cov_n1 = acovn1[i]
        for j in range(5):
            temp_matrix[:, j] = np.sum(diff * cov_n1[:, j], axis=1)
        chi2 = np.sum(temp_matrix * diff, axis=1)
        """ Find the (s, t) that gives Maximum likelihood """
        if np.isnan(aT_pred[i]):
            sopt.append(np.nan)
            serr.append(np.nan)
            pdfs[i] = np.zeros(len(logsigmas_untouched), dtype=float)
            b_model.append(np.full(5, np.nan, dtype=float))
        else:
            s_ML = logsigmas_untouched[chi2.argmin()]
            """ Continue saving """
            pr = np.exp(-0.5 * chi2)
            mask = chi2 < np.nanmin(chi2) + 12
            logsigmas_cp, pr_cp = logsigmas_untouched[mask], pr[mask]
            #
            ids = np.argsort(logsigmas_cp)
            logsigmas_cp = logsigmas_cp[ids]
            prs = pr_cp[ids]
            csp = np.cumsum(prs)[:-1]
            try:
                csp = np.append(0, csp / csp[-1])
            except IndexError:
                print(ased[i])
                print(aT_pred[i])
                print(prs.shape)
                print(csp.shape)
                print(ids.shape)
                print('Force quit: something went wrong...')
                return 0
            sss = np.interp([0.16, 0.5, 0.84], csp, logsigmas_cp).tolist()
            #
            """ Saving to results """
            sopt.append(s_ML)
            serr.append(max(sss[2]-s_ML, s_ML-sss[0]))
            """ New: saving PDF """
            pdfs[i] = pr / np.sum(pr)
            b_model.append(models[Ts_untouched == aT_pred[i]][0]
                           [chi2.argmin()])
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.")
    # Saving to h5 file
    # Total_gas and dust in M_sun/pc**2
    # Temperature in K
    # SED in MJy/sr
    # D in Mpc
    # Galaxy_distance in Mpc
    # Galaxy_center in pixel [y, x]
    with File('output/Dust_data_fbft_' + name + '.h5', 'a') as hf:
        hf.create_dataset('Dust_surface_density_log', data=sopt)
        hf.create_dataset('Dust_surface_density_err_dex', data=serr)
        hf.create_dataset('Dust_temperature', data=aT_pred)
        hf.create_dataset('Dust_temperature_err', data=np.full_like(sopt, 0.0))
        hf.create_dataset('beta', data=np.full_like(sopt, beta_f))
        hf.create_dataset('beta_err', data=np.full_like(sopt, 0.0))
        hf.create_dataset('logsigmas', data=logsigmas_untouched)
        hf.create_dataset('PDF', data=np.array(pdfs).astype(float))
        hf.create_dataset('Best_fit_model', data=b_model)
    print("Datasets saved.\n")
