from time import clock
# import emcee
from h5py import File
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import c, h, k_B
# import corner
from idc_lib import idc_voronoi, gal_data

# Before making beta a variable
const_beta = 2.

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
cali_mat2 = np.array([[0.1**2 + 0.02**2, 0.1**2, 0, 0, 0],
                      [0.1**2, 0.1**2 + 0.02**2, 0, 0, 0],
                      [0, 0, 0.08**2 + 0.015**2, 0.08**2, 0.08**2],
                      [0, 0, 0.08**2, 0.08**2 + 0.015**2, 0.08**2],
                      [0, 0, 0.08**2, 0.08**2, 0.08**2 + 0.015**2]])

# Number of fitting parameters
ndim = 3


def _B(T, freq=nu):
    """Return blackbody SED of temperature T(with unit) in MJy"""
    with np.errstate(over='ignore'):
        return (2 * h * freq**3 / c**2 / (np.exp(h * freq / k_B / T) - 1)
                ).to(u.Jy).value * 1E-6

const_beta = 2.


def _model(wl, sigma, T, beta, freq=nu):
    """Return fitted SED in MJy"""
    return const * kappa160 * (160.0 / wl)**beta * \
        sigma * _B(T * u.K, freq)


def fit_dust_density(name):
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

    popt = np.full_like(sed[:, :, :ndim], np.nan)
    perr = popt.copy()
    cov_n1_map = np.full([sed.shape[0], sed.shape[1], 5, 5], np.nan)
    binmap = np.full_like(sed[:, :, 0], np.nan, dtype=int)
    radiusmap = np.full_like(sed[:, :, 0], np.nan)
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
        # Save something just to avoid reading error
        ept = np.empty_like(total_gas, dtype=int)
        with File('output/dust_data.h5', 'a') as hf:
            grp = hf.create_group(name)
            grp.create_dataset('Total_gas', data=total_gas)
            grp.create_dataset('Herschel_SED', data=sed)
            """SED not binned yet"""
            grp.create_dataset('Dust_surface_density', data=ept)
            grp.create_dataset('Dust_surface_density_err', data=ept)
            grp.create_dataset('Dust_temperature', data=ept)
            grp.create_dataset('Dust_temperature_err', data=ept)
            grp.create_dataset('Dust_surface_density_max', data=ept)
            grp.create_dataset('Dust_surface_density_err_max', data=ept)
            grp.create_dataset('Dust_temperature_max', data=ept)
            grp.create_dataset('Dust_temperature_err_max', data=ept)
            grp.create_dataset('Binmap', data=ept)
            grp.create_dataset('Galaxy_distance', data=D)
            grp.create_dataset('Galaxy_center', data=glx_ctr)
            grp.create_dataset('INCL', data=INCL)
            grp.create_dataset('PA', data=PA)
            grp.create_dataset('PS', data=PS)
            grp.create_dataset('Radius_map', data=ept)  # kpc

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
    binNumlist = np.unique(binNum)
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    sed_avg = np.zeros([len(binNumlist), 5])

    print("Generating grid...")
    """ Grid parameters """
    logsigma_step = 0.01
    min_logsigma = -4.
    max_logsigma = 1.
    T_step = 0.1
    min_T = 5.
    max_T = 50.
    beta_step = 0.1
    min_beta = 0.8
    max_beta = 2.5
    logsigmas = np.arange(min_logsigma, max_logsigma, logsigma_step)
    Ts = np.arange(min_T, max_T, T_step)
    betas = np.arange(min_beta, max_beta, beta_step)
    logsigmas, Ts, betas = np.meshgrid(logsigmas, Ts, betas)
    try:
        with File('output/rsrf_models.h5', 'r') as hf:
            models = np.array(hf['models'])
    except IOError:
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
        pacs_models = np.zeros([Ts.shape[0], Ts.shape[1], Ts.shape[2],
                                len(pacs_wl)])
        for i in range(len(pacs_wl)):
            pacs_models[:, :, :, i] = _model(pacs_wl[i], 10**logsigmas, Ts,
                                             betas, pacs_nu[i])
        del pacs_nu
        models[:, :, :, 0] = np.sum(pacs_models * pacs100dnu, axis=3) / \
            np.sum(pacs100dnu * pacs_wl / wl[0])
        models[:, :, :, 1] = np.sum(pacs_models * pacs160dnu, axis=3) / \
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
        spire_models = np.zeros([Ts.shape[0], Ts.shape[1], Ts.shape[2],
                                 len(spire_wl)])
        for i in range(len(spire_wl)):
            spire_models[:, :, :, i] = _model(spire_wl[i], 10**logsigmas, Ts,
                                              betas, spire_nu[i])
        del spire_nu
        models[:, :, :, 2] = np.sum(spire_models * spire250dnu, axis=3) / \
            np.sum(spire250dnu * spire_wl / wl[2])
        models[:, :, :, 3] = np.sum(spire_models * spire350dnu, axis=3) / \
            np.sum(spire350dnu * spire_wl / wl[3])
        models[:, :, :, 4] = np.sum(spire_models * spire500dnu, axis=3) / \
            np.sum(spire500dnu * spire_wl / wl[4])
        #
        del spire_wl, spire250dnu, spire350dnu, spire500dnu
        del spire_models
        print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
        with File('output/rsrf_models.h5', 'a') as hf:
            hf.create_dataset('models', data=models)
    """
    Start fitting
    """
    print("Start fitting", name, "dust surface density...")
    tic = clock()
    p = 0
    pdfs = pd.DataFrame()
    # results = [] # array for saving all the raw chains
    for i in range(len(binNumlist)):
        if (i + 1) / len(binNumlist) > p:
            print('Step', (i + 1), '/', str(len(binNumlist)) + '.',
                  "Elapsed time:", round(clock()-tic, 3), "s.")
            p += 0.1
        """ Binning everything """
        bin_ = (binmap == binNumlist[i])
        # total_gas weighted radius / total gas
        radiusmap[bin_] = np.sum(dp_radius[bin_] * total_gas[bin_]) / \
            np.sum(total_gas[bin_])
        total_gas[bin_] = np.nanmean(total_gas[bin_])
        # mean sed
        sed_avg[i] = np.mean(sed[bin_], axis=0)
        sed[bin_] = sed_avg[i]
        sed_vec = sed_avg[i].reshape(1, 5)
        # bkg covariance matrix
        bkgcov_avg = bkgcov / np.sum(bin_)
        # uncertainty diagonal matrix
        unc2_avg = np.mean(sed_unc[bin_]**2, axis=0)
        unc2_avg[np.isnan(unc2_avg)] = 0
        unc2cov_avg = np.identity(5) * unc2_avg
        # calibration error covariance matrix
        calcov = sed_vec.T * cali_mat2 * sed_vec
        # Finally everything for covariance matrix is here...
        cov_n1 = np.linalg.inv(bkgcov_avg + unc2cov_avg + calcov)
        cov_n1_map[bin_] = cov_n1
        """ Grid fitting """
        # sed_avg[i].shape = (5)
        # models.shape = (len(logsigmas), len(Ts), 5)
        diff = models - sed_avg[i]
        temp_matrix = np.empty_like(diff)
        for i in range(5):
            temp_matrix[:, :, :, i] = np.sum(diff * cov_n1[:, i], axis=3)
        chi2 = np.sum(temp_matrix * diff, axis=3)
        """ Find the (s, t) that gives Maximum likelihood """
        temp = chi2.argmin()
        tempa = temp // (chi2.shape[1] * chi2.shape[2])
        temp = temp % (chi2.shape[1] * chi2.shape[2])
        tempb = temp // chi2.shape[2]
        tempc = temp % chi2.shape[2]
        s_ML = logsigmas[tempa, tempb, tempc]
        t_ML = Ts[tempa, tempb, tempc]
        b_ML = betas[tempa, tempb, tempc]
        """ Show map """
        # plt.figure()
        # imshowid(np.log10(-lnprobs))

        """ Randomly choosing something to plot here """
        # if np.random.rand() > 0.0:
        #     plot_single_bin(name, binNumlist[i], samples, sed_avg[i],
        #                     inv_sigma2, sopt, topt, lnprobs, Ts, logsigmas)
        """ Continue saving """
        pr = np.exp(-0.5 * chi2)
        mask = chi2 < np.nanmin(chi2) + 12
        logsigmas_cp, Ts_cp, betas_cp, pr_cp = \
            logsigmas[mask], Ts[mask], betas[mask], pr[mask]
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
        idb = np.argsort(betas_cp)
        betas_cp = betas_cp[idb]
        prb = pr_cp[idb]
        csp = np.cumsum(prb)[:-1]
        csp = np.append(0, csp / csp[-1])
        ssb = np.interp([0.16, 0.5, 0.84], csp, betas_cp).tolist()
        """ Saving to results """
        sss[1], sst[1], ssb[1] = s_ML, t_ML, b_ML
        popt[bin_] = np.array([sss[1], sst[1], ssb[1]])
        perr[bin_] = np.array([max(sss[2]-sss[1], sss[1]-sss[0]),
                               max(sst[2]-sst[1], sst[1]-sst[0]),
                               max(ssb[2]-ssb[1], ssb[1]-ssb[0])])
        """ New: saving PDF """
        pdf = np.sum(pr, axis=(0, 2))
        pdf /= np.sum(pdf)
        pdfs = pdfs.append([pdf])

    pdfs = pdfs.set_index(binNumlist)
    pdfs.to_csv('output/' + name + '_pdf.csv', index=True)
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
    with File('output/dust_data.h5', 'a') as hf:
        grp = hf.create_group(name)
        grp.create_dataset('Total_gas', data=total_gas)
        grp.create_dataset('Herschel_SED', data=sed)
        grp.create_dataset('Dust_surface_density_log', data=popt[:, :, 0])
        # sopt in log scale (search sss)
        grp.create_dataset('Dust_surface_density_err_dex', data=perr[:, :, 0])
        # serr in dex
        grp.create_dataset('Dust_temperature', data=popt[:, :, 1])
        grp.create_dataset('Dust_temperature_err', data=perr[:, :, 1])
        grp.create_dataset('beta', data=popt[:, :, 2])
        grp.create_dataset('beta_err', data=perr[:, :, 2])
        grp.create_dataset('Herschel_covariance_matrix', data=cov_n1_map)
        grp.create_dataset('Binmap', data=binmap)
        grp.create_dataset('Galaxy_distance', data=D)
        grp.create_dataset('Galaxy_center', data=glx_ctr)
        grp.create_dataset('INCL', data=INCL)
        grp.create_dataset('PA', data=PA)
        grp.create_dataset('PS', data=PS)
        grp.create_dataset('Radius_map', data=radiusmap)  # kpc
        grp.create_dataset('logsigmas', data=logsigmas[0])
    print("Datasets saved.")
