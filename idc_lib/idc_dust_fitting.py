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
from . import idc_voronoi, gal_data


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
    logsigma_step = 0.005
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
        grp.create_dataset('logsigmas', data=logsigmas[0, :, 0])
    print("Datasets saved.")

"""
# Code for plotting the results
def plot_single_bin(name, binnum, samples, sed_avg, inv_sigma2, sopt, topt,
                    lnprobs, Ts, logsigmas):
    bins = 50
    nwalkers, nsteps, ndim = samples.shape
    lnpr = np.zeros([nwalkers, nsteps, 1])

    for w in range(nwalkers):
        for n in range(nsteps):
            lnpr[w, n, 0] = _lnprob(samples[w, n], wl, sed_avg, inv_sigma2)
    samples = np.concatenate([samples, lnpr], axis=2)

    # Plot fitting results versus step number
    fig, ax = plt.subplots(1, 3)
    ax[0, 0].set_title('Surface density')
    ax[0, 1].set_title('Temperature')
    ax[0, 2].set_title('ln(Probability)')
    for w in range(nwalkers):
        ax[0, 0].plot(samples[w, :, 0], c='b')
        ax[0, 1].plot(samples[w, :, 1], c='b')
        ax[0, 2].plot(samples[w, :, 2], c='b')
    ax[0, 2].set_ylim(-50, np.max(samples[:, :, 2]))
    fig.suptitle(name + ' bin no.' + str(binnum) + ' mcmc')
    fig.savefig('output/' + name + 'bin_' + str(binnum) + 'mcmc.png')
    fig.clf()

    # MCMC Corner plot
    samples = samples.reshape(-1, ndim + 1)
    smax = np.max(samples[:, 0])
    smin = np.min(samples[:, 0])
    tmax = np.max(samples[:, 1]) + 1.
    tmin = np.min(samples[:, 1]) - 1.
    lpmax = np.max(samples[:, 2]) + 1.
    lpmin = np.max([-50., np.min(samples[:, 2])])
    lnpropt = _lnprob((sopt, topt), wl, sed_avg, inv_sigma2)
    corner.corner(samples, bins=bins, truths=[sopt, topt, lnpropt],
                  labels=["$\Sigma_d$", "$T$", "$\ln(Prob)$"],
                  range=[(smin, smax), (tmin, tmax), (lpmin, lpmax)],
                  show_titles=True)
    plt.suptitle(name + ' bin no.' + str(binnum) + ' mcmc corner plot')
    plt.savefig('output/' + name + 'bin_' + str(binnum) + 'mcmc_corner.png')
    plt.clf()

    # PDF from grid-based method
    lnprobs = lnprobs.flatten()
    prs = np.exp(lnprobs)
    sigmas = 10**logsigmas.flatten()
    Ts = Ts.flatten()
    mask = (lnprobs > np.max(lnprobs) - 6)
    prs, sigmas, Ts, lnprobs = prs[mask], sigmas[mask], Ts[mask], lnprobs[mask]
    samples2 = np.concatenate([sigmas.reshape(-1, 1), Ts.reshape(-1, 1),
                               lnprobs.reshape(-1, 1)], axis=1)
    corner.corner(samples2, bins=bins, truths=[sopt, topt, lnpropt],
                  labels=["$\Sigma_d$", "$T$", "$\ln(Prob)$"],
                  # range = [(smin, smax), (tmin, tmax), (lpmin, lpmax)],
                  show_titles=True, weights=prs)
    plt.suptitle('Grid-based PDF')
    plt.savefig('output/' + name + 'bin_' + str(binnum) + 'grid_pdf.png')
    plt.clf()

    plt.close("all")

    # Plotting data versus model
    n = 50
    alpha = 0.1
    samples = samples[:,nsteps/2:,:].reshape(-1, ndim+1)
    sexp, texp = popt[yt, xt]
    wlexp = np.linspace(70,520)
    nuexp = (c / wlexp / u.um).to(u.Hz)
    modelexp = _model(wlexp, sexp, texp, nuexp)

    plt.figure()
    list_ = np.random.randint(0, len(samples), n)
    for i in xrange(n):
        model_i = _model(wlexp, samples[list_[i],0], samples[list_[i],1],
                         nuexp)
        plt.plot(wlexp, model_i, alpha = alpha, c = 'g')
    plt.plot(wlexp, modelexp, label='Expectation', c = 'b')
    plt.errorbar(wl, sed_avg[r_index], yerr = bkgerr, fmt='ro', label='Data')
    plt.axis('tight')
    plt.legend()
    plt.title('NGC 3198 ['+str(yt)+','+str(xt)+']')
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel('SED')
    plt.savefig('output/NGC 3198 ['+str(yt)+','+str(xt)+']_datamodel.png')

"""


def read_dust_file(name='NGC5457', bins=30, off=45., cmap0='gist_heat',
                   dr25=0.025):
    # name = 'NGC3198'
    # bins = 10
    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        logs_d = np.array(grp['Dust_surface_density_log'])  # in log
        serr = np.array(grp['Dust_surface_density_err_dex'])  # in dex
        topt = np.array(grp['Dust_temperature'])
        terr = np.array(grp['Dust_temperature_err'])
        bopt = np.array(grp['beta'])
        berr = np.array(grp['beta_err'])
        total_gas = np.array(grp['Total_gas'])
        sed = np.array(grp['Herschel_SED'])
        cov_n1_map = np.array(grp['Herschel_covariance_matrix'])
        binmap = np.array(grp['Binmap'])
        radiusmap = np.array(grp['Radius_map'])  # kpc
        D = float(np.array(grp['Galaxy_distance']))
        logsigmas = np.array(grp['logsigmas'])
        # readme = np.array(grp.get('readme'))

    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        things = np.array(grp['THINGS']) * col2sur * H2HaHe
        heracles = np.array(grp['HERACLES'])
        # total_gas_ub = np.array(grp['Total_gas'])
        diskmask = np.array(grp['Diskmask'])
        dp_radius = np.array(grp['DP_RADIUS'])

    plt.ioff()

    nanmask = np.isnan(total_gas) + np.isnan(things) + np.isnan(heracles)
    total_gas[nanmask], things[nanmask], heracles[nanmask] = -1., -1., -1.
    total_gas[np.less_equal(total_gas, 0)] = np.nan
    things[np.less_equal(things, 0)] = np.nan
    heracles[np.less_equal(heracles, 0)] = np.nan

    chi2 = np.full_like(logs_d, np.nan)
    binlist = np.unique(binmap[diskmask])
    for bin_ in binlist:
        mask = binmap == bin_
        cov_n1 = cov_n1_map[mask][0]
        model = _model(wl, 10**logs_d[mask][0], topt[mask][0], bopt[mask][0],
                       nu)
        diff = (sed[mask][0] - model).reshape(5, 1)
        chi2[mask] = np.dot(np.dot(diff.T, cov_n1), diff)[0, 0]
        if chi2[mask][0] > off or serr[mask][0] > 1.:
            logs_d[mask], topt[mask], serr[mask], terr[mask], chi2[mask] = \
                np.nan, np.nan, np.nan, np.nan, np.nan
            total_gas[mask], things[mask], heracles[mask] = \
                np.nan, np.nan, np.nan
            bopt[mask], berr[mask] = np.nan, np.nan
    logs_gas = np.log10(total_gas)
    # logs_HI = np.log10(things)
    logs_H2 = np.log10(heracles)
    logdgr = logs_d - logs_gas

    # D in Mpc. Need r25 in kpc
    R25 = gal_data.gal_data([name]).field('R25_DEG')[0]
    R25 *= (np.pi / 180.) * (D * 1E3)

    # Fitting results
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    ax[0, 0].set_title(r'$\Sigma_d$ $(\log_{10}(M_\odot pc^{-2}))$', size=30)
    cax[0, 0] = ax[0, 0].imshow(logs_d, origin='lower', cmap=cmap0)
    ax[1, 0].set_title(r'$\Sigma_d$ error (dex)', size=30)
    cax[1, 0] = ax[1, 0].imshow(serr, origin='lower', cmap=cmap0)
    ax[0, 1].set_title(r'$T_d$ ($K$)', size=30)
    cax[0, 1] = ax[0, 1].imshow(topt, origin='lower', cmap=cmap0)
    ax[1, 1].set_title(r'$T_d$ error ($K$)', size=30)
    cax[1, 1] = ax[1, 1].imshow(terr, origin='lower', cmap=cmap0)
    ax[0, 2].set_title(r'$\beta$', size=30)
    cax[0, 2] = ax[0, 2].imshow(bopt, origin='lower', cmap=cmap0)
    ax[1, 2].set_title(r'$\beta$ error', size=30)
    cax[1, 2] = ax[1, 2].imshow(berr, origin='lower', cmap=cmap0)
    for i in range(2):
        for j in range(3):
            fig.colorbar(cax[i, j], ax=ax[i, j])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    fig.savefig('output/' + name + '_Sd_Td.png')
    fig.clf()

    # Total gas & DGR
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    cax[0, 0] = ax[0, 0].imshow(logdgr, origin='lower', cmap=cmap0)
    ax[0, 0].set_title('DGR (log)', size=20)
    cax[0, 1] = ax[0, 1].imshow(logs_gas, origin='lower', cmap=cmap0)
    ax[0, 1].set_title(r'$\Sigma_{gas}$ $(\log_{10}(M_\odot pc^{-2}))$',
                       size=20)
    cax[0, 2] = ax[0, 2].imshow(logs_H2, origin='lower', cmap=cmap0)
    ax[0, 2].set_title(r'HERACLES (log, not binned)', size=20)
    cax[1, 0] = ax[1, 0].imshow(logs_d, origin='lower', cmap=cmap0)
    ax[1, 0].set_title(r'$\Sigma_{d}$ $(\log_{10}(M_\odot pc^{-2}))$', size=20)
    cax[1, 1] = ax[1, 1].imshow(serr, origin='lower', cmap=cmap0)
    ax[1, 1].set_title(r'$\Sigma_d$ error (dex)', size=20)
    cax[1, 2] = ax[1, 2].imshow(chi2, origin='lower', cmap=cmap0)
    ax[1, 2].set_title(r'$\chi^2$', size=20)
    for i in range(2):
        for j in range(3):
            fig.colorbar(cax[i, j], ax=ax[i, j])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_DGR_Sd_GAS.png')
    fig.clf()

    plt.figure()
    plt.imshow(logdgr, alpha=0.8, origin='lower', cmap='gist_heat')
    plt.colorbar()
    plt.imshow(logs_H2, alpha=0.6, origin='lower', cmap='bone')
    plt.colorbar()
    plt.title(r'DGR overlay with $H_2$ map', size=24)
    plt.savefig('output/' + name + '_DGR_H2_OL.png')

    # hist2d
    if len(logsigmas.shape) == 2:
        logsigmas = logsigmas[:, 0]
    sigmas = 10**logsigmas
    #
    pdfs = pd.read_csv('output/' + name + '_pdf.csv', index_col=0)
    pdfs.index = pdfs.index.astype(int)
    #
    radiusmap /= R25
    dp_radius /= R25
    #
    r, s, w = [], [], []
    for i in pdfs.index:
        bin_ = binmap == i
        temp_g = total_gas[bin_][0]
        temp_r = radiusmap[bin_][0]
        mask2 = (pdfs.iloc[i] > pdfs.iloc[i].max() / 1000).values
        temp_s = sigmas[mask2]
        temp_p = pdfs.iloc[i][mask2]
        temp_p /= np.sum(temp_p)
        for j in range(len(temp_s)):
            r.append(temp_r)
            s.append(temp_s[j] / temp_g)
            w.append(temp_g * temp_p[j])
        if i % (len(pdfs.index) // 10) == 0:
            print('Current bin:' + str(i) + ' of ' + str(len(pdfs.index)))
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
    """ Generates H2 & HI radial profile """
    mask = ~np.isnan(things)
    r_HI = dp_radius[mask]
    sd_HI = things[mask]
    total_sd, _ = np.histogram(r_HI, rbins, weights=sd_HI)
    count_HI, _ = np.histogram(r_HI, rbins)
    with np.errstate(invalid='ignore'):
        mean_HI = total_sd / count_HI
    #
    mask = ~np.isnan(heracles)
    r_H2 = dp_radius[mask]
    sd_H2 = heracles[mask]
    total_sd, _ = np.histogram(r_H2, rbins, weights=sd_H2)
    count_H2, _ = np.histogram(r_H2, rbins)
    with np.errstate(invalid='ignore'):
        mean_H2 = total_sd / count_H2
    #
    sbins = (sbins[:-1] + sbins[1:]) / 2
    rbins = (rbins[:-1] + rbins[1:]) / 2
    dgr_median = []
    # dgr_avg = []
    zeromask = np.full(counts.shape[1], True, dtype=bool)
    #
    for i in range(counts.shape[1]):
        if np.sum(counts[:, i]) > 0:
            mask = counts[:, i] > (np.max(counts[:, i]) / 1000)
            smax = np.max(counts[mask, i])
            smin = np.min(counts[mask, i])
            csp = np.cumsum(counts[:, i])[:-1]
            csp = np.append(0, csp / csp[-1])
            sss = np.interp([0.16, 0.5, 0.84], csp, sbins)
            fig, ax = plt.subplots(2, 1)
            ax[0].semilogx([sss[0]] * len(counts[:, i]), counts[:, i],
                           label='16')
            ax[0].semilogx([sss[1]] * len(counts[:, i]), counts[:, i],
                           label='50')
            ax[0].semilogx([sss[2]] * len(counts[:, i]), counts[:, i],
                           label='84')
            # dgr_avg.append(np.sum(counts[:, i] * sbins))
            # ax[0].semilogx([dgr_avg[-1]] * len(counts[:, i]),
            #                counts[:, i], label='Exp')
            ax[0].semilogx(sbins, counts[:, i], label='PDF')
            ax[0].set_xlim([smin, smax])
            ax[0].legend()

            dgr_median.append(sss[1])

            ax[1].semilogx(sbins, counts2[:, i], label='Unweighted PDF')
            ax[1].set_xlim([smin, smax])
            ax[1].legend()
            fig.savefig('output/' + name + '_' + str(i) + '_' +
                        str(rbins[i]) + '.png')
            plt.close('all')

        else:
            zeromask[i] = False
    #
    cmap = 'Reds'
    c_median = 'c'
    #
    plt.figure(figsize=(9, 6))
    plt.pcolormesh(rbins, sbins, counts, norm=LogNorm(), cmap=cmap, vmin=1E-3)
    plt.yscale('log')
    plt.colorbar()
    plt.plot(rbins[zeromask], dgr_median, c_median, label='Median')
    plt.ylim([1E-5, 1E-1])
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'DGR', size=16)
    plt.title('Gas mass weighted DGR PDF', size=20)
    plt.savefig('output/' + name + 'hist2d_plt_pcolormesh.png')

    plt.figure(figsize=(9, 6))
    plt.semilogy(rbins, mean_HI, label='HI')
    plt.semilogy(rbins, mean_H2, label=r'H$_2$')
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'Surface density', size=16)
    plt.title('Gas surface densities', size=20)
    plt.legend()
    plt.savefig('output/' + name + 'hist2d_plt_HIH2.png')

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
