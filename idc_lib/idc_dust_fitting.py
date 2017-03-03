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


def fit_dust_density(name, nwalkers=20, nsteps=150):
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
        bkgerr = np.array(grp['Herschel_bkgerr'])
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
    binmap = np.full_like(sed[:, :, 0], np.nan, dtype=int)
    bkgmap = np.full_like(sed, np.nan)
    uncmap = np.full_like(sed, np.nan)
    radiusmap = np.full_like(sed[:, :, 0], np.nan)
    # Voronoi binning
    # d --> diskmasked, len() = sum(diskmask);
    # b --> binned, len() = number of binned area
    print("Start binning " + name + "...")
    tic = clock()
    signal_d = np.min(np.abs(sed[diskmask] / bkgerr), axis=1)
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
            grp.create_dataset('Herschel_binned_bkg', data=ept)
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
    min_logsigma = -5.
    max_logsigma = 3.
    T_step = 0.05
    min_T = T_step
    max_T = 50.
    logsigmas = np.arange(min_logsigma, max_logsigma, logsigma_step)
    Ts = np.arange(min_T, max_T, T_step)
    logsigmas, Ts = np.meshgrid(logsigmas, Ts)
    try:
        with File('output/rsrf_models.h5', 'r') as hf:
            models = np.array(hf['models'])
    except IOError:
        models = np.zeros([Ts.shape[0], Ts.shape[1], 5])
        """
        Applying RSRFs to generate fake-observed models
        """
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
        pacs_dnu = pacs_rsrf['dnu'].values[0]
        del pacs_rsrf
        #
        pacs_models = np.zeros([Ts.shape[0], Ts.shape[1], len(pacs_wl)])
        for i in range(len(pacs_wl)):
            pacs_models[:, :, i] = _model(pacs_wl[i], 10**logsigmas, Ts,
                                          pacs_nu[i])
        del pacs_nu
        models[:, :, 0] = np.sum(pacs_models * pacs_dnu * pacs_100,
                                 axis=2) / np.sum(pacs_dnu * pacs_100 *
                                                  pacs_wl / wl[0])
        models[:, :, 1] = np.sum(pacs_models * pacs_dnu * pacs_160,
                                 axis=2) / np.sum(pacs_dnu * pacs_160 *
                                                  pacs_wl / wl[1])
        #
        del pacs_wl, pacs_100, pacs_160, pacs_dnu, pacs_models
        ##
        print("Constructing SPIRE RSRF model...")
        spire_rsrf = pd.read_csv("data/RSRF/SPIRE_RSRF.csv")
        spire_wl = spire_rsrf['Wavelength'].values
        spire_nu = (c / spire_wl / u.um).to(u.Hz)
        spire_250 = spire_rsrf['SPIRE_250'].values
        spire_350 = spire_rsrf['SPIRE_350'].values
        spire_500 = spire_rsrf['SPIRE_500'].values
        spire_dnu = spire_rsrf['dnu'].values[0]
        del spire_rsrf
        #
        spire_models = np.zeros([Ts.shape[0], Ts.shape[1], len(spire_wl)])
        for i in range(len(spire_wl)):
            spire_models[:, :, i] = _model(spire_wl[i], 10**logsigmas, Ts,
                                           spire_nu[i])
        del spire_nu
        models[:, :, 2] = np.sum(spire_models * spire_dnu * spire_250,
                                 axis=2) / np.sum(spire_dnu * spire_250 *
                                                  spire_wl / wl[2])
        models[:, :, 3] = np.sum(spire_models * spire_dnu * spire_350,
                                 axis=2) / np.sum(spire_dnu * spire_350 *
                                                  spire_wl / wl[3])
        models[:, :, 4] = np.sum(spire_models * spire_dnu * spire_500,
                                 axis=2) / np.sum(spire_dnu * spire_500 *
                                                  spire_wl / wl[4])
        #
        del spire_wl, spire_250, spire_350, spire_500, spire_dnu
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
    for i in xrange(len(binNumlist)):
        if (i + 1) / len(binNumlist) > p:
            print('Step', (i + 1), '/', len(binNumlist))
            p += 0.1
        """ Binning everything """
        bin_ = (binmap == binNumlist[i])
        radiusmap[bin_] = np.sum(dp_radius[bin_] * total_gas[bin_]) / \
            np.sum(total_gas[bin_])   # total_gas weighted radius
        bkgerr_avg = bkgerr / np.sqrt(np.sum(bin_))
        unc_avg = np.sqrt(np.mean(sed_unc[bin_]**2, axis=0))
        unc_avg[np.isnan(unc_avg)] = 0
        bkgmap[bin_] = bkgerr_avg
        uncmap[bin_] = unc_avg
        total_gas[bin_] = np.nanmean(total_gas[bin_])
        sed_avg[i] = np.mean(sed[bin_], axis=0)
        sed[bin_] = sed_avg[i]
        calerr2 = calerr_matrix2 * sed_avg[i]**2
        inv_sigma2 = 1 / (bkgerr_avg**2 + calerr2 + unc_avg**2)
        """ Grid fitting """
        lnprobs = -0.5 * (np.sum((sed_avg[i] - models)**2 * inv_sigma2,
                                 axis=2))
        """ Show map """
        # plt.figure()
        # imshowid(np.log10(-lnprobs))

        """ Randomly choosing something to plot here """
        # if np.random.rand() > 0.0:
        #     plot_single_bin(name, binNumlist[i], samples, sed_avg[i],
        #                     inv_sigma2, sopt, topt, lnprobs, Ts, logsigmas)
        """ Continue saving """
        pr = np.exp(lnprobs)
        mask = lnprobs > np.nanmax(lnprobs) - 6
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
        """ Saving to results """
        popt[bin_] = np.array([sss[1], sst[1]])
        perr[bin_] = np.array([max(sss[2]-sss[1], sss[1]-sss[0]),
                               max(sst[2]-sst[1], sst[1]-sst[0])])
        """ New: saving PDF """
        pdf = np.sum(pr, axis=0)
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
        grp.create_dataset('Herschel_SED_binned_unc', data=uncmap)
        grp.create_dataset('Herschel_binned_bkg', data=bkgmap)
        grp.create_dataset('Dust_surface_density_log', data=popt[:, :, 0])
        # sopt in log scale (search sss)
        grp.create_dataset('Dust_surface_density_err_dex', data=perr[:, :, 0])
        # serr in dex
        grp.create_dataset('Dust_temperature', data=popt[:, :, 1])
        grp.create_dataset('Dust_temperature_err', data=perr[:, :, 1])
        grp.create_dataset('Binmap', data=binmap)
        grp.create_dataset('Galaxy_distance', data=D)
        grp.create_dataset('Galaxy_center', data=glx_ctr)
        grp.create_dataset('INCL', data=INCL)
        grp.create_dataset('PA', data=PA)
        grp.create_dataset('PS', data=PS)
        grp.create_dataset('Radius_map', data=radiusmap)  # kpc
        grp.create_dataset('logsigmas', data=logsigmas)
    print("Datasets saved.")


# Code for plotting the results
def plot_single_bin(name, binnum, samples, sed_avg, inv_sigma2, sopt, topt,
                    lnprobs, Ts, logsigmas):
    bins = 50
    nwalkers, nsteps, ndim = samples.shape
    lnpr = np.zeros([nwalkers, nsteps, 1])

    for w in xrange(nwalkers):
        for n in xrange(nsteps):
            lnpr[w, n, 0] = _lnprob(samples[w, n], wl, sed_avg, inv_sigma2)
    samples = np.concatenate([samples, lnpr], axis=2)

    # Plot fitting results versus step number
    fig, ax = plt.subplots(1, 3)
    ax[0, 0].set_title('Surface density')
    ax[0, 1].set_title('Temperature')
    ax[0, 2].set_title('ln(Probability)')
    for w in xrange(nwalkers):
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
    """
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


def read_dust_file(name='NGC3198', bins=30, off=-22.5, cmap0='gist_heat',
                   dr25=0.025):
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

    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        things = np.array(grp['THINGS']) * col2sur * H2HaHe
        heracles = np.array(grp['HERACLES'])
        total_gas_ub = np.array(grp['Total_gas'])
        diskmask = np.array(grp['Diskmask'])
        dp_radius = np.array(grp['DP_RADIUS'])

    pdfs = pd.read_csv('output/' + name + '_pdf.csv')

    nanmask = np.isnan(total_gas) + np.isnan(things) + np.isnan(heracles)
    total_gas[nanmask], things[nanmask], heracles[nanmask] = -1., -1., -1.
    total_gas[np.less_equal(total_gas, 0)] = np.nan
    things[np.less_equal(things, 0)] = np.nan
    heracles[np.less_equal(heracles, 0)] = np.nan

    lnprob = np.full_like(logs_d, np.nan)
    binlist = np.unique(binmap[diskmask])
    for bin_ in binlist:
        mask = binmap == bin_
        calerr2 = calerr_matrix2 * sed[mask][0]**2
        inv_sigma2 = 1 / (bkgerr[mask][0]**2 + calerr2)
        lnprob[mask] = _lnprob([10**logs_d[mask][0], topt[mask][0]], wl,
                               sed[mask][0], inv_sigma2)
        if lnprob[mask][0] < off or serr[mask][0] > 1.:
            logs_d[mask], topt[mask], serr[mask], terr[mask], lnprob[mask] = \
                np.nan, np.nan, np.nan, np.nan, np.nan
            total_gas[mask], things[mask], heracles[mask] = \
                np.nan, np.nan, np.nan

    logs_gas = np.log10(total_gas)
    logs_HI = np.log10(things)
    logs_H2 = np.log10(heracles)
    logdgr = logs_d - logs_gas

    # D in Mpc. Need r25 in kpc
    R25 = gal_data([name]).field('R25_DEG')[0]
    R25 *= (np.pi / 180.) * (D * 1E3)

    # Fitting results
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    ax[0, 0].set_title(r'$\Sigma_d$ $(\log_{10}(M_\odot pc^{-2}))$', size=30)
    cax[0, 0] = ax[0, 0].imshow(logs_d, origin='lower', cmap=cmap0)
    ax[0, 1].set_title(r'$\Sigma_d$ error (dex)', size=30)
    cax[0, 1] = ax[0, 1].imshow(serr, origin='lower', cmap=cmap0)
    ax[1, 0].set_title(r'$T_d$ ($K$)', size=30)
    cax[1, 0] = ax[1, 0].imshow(topt, origin='lower', cmap=cmap0)
    ax[1, 1].set_title(r'$T_d$ error ($K$)', size=30)
    cax[1, 1] = ax[1, 1].imshow(terr, origin='lower', cmap=cmap0)
    for i in range(2):
        for j in range(2):
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
    cax[1, 2] = ax[1, 2].imshow(-lnprob, origin='lower', cmap=cmap0)
    ax[1, 2].set_title(r'$\chi^2$', size=20)
    for i in range(2):
        for j in range(3):
            fig.colorbar(cax[i, j], ax=ax[i, j])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_DGR_Sd_GAS.png')
    fig.clf()

    # Reducing data to 1-dim
    r_r25 = np.array([radiusmap[binmap == temp][0] for temp in binlist]) / R25
    r_logdgr = np.array([logdgr[binmap == temp][0] for temp in binlist])
    r_logsg = np.array([logs_gas[binmap == temp][0] for temp in binlist])
    r_logHI = np.array([logs_HI[binmap == temp][0] for temp in binlist])
    r_logH2 = np.array([logs_H2[binmap == temp][0] for temp in binlist])
    r_area = np.array([np.sum(binmap == temp) for temp in binlist])

    nanmask = np.isnan(r_logsg)
    r_logsg[nanmask] = 0.
    r_gmass = r_area * 10**r_logsg
    r_gmass[nanmask] = np.nan
    mask = ~np.isnan(r_r25 + r_logsg + r_area + r_gmass)
    r_r25, r_logdgr, r_logsg, r_area, r_gmass, r_logHI, r_logH2 = \
        r_r25[mask], r_logdgr[mask], r_logsg[mask], r_area[mask], \
        r_gmass[mask], r_logHI[mask], r_logH2[mask]

    # Creating mass / ring mass weighting
    dr = np.max(r_r25) / bins
    masks = [(r_r25 < dr)]
    for i in range(1, bins - 1):
        masks.append((r_r25 >= i * dr) * (r_r25 < (i + 1) * dr))
    masks.append(r_r25 >= (bins - 1) * dr)
    r_gmassdtm = np.empty_like(r_gmass)
    for i in range(bins):
        r_gmassdtm[masks[i]] = r_gmass[masks[i]] / np.sum(r_gmass[masks[i]])

    # DGR profile
    fig, ax = plt.subplots(1, 2, figsize=(21, 7))
    cax = np.empty_like(ax)
    # fig.set_size_inches(15, 12)
    cax[0] = ax[0].hist2d(r_r25, r_logdgr, bins=bins, weights=r_gmass,
                          cmap='Greys', norm=LogNorm())
    ax[0].set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax[0].set_ylabel(r'DGR (log)', size=16)
    ax[0].set_title('Gas mass weighted', size=20)
    fig.colorbar(cax[0][3], ax=ax[0])

    cax[1] = ax[1].hist2d(r_r25, r_logdgr, bins=bins, weights=r_gmassdtm,
                          cmap='Greys', norm=LogNorm())
    ax[1].set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax[1].set_ylabel(r'DGR (log)', size=16)
    ax[1].set_title('Ring-normalized gas mass weighted', size=20)
    fig.colorbar(cax[1][3], ax=ax[1])

    fig.subplots_adjust(wspace=0.25)
    for i in range(2):
        ax[i].set_xlim([0., np.max(r_r25)+0.1])
        ax[i].set_ylim([np.min(r_logdgr)-0.1, np.max(r_logdgr)+0.1])
    fig.savefig('output/' + name + '_DGR_hist2d.png')
    fig.clf()

    # Redistributing data to rings
    nlayers = int(np.max(r_r25) // dr25)
    masks = [(r_r25 < dr25)]
    for i in range(1, nlayers - 1):
        masks.append((r_r25 >= i * dr25) * (r_r25 < (i + 1) * dr25))
    masks.append(r_r25 >= (nlayers - 1) * dr25)
    masks = np.array(masks)
    tm = np.array([np.sum(r_gmass[masks[i]]) for i in range(len(masks))])
    masks, tm = masks[tm.astype(bool)], tm[tm.astype(bool)]
    nlayers = len(masks)
    r_ri, dgr_ri, HI_ri, H2_ri = np.empty(nlayers), np.empty(nlayers), \
        np.empty(nlayers), np.empty(nlayers)
    for i in range(nlayers):
        mask = masks[i]
        r_ri[i] = np.sum(r_r25[mask] * r_gmass[mask]) / tm[i]
        dgr_ri[i] = np.log10(np.sum(10**r_logdgr[mask] * r_gmass[mask]) /
                             tm[i])
        with np.errstate(invalid='ignore'):
            HI_ri[i] = np.log10(np.sum(10**r_logHI[mask] * r_gmass[mask]) /
                                tm[i])
        H2_ri[i] = np.log10(np.sum(10**r_logH2[mask] * r_gmass[mask]) / tm[i])

    fig, ax = plt.subplots(2, 1, figsize=(10, 15))
    # fig.set_size_inches(15, 12)
    # Note: it's the ring-mass normalized one here
    cax = ax[0].hist2d(r_r25, r_logdgr, bins=bins, weights=r_gmassdtm,
                       cmap='Greys', norm=LogNorm())
    # fig.colorbar(cax[3], ax=ax[0])
    ax[0].plot(r_ri, dgr_ri, lw=5.)
    ax[0].set_ylabel(r'DGR (log)', size=16)
    ax[0].set_title(name + 'Gas mass weighted profile', size=24, y=1.05)

    ax[1].plot(r_ri, HI_ri, label="HI", lw=5.)
    ax[1].plot(r_ri, H2_ri, label="H2", lw=5.)
    ax[1].set_ylabel(r'Surface density (log)', size=16)
    ax[1].set_xlabel(r'r25', size=16)
    ax[1].legend(loc=3, fontsize=16)
    for i in range(2):
        ax[i].set_xlim([0., np.max(r_r25)+0.1])
    fig.savefig('output/' + name + '_DGR_profile.png')
    fig.clf()
    plt.close("all")


def vs_KINGFISH(name='NGC5457', targetSNR=10, dr25=0.025):
    df = pd.DataFrame()

    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        with np.errstate(invalid='ignore'):
            df['dust_fit'] = \
                10**(np.array(grp.get('Dust_surface_density_log')).flatten())
        serr = np.array(grp.get('Dust_surface_density_err_dex')).flatten()
        D = float(np.array(grp['Galaxy_distance']))
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        df['dust_kf'] = np.array(grp.get('KINGFISH')).flatten() / 1E6
        df['snr_kf'] = df['dust_kf'] / \
            np.array(grp.get('KINGFISH_unc')).flatten() * 1E6
        df['total_gas'] = np.array(grp.get('Total_gas')).flatten()
        radius = np.array(grp.get('DP_RADIUS')).flatten()

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
