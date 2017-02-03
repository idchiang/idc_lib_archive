from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
from time import clock
# import emcee
from h5py import File
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import c, h, k_B
import corner
from astro_idchiang.external import voronoi_2d_binning_m, gal_data

# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
const = 2.0891E-4
kappa160 = 9.6 * np.pi # fitting uncertainty = 1.3
                       # Calibration uncertainty = 2.5
                       # 01/13/2017: pi facor added from erratum
WDC = 2900 # Wien's displacement constant (um*K)

# Column density to mass surface density M_sun/pc**2
col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value

THINGS_Limit = 1.0E18 # HERACLES_LIMIT: heracles*2 > things

FWHM = {'SPIRE_500': 36.09, 'SPIRE_350': 24.88, 'SPIRE_250': 18.15, 
        'Gauss_25': 25, 'PACS_160': 11.18, 'PACS_100': 7.04, 
        'HERACLES': 13}
fwhm_sp500 = FWHM['SPIRE_500'] * u.arcsec.to(u.rad) # in rad

# Calibration error of PACS_100, PACS_160, SPIRE_250, SPIRE_350, SPIRE_500
# For extended source
calerr_matrix2 = np.array([0.10,0.10,0.08,0.08,0.08]) ** 2

# Number of fitting parameters
ndim = 2

# Probability functions & model functions for fitting (internal)
def _B(T, freq=nu): 
    """Return blackbody SED of temperature T(with unit) in MJy"""
    return (2 * h * freq**3 / c**2 / (np.exp(h * freq / k_B / T) - 1)).\
           to(u.Jy).value * 1E-6

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
            grp.create_dataset('Radius_map', data=ept) # kpc    
        # return None
    
    fwhm_radius = fwhm_sp500 * D * 1E3 / np.cos(INCL * np.pi / 180)
    nlayers = int(np.nanmax(dp_radius) // fwhm_radius)
    masks = []
    masks.append(dp_radius < fwhm_radius)
    for i in range(1, nlayers - 1):
        masks.append((dp_radius >= i * fwhm_radius) * 
                     (dp_radius < (i + 1) * fwhm_radius))
    masks.append(dp_radius >= (nlayers - 1) * fwhm_radius)
    masks = [masks[i][diskmask] for i in range(nlayers)]
    ##### test image: original layers #####
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
    ##### test image: combined layers #####
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
                voronoi_2d_binning_m(x_l, y_l, signal_l, noise_l, targetSN, 
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
    models = np.zeros([Ts.shape[0], Ts.shape[1], 5])
    for i in range(len(wl)):
        models[:, :, i] = _model(wl[i], 10**logsigmas, Ts, nu[i])

    print("Start fitting", name, "dust surface density...")
    tic = clock()
    p = 0
    # results = [] # array for saving all the raw chains
    for i in xrange(len(binNumlist)):
        if (i + 1) / len(binNumlist) > p:
            print('Step', (i + 1), '/', len(binNumlist))
            p += 0.1
        """ Binning everything """
        bin_ = (binmap == binNumlist[i])
        radiusmap[bin_] = np.sum(dp_radius[bin_] * total_gas[bin_]) / \
                          np.sum(total_gas[bin_])  # total_gas weighted radius
        bkgerr_avg = bkgerr / np.sqrt(np.sum(bin_))
        bkgmap[bin_] = bkgerr_avg
        total_gas[bin_] = np.nanmean(total_gas[bin_])
        sed_avg[i] = np.mean(sed[bin_], axis=0)
        sed[bin_] = sed_avg[i]
        calerr2 = calerr_matrix2 * sed_avg[i]**2
        inv_sigma2 = 1 / (bkgerr_avg**2 + calerr2)
        """ Grid fitting """
        lnprobs = -0.5 * (np.sum((sed_avg[i] - models)**2 * inv_sigma2, 
                                 axis=2))
        """ Show map """
        ##plt.figure()
        ##imshowid(np.log10(-lnprobs))

        """ Randomly choosing something to plot here """
        ##if np.random.rand() > 0.0:
        ##    plot_single_bin(name, binNumlist[i], samples, sed_avg[i], 
        ##                    inv_sigma2, sopt, topt, lnprobs, Ts, logsigmas)
        """ Continue saving """
        mask = lnprobs > np.max(lnprobs) - 6
        lnprobs_cp, logsigmas_cp, Ts_cp = \
            lnprobs[mask], logsigmas[mask], Ts[mask]
        pr = np.exp(lnprobs_cp)
        #
        ids = np.argsort(logsigmas_cp)
        logsigmas_cp = logsigmas_cp[ids]
        prs = pr[ids]
        csp = np.cumsum(prs)[:-1]
        csp = np.append(0, csp / csp[-1])
        sss = np.interp([0.16, 0.5, 0.84], csp, logsigmas_cp).tolist()
        #
        idT = np.argsort(Ts_cp)
        Ts_cp = Ts_cp[idT]
        prT = pr[idT]
        csp = np.cumsum(prT)[:-1]
        csp = np.append(0, csp / csp[-1])
        sst = np.interp([0.16, 0.5, 0.84], csp, Ts_cp).tolist()
        """ Saving to results """
        popt[bin_] = np.array([sss[1], sst[1]])
        perr[bin_] = np.array([max(sss[2]-sss[1], sss[1]-sss[0]), 
                               max(sst[2]-sst[1], sst[1]-sst[0])])
        
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    # Saving to h5 file
    # Total_gas and dust in M_sun/pc**2
    # Temperature in K
    # SED in MJy/sr
    # D in Mpc
    # Galaxy_distance in 
    # Galaxy_center in pixel [y, x]
    # INCL, PA in degrees
    # PS in arcsec
    with File('output/dust_data.h5', 'a') as hf:
        grp = hf.create_group(name)
        grp.create_dataset('Total_gas', data=total_gas)
        grp.create_dataset('Herschel_SED', data=sed)
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
        grp.create_dataset('Radius_map', data=radiusmap) # kpc
    print("Datasets saved.")

## Code for plotting the results

def plot_single_bin(name, binnum, samples, sed_avg, inv_sigma2, sopt, topt, 
                    lnprobs, Ts, logsigmas):
    bins = 50
    nwalkers, nsteps, ndim = samples.shape
    lnpr = np.zeros([nwalkers, nsteps, 1])
    
    for w in xrange(nwalkers):
        for n in xrange(nsteps):
            lnpr[w, n, 0] = _lnprob(samples[w, n], wl, sed_avg, inv_sigma2)
    samples = np.concatenate([samples, lnpr], axis = 2)
        
    # Plot fitting results versus step number
    fig, ax = plt.subplots(1, 3)
    ax[0, 0].set_title('Surface density')
    ax[0, 1].set_title('Temperature')
    ax[0, 2].set_title('ln(Probability)')        
    for w in xrange(nwalkers):
        ax[0, 0].plot(samples[w,:,0], c='b')
        ax[0, 1].plot(samples[w,:,1], c='b')
        ax[0, 2].plot(samples[w,:,2], c='b')
    ax[0, 2].set_ylim(-50, np.max(samples[:,:,2]))
    fig.suptitle(name + ' bin no.' + str(binnum) + ' mcmc')
    fig.savefig('output/' + name + 'bin_' + str(binnum) + 'mcmc.png')
    fig.clf()

    # MCMC Corner plot
    samples = samples.reshape(-1, ndim + 1)
    smax = np.max(samples[:,0])
    smin = np.min(samples[:,0])
    tmax = np.max(samples[:,1]) + 1.
    tmin = np.min(samples[:,1]) - 1.
    lpmax = np.max(samples[:,2]) + 1.
    lpmin = np.max([-50., np.min(samples[:,2])])
    lnpropt = _lnprob((sopt, topt), wl, sed_avg, inv_sigma2)
    corner.corner(samples, bins = bins, truths=[sopt, topt, lnpropt], 
                  labels=["$\Sigma_d$", "$T$", "$\ln(Prob)$"], 
                  range = [(smin, smax), (tmin, tmax), (lpmin, lpmax)],
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
                               lnprobs.reshape(-1, 1)], axis = 1)
    corner.corner(samples2, bins = bins, truths=[sopt, topt, lnpropt], 
                  labels=["$\Sigma_d$", "$T$", "$\ln(Prob)$"], 
                  #range = [(smin, smax), (tmin, tmax), (lpmin, lpmax)],
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

def read_dust_file(name='NGC3198', bins=30, off=-22.5):
    # name = 'NGC3198'
    # bins = 10
    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        logs_d = np.array(grp.get('Dust_surface_density_log')) # in log
        serr = np.array(grp.get('Dust_surface_density_err_dex')) # in dex
        topt = np.array(grp.get('Dust_temperature'))
        terr = np.array(grp.get('Dust_temperature_err'))
        total_gas = np.array(grp.get('Total_gas'))
        sed = np.array(grp.get('Herschel_SED'))
        bkgerr = np.array(grp.get('Herschel_binned_bkg'))
        binmap = np.array(grp.get('Binmap'))
        radiusmap = np.array(grp.get('Radius_map')) # kpc
        D = float(np.array(grp['Galaxy_distance']))
        # readme = np.array(grp.get('readme'))

    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        things = np.array(grp['THINGS'])
        heracles = np.array(grp['HERACLES'])
        diskmask = np.array(grp['Diskmask'])

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
    # logs_HI = np.log10(things)
    logs_H2 = np.log10(heracles)
    logdgr = logs_d - logs_gas
    
    # D in Mpc. Need r25 in kpc
    R25 = gal_data([name]).field('R25_DEG')[0]
    R25 *= (np.pi / 180.) * (D * 1E3)

    # Fitting results
    fig, ax = plt.subplots(2, 2, figsize=(15,12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    ax[0, 0].set_title(r'$\Sigma_d$ $(\log_{10}(M_\odot pc^{-2}))$', size=30)
    cax[0, 0] = ax[0, 0].imshow(logs_d, origin='lower')
    ax[0, 1].set_title(r'$\Sigma_d$ error (dex)', size=30)
    cax[0, 1] = ax[0, 1].imshow(serr, origin='lower')
    ax[1, 0].set_title(r'$T_d$ ($K$)', size=30)
    cax[1, 0] = ax[1, 0].imshow(topt, origin='lower')
    ax[1, 1].set_title(r'$T_d$ error ($K$)', size=30)
    cax[1, 1] = ax[1, 1].imshow(terr, origin='lower')
    for i in range(2):
        for j in range(2):
            fig.colorbar(cax[i, j], ax=ax[i, j])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    fig.savefig('output/' + name + '_Sd_Td.png')
    fig.clf()
    
    # Total gas & DGR
    fig, ax = plt.subplots(2, 3, figsize=(20,12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    cax[0, 0] = ax[0, 0].imshow(logdgr, origin='lower')
    ax[0, 0].set_title('DGR (log)', size=20)
    cax[0, 1] = ax[0, 1].imshow(logs_gas, origin='lower')
    ax[0, 1].set_title(r'$\Sigma_{gas}$ $(\log_{10}(M_\odot pc^{-2}))$', size=20)
    cax[0, 2] = ax[0, 2].imshow(logs_H2, origin='lower')
    ax[0, 2].set_title(r'HERACLES (log, not binned)', size=20)
    cax[1, 0] = ax[1, 0].imshow(logs_d, origin='lower')
    ax[1, 0].set_title(r'$\Sigma_{d}$ $(\log_{10}(M_\odot pc^{-2}))$', size=20)
    cax[1, 1] = ax[1, 1].imshow(serr, origin='lower')
    ax[1, 1].set_title(r'$\Sigma_d$ error (dex)', size=20)
    cax[1, 2] = ax[1, 2].imshow(-lnprob, origin='lower')
    ax[1, 2].set_title(r'$\chi^2$', size=20)
    for i in range(2):
        for j in range(3):
            fig.colorbar(cax[i, j], ax=ax[i, j])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_DGR_Sd_GAS.png')
    fig.clf()
    
    r_r25 = np.array([radiusmap[binmap==temp][0] for temp in binlist]) / R25
    r_logdgr = np.array([logdgr[binmap==temp][0] for temp in binlist])
    r_logsg = np.array([logs_gas[binmap==temp][0] for temp in binlist])
    r_area = np.array([np.sum(binmap==temp) for temp in binlist])
    r_gmass = r_area * 10**r_logsg
    mask = ~np.isnan(r_r25 + r_logsg + r_area)
    r_r25, r_logdgr, r_logsg, r_area, r_gmass = \
        r_r25[mask], r_logdgr[mask], r_logsg[mask], r_area[mask], r_gmass[mask]
    # DGR profile
    fig, ax = plt.subplots(1, 3, figsize=(21,7))
    cax = np.empty_like(ax)
    # fig.set_size_inches(15, 12)
    fig.suptitle(name + ' DGR profile', size=28, y=0.995)
    ax[0].scatter(r_r25, r_logdgr)
    ax[0].set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax[0].set_ylabel(r'DGR (log)', size=16)
    ax[0].set_title('No weights', size=20)
    
    caxt = ax[1].hist2d(r_r25, r_logdgr, bins=bins, weights=r_gmass, cmin=1)
    cmax = np.nanmean(caxt[0]) + 3 * np.nanstd(caxt[0])
    cax[1] = ax[1].hist2d(r_r25, r_logdgr, bins=bins, weights=r_gmass, cmin=1, 
             cmax = cmax)
    ax[1].set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax[1].set_ylabel(r'DGR (log)', size=16)
    ax[1].set_title('Gas mass weighted', size=20)
    fig.colorbar(cax[1][3], ax=ax[1])

    caxt = ax[2].hist2d(r_r25, r_logdgr, bins=bins, weights=r_area, cmin=1)
    cmax = np.nanmean(caxt[0]) + 3 * np.nanstd(caxt[0])
    cax[2] = ax[2].hist2d(r_r25, r_logdgr, bins=bins, weights=r_gmass, cmin=1, 
             cmax = cmax)
    ax[2].set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax[2].set_ylabel(r'DGR (log)', size=16)
    ax[2].set_title('Area weighted', size=20)
    fig.colorbar(cax[2][3], ax=ax[2])
    
    fig.subplots_adjust(wspace=0.25)
    for i in range(3):
        ax[i].set_xlim([0., np.max(r_r25)+0.1])
        ax[i].set_ylim([np.min(r_logdgr)-0.1, np.max(r_logdgr)+0.1])
    fig.savefig('output/' + name + '_DGR_hist2d.png')
    fig.clf()
    plt.close("all")

def read_dust_test(name='NGC3198', bins=20, cmaxm = 3000, cmaxa = 2000, off=-22.5):
    pass
    
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