from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import matplotlib
matplotlib.use('Agg')
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
    if np.log10(sigma) < 3 and 0 < T < 200:
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
    judgement = np.min(np.abs(np.sum(signal_d)) / np.sqrt(len(signal_d)))
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
    ##### test image: original layers #####
    """    
    image_test1 = np.full_like(dp_radius, np.nan)
    for i in range(nlayers):
        image_test1[masks[i]] = i
    """
    #######################################
    
    for i in range(nlayers - 1, -1, -1):
        judgement = np.min(np.abs(np.sum(sed[masks[i]], axis=0)) / 
                           np.sqrt(np.sum(masks[i])) / bkgerr)
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
        sss = np.interp([0.16, 0.5, 0.84], csp, 10**logsigmas_cp).tolist()
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
        grp.create_dataset('Dust_surface_density', data=popt[:, :, 0])
        grp.create_dataset('Dust_surface_density_err', data=perr[:, :, 0])
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

def read_dust_file(name='NGC3198', bins=10, off=-22.5):
    # name = 'NGC3198'
    # bins = 10
    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        sopt = np.array(grp.get('Dust_surface_density'))
        serr = np.array(grp.get('Dust_surface_density_err'))
        topt = np.array(grp.get('Dust_temperature'))
        terr = np.array(grp.get('Dust_temperature_err'))
        total_gas = np.array(grp.get('Total_gas'))
        sed = np.array(grp.get('Herschel_SED'))
        bkgerr = np.array(grp.get('Herschel_binned_bkg'))
        binmap = np.array(grp.get('Binmap'))
        radiusmap = np.array(grp.get('Radius_map')) # kpc
        D = float(np.array(grp['Galaxy_distance']))
        # readme = np.array(grp.get('readme'))
    
    lnprob = np.full_like(sopt, np.nan)
    binlist = np.unique(binmap[~np.isnan(binmap)])
    for bin_ in binlist:
        mask = binmap == bin_
        calerr2 = calerr_matrix2 * sed[mask][0]**2
        inv_sigma2 = 1 / (bkgerr[mask][0]**2 + calerr2)
        lnprob[mask] = _lnprob([sopt[mask][0], topt[mask][0]], wl, 
                               sed[mask][0], inv_sigma2)
        if lnprob[mask][0] < off:
            sopt[mask], topt[mask], serr[mask], terr[mask], lnprob[mask] = \
                np.nan, np.nan, np.nan, np.nan, np.nan
    
    dgr = sopt / total_gas

    # Fitting results
    fig, ax = plt.subplots(2, 2, figsize=(15,12))
    cax = np.empty_like(ax)
    ax[0, 0].set_title('Surface density', size=30)
    ax[0, 1].set_title('Surface density uncertainty', size=30)
    ax[1, 0].set_title('Temperature', size=30)
    ax[1, 1].set_title('Temperature uncertainty', size=30)
    cax[0, 0] = ax[0, 0].imshow(sopt, origin='lower')
    cax[0, 1] = ax[0, 1].imshow(serr, origin='lower')
    cax[1, 0] = ax[1, 0].imshow(topt, origin='lower')
    cax[1, 1] = ax[1, 1].imshow(terr, origin='lower')
    for i in range(2):
        for j in range(2):
            fig.colorbar(cax[i, j], ax=ax[i, j])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    fig.savefig('output/' + name + '_fitting_results.png')
    fig.clf()
    
    # Total gas & DGR
    fig, ax = plt.subplots(2, 2, figsize=(15,12))
    # fig.set_size_inches(15, 12)
    cax = np.empty_like(ax)
    ax[0, 0].set_title('Total gas (log)', size=30)
    ax[0, 1].set_title('DGR (log)', size=30)
    ax[1, 0].set_title('Surface density', size=30)
    ax[1, 1].set_title('Temperature', size=30)
    cax[0, 0] = ax[0, 0].imshow(np.log10(total_gas), origin='lower')
    cax[0, 1] = ax[0, 1].imshow(np.log10(dgr), origin='lower')
    cax[1, 0] = ax[1, 0].imshow(sopt, origin='lower')
    cax[1, 1] = ax[1, 1].imshow(topt, origin='lower')
    for i in range(2):
        for j in range(2):
            fig.colorbar(cax[i, j], ax=ax[i, j])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    fig.savefig('output/' + name + '_fitting_results2.png')
    fig.clf()
    
    # Weighted histogram
    fig, ax = plt.subplots(1, 3, figsize=(13,7))
    # fig.set_size_inches(15, 12)
    ax[0].set_title('Gas weighted surface density', size=18, y=1.04)
    ax[1].set_title('Gas weighted temperature', size=18, y=1.04)
    ax[2].set_title('Gas weighted DGR', size=18, y=1.04)
    ax[0].hist(np.log10(sopt[sopt>0]), bins = bins, weights=total_gas[sopt>0])
    ax[1].hist(topt[topt>0], bins = bins, weights=total_gas[topt>0])
    ax[2].hist(np.log10(dgr[dgr>0]), bins = bins, weights=total_gas[dgr>0])
    ax[0].set_xlabel(r'Surface density ($\ln(M_\odot/pc^2)$)', size=14)
    ax[1].set_xlabel('Temperature (K)', size=14)
    ax[2].set_xlabel('DGR (log scale)', size=14)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_weighted_hist.png')
    fig.clf()
    
    # Gas versus dust
    fig, ax = plt.subplots(1, 2, figsize=(12,7))
    # fig.set_size_inches(15, 12)
    ax[0].set_title('Linear scale', size=20)
    ax[1].set_title('Log scale', size=20)
    fig.suptitle('Total gas vs. dust', size=28, y=1.001)
    ax[0].scatter(total_gas.flatten(), sopt.flatten())
    ax[1].scatter(np.log10(total_gas.flatten()), np.log10(sopt.flatten()))
    ax[0].set_xlabel(r'Total gas surface mass density ($M_\odot/pc^2$)', size=16)
    ax[0].set_ylabel(r'Dust surface mass density ($M_\odot/pc^2$)', size=16)
    ax[1].set_xlabel(r'Total gas surface mass density ($M_\odot/pc^2$)', size=16)
    ax[1].set_ylabel(r'Dust surface mass density ($M_\odot/pc^2$)', size=16)
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_gas_vs_dust.png')
    fig.clf()

    # D in Mpc. Need r25 in kpc
    r25 = gal_data([name]).field('R25_DEG')[0]
    r25 *= (np.pi / 180.) * (D * 1E3)
    binlist = np.unique(binmap)
    r_reduced = np.array([radiusmap[binmap==temp][0] for temp in binlist])
    dgr_reduced = np.array([dgr[binmap==temp][0] for temp in binlist])
    # DGR profile
    fig, ax = plt.subplots(1, 2, figsize=(12,7))
    # fig.set_size_inches(15, 12)
    fig.suptitle('DGR profile', size=28)
    ax[0].scatter(r_reduced, dgr_reduced)
    ax[1].scatter(r_reduced / r25, dgr_reduced)
    ax[0].set_xlabel(r'Radius ($kpc$)', size=16)
    ax[0].set_ylabel(r'Dust-to-Gas-Ratio', size=16)
    ax[1].set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax[1].set_ylabel(r'Dust-to-Gas-Ratio', size=16)
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_dgr_profile.png')
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

def read_change_XCO(name='NGC3198', XCO1=0.5, XCO2=2.0):
    # name = 'NGC3198'
    # bins = 10
    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        sopt = np.array(grp.get('Dust_surface_density'))
        # serr = np.array(grp.get('Dust_surface_density_err'))
        total_gas = np.array(grp.get('Total_gas'))
        binmap = np.array(grp.get('Binmap'))
        radiusmap = np.array(grp.get('Radius_map')) # kpc
        D = float(np.array(grp['Galaxy_distance']))
        # readme = np.array(grp.get('readme'))

    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        total_gas1 = np.array(grp['Total_gas_XCOM=' + str(round(XCO1, 2))])
        total_gas2 = np.array(grp['Total_gas_XCOM=' + str(round(XCO2, 2))])
        # THINGS_Limit = np.array(grp['THINGS_LIMIT'])

    binlist = np.unique(binmap[~np.isnan(binmap)])
    for bin_ in binlist:
          mask = binmap == bin_
          total_gas1[mask] = np.nanmean(total_gas1[mask])
          total_gas2[mask] = np.nanmean(total_gas2[mask])
          
    dgr = sopt / total_gas
    dgr1 = sopt / total_gas1
    dgr2 = sopt / total_gas2
    
    # D in Mpc. Need r25 in kpc
    r25 = gal_data([name]).field('R25_DEG')[0]
    r25 *= (np.pi / 180.) * (D * 1E3)
    r_reduced = np.array([radiusmap[binmap==temp][0] for temp in binlist])
    dgr_reduced = np.array([dgr[binmap==temp][0] for temp in binlist])
    dgr1_reduced = np.array([dgr1[binmap==temp][0] for temp in binlist])
    dgr2_reduced = np.array([dgr2[binmap==temp][0] for temp in binlist])
    # DGR profile
    fig, ax = plt.subplots(1, 2, figsize=(12,7))
    # fig.set_size_inches(15, 12)
    ax[0].scatter(r_reduced, dgr_reduced, c='b', label='XCO=2E20')
    ax[0].scatter(r_reduced, dgr1_reduced, c='r', label='XCO=1E20')
    ax[0].scatter(r_reduced, dgr2_reduced, c='g', label='XCO=4E20')
    ax[1].scatter(r_reduced / r25, dgr_reduced, c='b', label='XCO=2E20')
    ax[1].scatter(r_reduced / r25, dgr1_reduced, c='r', label='XCO=1E20')
    ax[1].scatter(r_reduced / r25, dgr2_reduced, c='g', label='XCO=4E20')
    ax[0].set_xlabel(r'Radius ($kpc$)', size=16)
    ax[0].set_ylabel(r'Dust-to-Gas-Ratio', size=16)
    ax[1].set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax[1].set_ylabel(r'Dust-to-Gas-Ratio', size=16)
    ax[0].legend()
    ax[1].legend()
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_dgr_changing_XCO.png')
    fig.clf()
    
    plt.close("all")
    
def read_gases_dust(name='NGC3198'):
    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        sopt = np.array(grp.get('Dust_surface_density'))
        # serr = np.array(grp.get('Dust_surface_density_err'))
        total_gas = np.array(grp.get('Total_gas'))
        binmap = np.array(grp.get('Binmap'))
        radiusmap = np.array(grp.get('Radius_map')) # kpc
        D = float(np.array(grp['Galaxy_distance']))
        # readme = np.array(grp.get('readme'))

    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        things = np.array(grp['THINGS'])
        heracles = np.array(grp['HERACLES'])
        # THINGS_Limit = np.array(grp['THINGS_LIMIT'])

    binlist = np.unique(binmap[~np.isnan(binmap)])
    for bin_ in binlist:
          mask = binmap == bin_
          things[mask] = np.nanmean(things[mask])
          heracles[mask] = np.nanmean(heracles[mask])
          
    dgr = sopt / total_gas
    
    # D in Mpc. Need r25 in kpc
    r_red = np.array([radiusmap[binmap==temp][0] for temp in binlist])
    dgr_red = np.array([dgr[binmap==temp][0] for temp in binlist])
    things_red = np.array([things[binmap==temp][0] for temp in binlist])
    heracles_red = np.array([heracles[binmap==temp][0] for temp in binlist])
    sopt_red = np.array([sopt[binmap==temp][0] for temp in binlist])
    mask = np.argsort(r_red)
    r_red = r_red[mask]
    dgr_red = dgr_red[mask]
    things_red = things_red[mask]
    heracles_red = heracles_red[mask]
    sopt_red = sopt_red[mask]
    mg = np.max([np.nanmax(things_red), np.nanmax(heracles_red)])
    # DGR profile
    fig, ax = plt.subplots(2, 1, figsize=(12,12))
    # fig.set_size_inches(15, 12)
    ax[0].scatter(r_red, dgr_red, s=15)
    ax[1].scatter(r_red, sopt_red / np.nanmax(sopt_red), c='b', label='Dust', s=15)
    ax[1].scatter(r_red, things_red / mg, c='r', label='HI', s=15)
    ax[1].scatter(r_red, heracles_red / mg, c='y', label='H2', s=15)
    ax[0].set_xlabel(r'Radius ($kpc$)', size=16)
    ax[0].set_ylabel(r'Dust-to-Gas-Ratio', size=16)
    ax[1].set_xlabel(r'Radius ($kpc$)', size=16)
    ax[1].set_ylabel(r'Max-normalized surface density', size=16)
    ax[1].legend()
    # fig.tight_layout()
    fig.suptitle(name, size=24)
    fig.savefig('output/' + name + '_all_gases_and_dusts.png')
    fig.clf()
    
    plt.close("all")