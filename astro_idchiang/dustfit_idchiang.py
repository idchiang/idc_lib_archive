from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import numpy as np
from time import clock
import emcee
from h5py import File
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import c, h, k_B
# import corner
from astro_idchiang.external import voronoi_2d_binning_m
from .plot_idchiang import imshowid

# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
const = 2.0891E-4
kappa160 = 9.6 # fitting uncertainty = 0.4; Calibration uncertainty = 2.5
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

def fit_dust_density(name, nwalkers=10, nsteps=500, nrounds=2, 
                     lim_lnprob=-10):
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
            plt.figure()
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
    
    print("Start fitting", name, "dust surface density...")
    tic = clock()

    best_lnpr = np.zeros(len(binNumlist))
    first_time_lnpr = np.zeros(len(binNumlist))
    final_runstep = np.zeros(len(binNumlist))
    p = 0
    # results = [] # array for saving all the raw chains
    for i in xrange(len(binNumlist)):
        if (i + 1) / len(binNumlist) > p:
            print('Step', (i + 1), '/', len(binNumlist))
            p += 0.1
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
        temp = WDC / wl[sed_avg[i].argsort()[-2:][::-1]]
        temp0 = np.random.uniform(np.min(temp), np.max(temp), [nwalkers, 1])
        init_sig = _sigma0(wl[np.argmax(sed_avg[i])], np.max(sed_avg[i]),
                           np.mean(temp))
        sigma0 = np.random.uniform(init_sig / 2, init_sig * 2, [nwalkers, 1])
        pos = np.concatenate([sigma0, temp0], axis=1)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _lnprob,
                                        args=(wl, sed_avg[i], inv_sigma2))
        round_ = 0
        max_lnpr = -1000
        while((max_lnpr < lim_lnprob) & (round_ < nrounds)):
            print("Round", round_ + 1)
            sampler.run_mcmc(pos, nsteps)
            # results.append(sampler.chain)
            samples = sampler.chain[:, sampler.chain.shape[1] // 2:, :] \
                      .reshape(-1, ndim)
            pr = np.array([np.exp(_lnprob(sample, wl, sed_avg[i], inv_sigma2)) 
                           for sample in samples]).reshape(-1, 1)
            max_lnpr = np.log(np.max(pr))
            if not first_time_lnpr[i]:
                first_time_lnpr[i] = max_lnpr
            round_ += 1
        best_lnpr[i] = max_lnpr
        final_runstep[i] = round_ * nsteps
        print('First time lnpr:', first_time_lnpr[i], '; Final lnpr:',
              best_lnpr[i], '; Final run steps:', final_runstep[i])
        # Saving results to popt and perr
        sss = np.percentile(samples[:, 0], [16, 50, 84])
        sst = np.percentile(samples[:, 1], [16, 50, 84])
        popt[bin_] = np.array([sss[1], sst[1]])
        perr[bin_] = np.array([max(sss[2]-sss[1], sss[1]-sss[0]), 
                               max(sst[2]-sst[1], sst[1]-sst[0])])
        
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    
    mask = final_runstep > nsteps
    plt.figure()
    plt.plot(final_runstep[mask], (best_lnpr[mask] - first_time_lnpr[mask]))
    plt.xlabel('Final runsteps')
    plt.ylabel('Improvement in lnpr')
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
    """
    testnums = np.random.randint(0,len(signal),5)

    for t in xrange(len(testnums)):
        yt, xt = y[testnums[t]], x[testnums[t]]
        r_index = np.argmax(binNumlist == binNum[testnums[t]])
        samples = results[r_index]
        lnpr = np.zeros([nwalkers, nsteps, 1])
        for w in xrange(nwalkers):
            for n in xrange(nsteps):
                lnpr[w, n] = _lnprob(samples[w, n], wl, sed_avg[r_index], 
                                     bkgerr)
        pr = np.exp(lnpr)
        samples = np.concatenate([samples, lnpr], axis = 2)
        
        # Plot fitting results versus step number
        plt.figure()
        plt.subplot(131)
        for w in xrange(nwalkers):
            plt.plot(samples[w,:,0], c='b')
        plt.title('Surface density')
        plt.subplot(132)
        for w in xrange(nwalkers):
            plt.plot(samples[w,:,1], c='b')
        plt.title('Temperature')
        plt.subplot(133)
        for w in xrange(nwalkers):
            plt.plot(samples[w,:,2], c='b')
        plt.ylim(-50,np.max(samples[w,:,2]))
        plt.title('ln(Probability)')        
        plt.suptitle('NGC 3198 ['+str(yt)+','+str(xt)+']')
        plt.savefig('output/NGC 3198 ['+str(yt)+','+str(xt)+']_results.png')

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
        plt.errorbar(wl, sed_avg[r_index], yerr = bkgerr, fmt='ro', \
                     label='Data')
        plt.axis('tight')
        plt.legend()
        plt.title('NGC 3198 ['+str(yt)+','+str(xt)+']')
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel('SED')
        plt.savefig('output/NGC 3198 ['+str(yt)+','+str(xt)+']_datamodel.png')

        # Corner plot
        smax = np.max(samples[:,0])
        smin = np.min(samples[:,0])
        tmax = np.max(samples[:,1]) + 1.
        tmin = np.min(samples[:,1]) - 1.
        lpmax = np.max(samples[:,2]) + 1.
        lpmin = np.max([-50., np.min(samples[:,2])])
        lnprexp = _lnprob((sexp, texp), wl, sed_avg[r_index], bkgerr)
        corner.corner(samples, bins = 50, truths=[sexp, texp, lnprexp], 
                      labels=["$\Sigma_d$", "$T$", "$\ln(Prob)$"], 
                      range = [(smin, smax), (tmin, tmax), (lpmin, lpmax)])
        plt.suptitle('NGC 3198 ['+str(yt)+','+str(xt)+']')
        plt.savefig('output/NGC 3198 ['+str(yt)+','+str(xt)+']_corner.png')
    """

def fit_dust_density_grid(name):
    """
    Inputs:
        name: <str>
            Object name to be calculated.
    Outputs (file):
        name_popt: <numpy array>
            Optimized parameters
        name_perr: <numpy array>
            Error of optimized parameters
    """
    # import and define
    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        sopt = np.array(grp.get('Dust_surface_density'))
        serr = np.array(grp.get('Dust_surface_density_err'))
        topt = np.array(grp.get('Dust_temperature'))
        terr = np.array(grp.get('Dust_temperature_err'))
        # total_gas = np.array(grp.get('Total_gas'))
        sed = np.array(grp.get('Herschel_SED'))
        bkgmap = np.array(grp.get('Herschel_binned_bkg'))
        binmap = np.array(grp.get('Binmap'))
        # radiusmap = np.array(grp.get('Radius_map')) # kpc
    sopt2 = np.empty_like(sopt)
    serr2 = np.empty_like(serr)
    topt2 = np.empty_like(topt)
    terr2 = np.empty_like(terr)
    mask = np.full_like(sopt, True, dtype=bool)
    off = -22.5

    ## Build all models.
    ## Maybe just save them?
    ## log(density): -5 ~ 3, spacing 0.01
    ## temperature: 0 ~ 50, spacing 0.1
    print("Generating models...")
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
    print("Start fitting...")
    tic = clock()
    p = 0
    binNumlist = np.unique(binmap)
    for i in xrange(len(binNumlist)):
        if (i + 1) / len(binNumlist) > p:
            print('Step', (i + 1), '/', len(binNumlist))
            p += 0.1
        bin_ = (binmap == binNumlist[i])
        bkgerr = bkgmap[bin_][0]
        sed_avg = sed[bin_][0]
        calerr2 = calerr_matrix2 * sed_avg**2
        inv_sigma2 = 1 / (bkgerr**2 + calerr2)
        """Probability function for fitting"""
        lnprobs = -0.5 * (np.sum((sed_avg - models)**2 * inv_sigma2, axis=2))
        am = np.argmax(lnprobs)
        sopt2[bin_] = 10**logsigmas.flatten()[am]
        topt2[bin_] = Ts.flatten()[am]
        if np.max(lnprobs) > off:
            mask[bin_] = False
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    
    """
    with File('output/dust_data.h5', 'a') as hf:
        grp = hf[name]
        grp.create_dataset('Dust_surface_density_grid', data=sopt2)
        grp.create_dataset('Dust_temperature_grid', data=topt2)
    print("Datasets saved.")
    """
    sopt2[mask] = np.nan
    topt2[mask] = np.nan
    plt.figure()
    plt.subplot(221)
    imshowid(sopt2)
    plt.title('Density')
    plt.subplot(222)
    imshowid(topt2)
    plt.title('Temeprature')
    plt.subplot(223)
    plt.scatter(sopt, sopt2)
    plt.xlabel('MCMC')
    plt.ylabel('Grid')
    plt.subplot(224)
    plt.scatter(topt, topt2)
    plt.xlabel('MCMC')
    plt.ylabel('Grid')
    plt.suptitle(name)
    plt.savefig('output/' + name + '_grid_vs_mcmc.png')

def read_dust_file(name='NGC_3198', bins=10, off=-22.5):
    # name = 'NGC_3198'
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
        
    plt.figure()
    imshowid(lnprob)
    plt.title('lnprob')
    
    dgr = sopt / total_gas
    
    plt.figure()
    plt.subplot(221)
    imshowid(sopt)
    plt.title('Surface density')
    plt.subplot(222)
    imshowid(topt)
    plt.title('Temperature')
    plt.subplot(223)
    imshowid(serr)
    plt.title('Surface density uncertainty')
    plt.subplot(224)
    imshowid(terr)
    plt.title('Temperature uncertainty')
    plt.suptitle(name)
    plt.savefig('output/' + name + '_fitting_results.png')
   
    plt.figure()
    plt.subplot(121)
    plt.hist(np.log10(sopt[sopt>0]), bins = bins, weights=total_gas[sopt>0])
    plt.title('Weighted surface density (log scale)')
    plt.subplot(122)
    plt.hist(topt[topt>0], bins = bins, weights=total_gas[topt>0])
    plt.title('Weighted temperature')
    plt.suptitle(name)
    plt.savefig('output/' + name + '_weighted_hist.png')

    plt.figure()
    plt.subplot(131)
    imshowid(np.log10(total_gas))
    plt.title('Total gas (log)')
    plt.subplot(132)
    imshowid(np.log10(dgr))
    plt.title('DGR')
    plt.subplot(133)
    plt.hist(np.log10(dgr[dgr>0]), bins = bins)
    plt.title('DGR')
    plt.suptitle(name + ' dust to gas ratio (log scale)')
    plt.savefig('output/' + name + '_fitting_DGR.png')

    plt.figure()
    plt.subplot(121)
    plt.scatter(total_gas.flatten(), sopt.flatten())
    plt.xlabel(r'Total gas surface mass density ($M_\odot/pc^2$)')
    plt.ylabel(r'Dust surface mass density ($M_\odot/pc^2$)')
    plt.title('Linear scale')    
    plt.subplot(122)
    plt.scatter(np.log10(total_gas.flatten()), np.log10(sopt.flatten()))
    plt.xlabel(r'Total gas surface mass density ($M_\odot/pc^2$)')
    plt.ylabel(r'Dust surface mass density ($M_\odot/pc^2$)')
    plt.title('Log scale')
    plt.suptitle('Total gas vs. dust')

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