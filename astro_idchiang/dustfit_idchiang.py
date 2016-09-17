from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import numpy as np
import emcee
import h5py
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import c, h, k_B
from astro_idchiang.external import voronoi_2d_binning
from .plot_idchiang import imshowid
import corner
from time import clock

# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
const = 2.0891E-4
kappa160 = 9.6 # fitting uncertainty = 0.4; Calibration uncertainty = 2.5
WDC = 2900 # Wien's displacement constant (um*K)

THINGS_Limit = 1.0E18 # HERACLES_LIMIT: heracles*2 > things

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
    
def _lnlike(theta, x, y, bkgerr):
    """Probability function for fitting"""
    sigma, T = theta
    model = _model(x, sigma, T)
    calerr2 = calerr_matrix2 * y**2
    inv_sigma2 = 1 / (bkgerr**2 + calerr2)
    if np.sum(np.isinf(inv_sigma2)):
        return -np.inf
    else:
        return -0.5 * (np.sum((y - model)**2 * inv_sigma2 - 
                       np.log(inv_sigma2)))
        
def _lnprior(theta):
    """Probability function for fitting"""
    sigma, T = theta
    if np.log10(sigma) < 3 and 0 < T < 200:
        return 0
    return -np.inf
        
def _lnprob(theta, x, y, yerr):
    """Probability function for fitting"""
    lp = _lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _lnlike(theta, x, y, yerr)
       
def fit_dust_density(df, name, nwalkers=20, nsteps=200):
    """
    Inputs:
        df: <pandas DataFrame>
            DataFrame contains map information
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
    # Dust density in Solar Mass / pc^2
    # kappa_lambda in cm^2 / g
    # SED in MJy / sr        
    things = df.loc[(name, 'THINGS')].RGD_MAP

    # Cutting off the nan region of THINGS map.
    # [lc[0,0]:lc[0,1],lc[1,0]:lc[1,1]]
    axissum = [0] * 2
    lc = np.zeros([2,2], dtype=int)
    for i in xrange(2):
        axissum[i] = np.nansum(things, axis=i, dtype=bool)
        for j in xrange(len(axissum[i])):
            if axissum[i][j]:
                lc[i-1, 0] = j
                break
        lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)
        
    # Defining image size
    sed = np.zeros([things.shape[0], things.shape[0], 5])            
    heracles = df.loc[(name, 'HERACLES')].RGD_MAP
    sed[:, :, 0] = df.loc[(name, 'PACS_100')].RGD_MAP
    sed[:, :, 1] = df.loc[(name, 'PACS_160')].RGD_MAP
    sed[:, :, 2] = df.loc[(name, 'SPIRE_250')].RGD_MAP
    sed[:, :, 3] = df.loc[(name, 'SPIRE_350')].RGD_MAP
    sed[:, :, 4] = df.loc[(name, 'SPIRE_500')].MAP
    nanmask = ~np.sum(np.isnan(sed), axis=2, dtype=bool)
    glxmask = (things > THINGS_Limit)
    diskmask = glxmask * nanmask
            
    # Using the variance of non-galaxy region as uncertainty
    bkgerr0 = np.zeros(5)
    for i in xrange(5):
        inv_glxmask2 = ~(np.isnan(sed[:,:,i]) + glxmask)
        temp = sed[inv_glxmask2, i]
        sed[:, :, i] -= np.median(temp)
        assert np.abs(np.mean(temp)) < np.max(temp) / 10
        temp = temp[np.abs(temp) < (3 * np.std(temp))]
        bkgerr0[i] = np.std(temp)
    
    # Cut the images and masks!!!
    things = things[lc[0,0]:lc[0,1], lc[1,0]:lc[1,1]]
    heracles = heracles[lc[0,0]:lc[0,1], lc[1,0]:lc[1,1]]
    heracles[np.isnan(heracles)] = 0 # To avoid np.nan in H2 + signal in HI
    """Build a total gas surface density map here. Check the units."""
    sed = sed[lc[0,0]:lc[0,1], lc[1,0]:lc[1,1], :]
    diskmask = diskmask[lc[0,0]:lc[0,1], lc[1,0]:lc[1,1]]

    popt = np.full([sed.shape[0], sed.shape[1], ndim], np.nan)
    perr = popt.copy()
    bkgerr = np.full_like(sed, bkgerr0)
    bkgerr[~diskmask] = np.nan
    # Voronoi binning
    print("Start binning " + name + "...")
    tic = clock()
    targetSN = 5
    signal = np.min(np.abs(sed[diskmask] / bkgerr[diskmask]), axis=1)
    noise = np.ones(signal.shape)
    x, y = np.meshgrid(range(sed.shape[1]), range(sed.shape[0]))
    x, y = x[diskmask], y[diskmask]
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
        voronoi_2d_binning(x, y, signal, noise, targetSN, cvt=True, 
                           pixelsize=None, plot=True, quiet=True, 
                           sn_func=None, wvt=True)
    plt.suptitle(name + ' Voronoi bin (TargetSN=' + str(targetSN) + ')')
    plt.savefig('output/Voronoi_binning/' + str(name) + '.png')
    binmap = np.full_like(popt[:, :, 0], np.nan)
    for i in range(len(signal)):
        binmap[y[i], x[i]] = binNum[i]
    binNumlist = np.unique(binNum)
    """Modify the error map here"""
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    sed_avg = np.zeros([len(binNumlist), 5])
    
    print("Start fitting", name, "dust surface density...")
    tic = clock()

    p = 0
    # results = [] # array for saving all the raw chains
    for i in xrange(len(binNumlist)):
        if (i + 1) / len(binNumlist) > p:
            print('Step', (i + 1), '/', len(binNumlist))
            p += 0.1
        bin_ = (binmap == binNumlist[i])
        bkgerr_avg = bkgerr / np.sqrt(np.sum(bin_))
        sed_avg[i] = np.mean(sed[bin_], axis=0)
        
        temp = WDC / wl[sed_avg[i].argsort()[-2:][::-1]]
        temp0 = np.random.uniform(np.min(temp), np.max(temp), [nwalkers, 1])
        init_sig = _sigma0(wl[np.argmax(sed_avg[i])], np.max(sed_avg[i]),
                           np.mean(temp))
        sigma0 = np.random.uniform(init_sig / 2, init_sig * 2, [nwalkers, 1])
        pos = np.concatenate([sigma0, temp0], axis=1)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _lnprob,
                                        args=(wl, sed_avg[i], bkgerr_avg))
        sampler.run_mcmc(pos, nsteps)
        # results.append(sampler.chain)
        samples = sampler.chain[:, nsteps/2:, :].reshape(-1, ndim)
        pr = np.array([np.exp(_lnprob(sample, wl, sed_avg[i], bkgerr_avg)) 
                       for sample in samples])
        del sampler
        z_ptt = np.sum(pr)
        # Saving results to popt and perr
        sexp = np.sum(samples[:, 0]*pr) / z_ptt
        texp = np.sum(samples[:, 1]*pr) / z_ptt
        popt[bin_] = np.array([sexp, texp])
        serr = np.sqrt(np.sum((samples[:, 0] - sexp)**2 * pr)/z_ptt)
        terr = np.sqrt(np.sum((samples[:, 1] - texp)**2 * pr)/z_ptt)
        perr[bin_] = np.array([serr, terr])
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
        
    plt.figure()
    plt.subplot(221)
    imshowid(popt[:,:,0])
    plt.title(name + 'Dust surface density')
    plt.subplot(222)
    imshowid(popt[:,:,1])
    plt.title(name + 'Dust temperature')
    plt.subplot(223)
    plt.hist(popt[~np.isnan(popt[:,:,0]),0])
    plt.title(name + 'Dust surface density')
    plt.subplot(224)
    plt.hist(popt[~np.isnan(popt[:,:,1]),1])
    plt.title(name + 'Dust temperature')
    plt.savefig('output/'+name+'_fitting_maps.png')    
    with h5py.File('output/dust_data.h5', 'a') as hf:
        hf.create_dataset(name+'_popt', data = popt)
        hf.create_dataset(name+'_perr', data = perr)
        hf.create_dataset(name+'binmap', data = binmap)
    print("Datasets saved.")
        
    ## Code for plotting the results
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