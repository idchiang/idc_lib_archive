from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import numpy as np
import emcee
from h5py import File
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import c, h, k_B
from astro_idchiang.external import voronoi_2d_binning
from .plot_idchiang import imshowid
# import corner
from time import clock

# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
const = 2.0891E-4
kappa160 = 9.6 # fitting uncertainty = 0.4; Calibration uncertainty = 2.5
WDC = 2900 # Wien's displacement constant (um*K)

# Column density to mass surface density M_sun/pc**2
col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value

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
       
def fit_dust_density(name, nwalkers=20, nsteps=200):
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
        # THINGS_Limit = np.array(grp['THINGS_LIMIT'])

    popt = np.full_like(sed[:, :, :ndim], np.nan)
    perr = popt.copy()
    binmap = np.full_like(sed[:, :, 0], np.nan)
    bkgmap = np.full_like(sed, np.nan)
    # Voronoi binning
    # d --> diskmasked, len() = sum(diskmask);
    # b --> binned, len() = number of binned area
    print("Start binning " + name + "...")
    tic = clock()
    signal_d = np.min(np.abs(sed[diskmask] / bkgerr), axis=1)
    noise_d = np.ones(signal_d.shape)
    temp = np.max(signal_d)
    if temp > (5 / 1.2):
        targetSN = 5
    else:
        targetSN = 1.2 * temp
    x_d, y_d = np.meshgrid(range(sed.shape[1]), range(sed.shape[0]))
    x_d, y_d = x_d[diskmask], y_d[diskmask]
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
        voronoi_2d_binning(x_d, y_d, signal_d, noise_d, targetSN, cvt=True, 
                           pixelsize=None, plot=True, quiet=True, 
                           sn_func=None, wvt=True)
    plt.suptitle(name + ' Voronoi bin (TargetSN=' + str(targetSN) + ')')
    plt.savefig('output/Voronoi_binning/' + str(name) + '.png')
    for i in range(len(signal_d)):
        binmap[y_d[i], x_d[i]] = binNum[i]
    binNumlist = np.unique(binNum)
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
        bkgmap[bin_] = bkgerr_avg
        total_gas[bin_] = np.nanmean(total_gas[bin_])
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
        samples = sampler.chain[:, nsteps//2:, :].reshape(-1, ndim)
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
        """SED not binned yet"""
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
        grp.create_dataset('Bin_xBar', data=xBar)
        grp.create_dataset('Bin_yBar', data=yBar)
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

def read_dust_file(name='NGC 3198', sig=1.0, bins=10):
    # name = 'NGC 3198'
    # bins = 10
    # sig = 1
    hf = File('output/dust_data.h5', 'r')
    grp = hf[name]
    sexp = np.array(grp.get('Dust_surface_density'))
    serr = np.array(grp.get('Dust_surface_density_err'))
    texp = np.array(grp.get('Dust_temperature'))
    terr = np.array(grp.get('Dust_temperature_err'))
    total_gas = np.array(grp.get('Total_gas'))
    # sed = np.array(grp.get('Herschel_SED'))
    # bkgerr = np.array(grp.get('Herschel_binned_bkg'))
    # binmap = np.array(grp.get('Binmap'))
    # readme = np.array(grp.get('readme'))
    hf.close()
    
    mask = (sexp < sig * serr) + (texp < sig * terr)
    sexp[mask], texp[mask], serr[mask], terr[mask] = \
        np.nan, np.nan, np.nan, np.nan
    dgr = sexp / total_gas
    
    plt.figure()
    plt.subplot(221)
    imshowid(sexp)
    plt.title('Surface density')
    plt.subplot(222)
    imshowid(texp)
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
    plt.hist(np.log10(sexp[sexp>0]), bins = bins)
    plt.title('Surface density distribution (log scale)')
    plt.subplot(122)
    plt.hist(texp[texp>0], bins = bins)
    plt.title('Temperature distribution')
    plt.suptitle(name)
    plt.savefig('output/' + name + '_fitting_hist.png')

    plt.figure()
    plt.subplot(121)
    imshowid(np.log10(dgr))
    plt.subplot(122)
    plt.hist(np.log10(dgr[dgr>0]), bins = bins)
    plt.suptitle(name + ' dust to gas ratio (log scale)')
    plt.savefig('output/' + name + '_fitting_DGR.png')

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