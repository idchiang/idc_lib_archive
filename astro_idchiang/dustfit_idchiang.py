import numpy as np
from scipy.optimize import minimize
import emcee
import h5py
import astropy.units as u
from astropy.constants import c, h, k_B

# Dust fitting constants
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
const = 2.0891E-4
kappa160 = 9.6
# fitting uncertainty = 0.4
# Calibration uncertainty = 2.5
THINGS_Limit = 1.0E18
# HERACLES_LIMIT: heracles*2 > things
calerr_matrix2 = np.array([0.10,0.10,0.08,0.08,0.08]) ** 2
# Calibration error of PACS_100, PACS_160, SPIRE_250, SPIRE_350, SPIRE_500
# For extended source
bkgerr = np.zeros(5)
# sigma, T
theta0 = [10.0, 50.0]
ndim = len(theta0)
# Probability functions for fitting
def lnlike(theta, x, y, bkgerr):
    """Probability function for fitting"""
    sigma, T = theta
    T = T * u.K
    nu = (c / x / u.um).to(u.Hz)
            
    B = 2*h*nu**3 / c**2 / (np.exp(h*nu/k_B/T) - 1)
    B = (B.to(u.Jy)).value * 1.0E-6   # to MJy    
        
    model = const * kappa160 * (160.0 / x)**2 * sigma * B
    calerr2 = calerr_matrix2 * y**2
    inv_sigma2 = 1.0/(bkgerr**2 + calerr2)
    
    if np.sum(np.isinf(inv_sigma2)):
        return -np.inf
    else:
        return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
        
def lnprior(theta):
    """Probability function for fitting"""
    sigma, T = theta
    if np.log10(sigma) < 3.0 and 0 < T < 200:
        return 0.0
    return -np.inf
        
def lnprob(theta, x, y, yerr):
    """Probability function for fitting"""
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
            
nll = lambda *args: -lnlike(*args)

def fit_dust_density(df, name, nwalkers = 10, nsteps = 200):
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
    axissum = [0]*2
    lc = np.zeros([2,2], dtype=int)
    for i in xrange(2):
        axissum[i] = np.nansum(things, axis=i, dtype = bool)
        for j in xrange(len(axissum[i])):
            if axissum[i][j]:
                lc[i-1, 0] = j
                break
        lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)
        
    # Defining image size
    l = df.loc[(name, 'SPIRE_500')].L
            
    sed = np.zeros([l[0], l[1], 5])
    popt = np.full([l[0], l[1], ndim], np.nan)
    perr = popt.copy()
            
    heracles = df.loc[(name, 'HERACLES')].RGD_MAP
    sed[:,:,0] = df.loc[(name, 'PACS_100')].RGD_MAP
    sed[:,:,1] = df.loc[(name, 'PACS_160')].RGD_MAP
    sed[:,:,2] = df.loc[(name, 'SPIRE_250')].RGD_MAP
    sed[:,:,3] = df.loc[(name, 'SPIRE_350')].RGD_MAP
    sed[:,:,4] = df.loc[(name, 'SPIRE_500')].MAP
    nanmask = ~np.sum(np.isnan(sed), axis=2, dtype=bool)
    glxmask = (things > THINGS_Limit)
    diskmask = glxmask * (~(heracles*2>things)) * nanmask
            
    # Using the variance of non-galaxy region as uncertainty
    for i in xrange(5):
        inv_glxmask2 = ~(np.isnan(sed[:,:,i]) + glxmask)
        temp = sed[inv_glxmask2,i]
        assert np.abs(np.mean(temp)) < np.max(temp)/10.0
        temp = temp[np.abs(temp) < (3.0*np.std(temp))]
        bkgerr[i] = np.std(temp)

    # Build a mask which only keeps region larger than sig*bkgerr
    sig = 1.0
    bkgmask = 1.0 
    for i in xrange(5):
        bkgmask *= sed[:,:,i] > (sig*bkgerr[i])

            
    # Random sampling for [i,j] with high SNR
    # i, j = 95, 86  # NGC 3198
    """
    while(True):
        i = np.random.randint(0,things.shape[0])
        j = np.random.randint(0,things.shape[1])
        if diskmask[i, j]*bkgmask[i,j]:
            print '[i,j] = ['+str(i)+','+str(j)+']'
            break
    """
            
    for i in xrange(l[0]):
        for j in xrange(l[1]):
            print name + ': (' + str(i+1) + '/' + str(l[0]) +', ' +\
                str(j+1) + '/' + str(l[1]) + ')'
            if diskmask[i, j]:
                result = minimize(nll, theta0, args=(wl, sed[i,j], bkgerr))
                pos = np.full([nwalkers, ndim], result['x'])
                for k in xrange(ndim):
                    pos[:,k] += np.random.normal(0.0, np.abs(pos[0,k]/10.0), \
                                nwalkers)
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob\
                              , args=(wl, sed[i,j], bkgerr))
                    sampler.run_mcmc(pos, nsteps)
                        
                    temp = sampler.chain[:,nsteps//2:,:].reshape(-1,ndim)
                    # sampler.chain in shape (nwalkers, nsteps, ndim)
                    del sampler
                    popt[i,j] = np.mean(temp, axis = 0)
                    perr[i,j] = np.std(temp, axis = 0)
                    ## Code for plotting the results
                    """
                    temp = np.mean(sampler.chain, axis = 0)
                    plt.figure()
                    plt.suptitle('NGC 3198 [95,86] (10 walkers)')
                    plt.subplot(121)
                    plt.plot(temp[:,0])
                    plt.ylabel(r'Dust surface mass density (M$_\odot$/pc$^2$)')
                    plt.xlabel('Run')
                    plt.subplot(122)
                    plt.plot(temp[:,1])
                    plt.ylabel('Temperature (T)')
                    plt.xlabel('Run')
                    """
                    ## Code for plotting model versus data
                    """
                    x = np.linspace(70,520)
                    sigma, T = popt[i,j]
                    T = T * u.K
                    nu = (c / x / u.um).to(u.Hz)
                    B = 2*h*nu**3 / c**2 / (np.exp(h*nu/k_B/T) - 1)
                    B = (B.to(u.Jy)).value * 1.0E-6   # to MJy    
                    model = const * kappa160 * (160.0 / x)**2 * sigma * B
                    plt.figure()
                    plt.plot(x, model, label='Model')
                    plt.errorbar(wl, sed[i,j], yerr = bkgerr, fmt='o', \
                                     label='Data')                        
                    plt.axis('tight')
                    plt.legend()
                    plt.title('NGC 3198 [95,86] (10 walkers)')
                    plt.xlabel(r'Wavelength ($\mu$m)')
                    plt.ylabel('SED')
                    """
    with h5py.File('outputs/dust_data.h5', 'a') as hf:
        hf.create_dataset(name+'_popt', data=popt)
        hf.create_dataset(name+'_perr', data=perr)
        
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