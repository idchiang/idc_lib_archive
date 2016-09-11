import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astro_idchiang import Surveys
from astro_idchiang.testing import voronoi_2d_binning
from astro_idchiang import imshowid
from scipy.optimize import minimize
import emcee
import h5py
import astropy.units as u
from astropy.constants import c, h, k_B

THINGS_Limit = 1.0E18
objects = ['NGC 3198']
all_surveys = ['THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250', \
               'PACS_160', 'PACS_100', 'HERACLES']
all_kernels = ['Gauss_25', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
MP2 = ['THINGS', 'HERACLES']
MP1 = ['SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
fine_surveys = ['THINGS', 'SPIRE_350', 'SPIRE_250', 'PACS_160', \
                'PACS_100', 'HERACLES']
                
cmaps = Surveys(objects, all_surveys)
cmaps.add_kernel(all_kernels, 'SPIRE_500')
cmaps.matching_PSF_1step(objects, MP1, 'SPIRE_500')
cmaps.matching_PSF_2step(objects, MP2, 'Gauss_25', 'SPIRE_500')
cmaps.WCS_congrid(objects, fine_surveys, 'SPIRE_500')

"""
Current testing: Expectation Value and Corner Plot
"""
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
const = 2.0891E-4
kappa160 = 9.6
THINGS_Limit = 1.0E18
calerr_matrix2 = np.array([0.10,0.10,0.08,0.08,0.08]) ** 2
bkgerr = np.zeros(5)
theta0 = [10.0, 50.0]
ndim = len(theta0)
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

df = cmaps.df
name = 'NGC 3198'
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
for i in xrange(5):
    inv_glxmask2 = ~(np.isnan(sed[:,:,i]) + glxmask)
    temp = sed[inv_glxmask2,i]
    assert np.abs(np.mean(temp)) < np.max(temp)/10.0
    temp = temp[np.abs(temp) < (3.0*np.std(temp))]
    bkgerr[i] = np.std(temp)

sig = 1.0
bkgmask = 1.0 
for i in xrange(5):
    bkgmask *= sed[:,:,i] > (sig*bkgerr[i])

# random sampling
# NGC 3198: 
while(True):
    i = np.random.randint(0,things.shape[0])
    j = np.random.randint(0,things.shape[1])
    if diskmask[i, j]*bkgmask[i,j]:
        print '[i,j] = ['+str(i)+','+str(j)+']'
        break
testset = [[92, 91], [108,84], [99, 87], [85, 95], [101,91]]

results = []
nwalkers = 10
nsteps = 200
for testij in testset:
    i, j = testij
    """
    np.logspace(-2.3, 1.7, 20)
    """
    result = minimize(nll, theta0, args=(wl, sed[i,j], bkgerr))
    pos = np.full([nwalkers, ndim], result['x'])
    for k in xrange(ndim):
        pos[:,k] += np.random.normal(0.0, np.abs(pos[0,k]/10.0), \
                                     nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob\
                                        , args=(wl, sed[i,j], bkgerr))
        sampler.run_mcmc(pos, nsteps)
        results.append(sampler.chain)
        del sampler

import corner
# If you make use of this code, please cite the JOSS paper:

for t in xrange(len(testset)):
    i, j = testset[t]
    plt.figure()
    for w in xrange(nwalkers):
        plt.plot(results[t][w,:,0], c='b')
    plt.title('NGC 3198 ['+str(i)+','+str(j)+'] surface density')
    plt.figure()
    for w in xrange(nwalkers):
        plt.plot(results[t][w,:,1], c='b')
    plt.title('NGC 3198 ['+str(i)+','+str(j)+'] temperature')
    
    samples = results[t].reshape(-1, ndim)
    lnp = np.array([lnprob(sample, wl, sed[i,j], bkgerr) \
                    for sample in samples]).reshape(-1,1)
    samples = np.concatenate((samples, lnp), axis=1)
    
    pr = np.exp(lnp).reshape(-1)
    sexp = np.sum(samples[:,0]*pr) / np.sum(pr)
    smean = np.mean(samples[:,0])
    texp = np.sum(samples[:,1]*pr) / np.sum(pr)
    tmean = np.mean(samples[:,1])
    lnpmean = np.mean(lnp)
    x = np.linspace(70,520)
    nu = (c / x / u.um).to(u.Hz)

    sigma, T = sexp, texp
    T = T * u.K
    B = 2*h*nu**3 / c**2 / (np.exp(h*nu/k_B/T) - 1)
    B = (B.to(u.Jy)).value * 1.0E-6   # to MJy    
    modelexp = const * kappa160 * (160.0 / x)**2 * sigma * B
    sigma, T = smean, tmean
    T = T * u.K
    B = 2*h*nu**3 / c**2 / (np.exp(h*nu/k_B/T) - 1)
    B = (B.to(u.Jy)).value * 1.0E-6   # to MJy    
    modelmean = const * kappa160 * (160.0 / x)**2 * sigma * B
    plt.figure()
    plt.plot(x, modelexp, label='Exp value')
    plt.plot(x, modelmean, label='mean value')
    plt.errorbar(wl, sed[i,j], yerr = bkgerr, fmt='o', \
                 label='Data')                        
    plt.axis('tight')
    plt.legend()
    plt.title('NGC 3198 ['+str(i)+','+str(j)+']')
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel('SED')
    
    smax = np.max(samples[:,0])
    smin = np.min(samples[:,0])
    tmax = np.max(samples[:,1])+1.
    tmin = np.min(samples[:,1])-1.  
    lpmax = np.max(lnp)
    lpmin = np.min(lnp)
    fig = corner.corner(samples, bins = 50, truths=[sexp, texp, lnpmean], \
                        labels=["$\Sigma_d$", "$T$", "$\ln Prob$"], \
                        range = [(smin, smax), (tmin, tmax), (lpmin, lpmax)])
    plt.suptitle('NGC 3198 ['+str(i)+','+str(j)+']')
    print 'NGC 3198 ['+str(i)+','+str(j)+']: '
    print 's_exp = ' + str(sexp) + '; s_mean = ' + str(smean)
    print 't_exp = ' + str(texp) + '; t_mean = ' + str(tmean)
    
for t in xrange(len(testset)):
    i, j = testset[t]
    samples = results[t]
    lnp = np.array([[lnprob(samples[p, q], wl, sed[i,j], bkgerr) \
                    for q in xrange(nsteps)] for p in xrange(nwalkers)])
    plt.figure()
    for w in xrange(nwalkers):
        plt.plot(lnp[w], c='b')
    plt.title('NGC 3198 ['+str(i)+','+str(j)+'] ln Prob')

    
"""
Current testing: Voronoi binning
"""
spire500 = cmaps.df.loc['NGC 3198', 'SPIRE_500'].MAP
things = cmaps.df.loc['NGC 3198', 'THINGS'].RGD_MAP
heracles = cmaps.df.loc['NGC 3198', 'HERACLES'].RGD_MAP

nanmask = ~np.isnan(spire500)
glxmask = (things > THINGS_Limit)
diskmask = glxmask * (~(heracles*2>things)) * nanmask

inv_glxmask2 = ~(np.isnan(spire500) + glxmask)
bkg = np.std(spire500[inv_glxmask2])
calerr2 = 0.08 ** 2
for i in xrange(1,6):
    targetSN = (i)
    plt.figure()
    binNum, xBin, yBin, xBar, yBar, sn, nPixels, scale = \
        voronoi_2d_binning(x, y, signal, noise, targetSN,
                           cvt=True, pixelsize=None, plot=True,
                       quiet=False, sn_func=None, wvt=True)
    plt.suptitle('Target_SN=' + str(i))
signal = spire500[diskmask]
# noise = np.sqrt(bkg**2 + signal**2 * calerr2)
"""
For binning, use the bkgerr only
"""
"""
How in the hell am I going to bin 5 images together?
"""
noise = np.full_like(signal, bkg)
targetSN = 10
x, y = np.meshgrid(range(spire500.shape[1]), range(spire500.shape[0]))
x, y = x[diskmask], y[diskmask]
# bkgsnr = np.abs(signal / bkg)

#bkgmask = bkgsnr >= 1.5
#signal2 = signal[bkgmask]
#noise2 = noise[bkgmask]
#x2 = x[bkgmask]
#y2 = y[bkgmask]

for i in xrange(1,6):
    targetSN = (i)
    plt.figure()
    targetSN = 3
    signal += 0.05
    binNum, xBin, yBin, xBar, yBar, sn, nPixels, scale = \
        voronoi_2d_binning(x, y, signal, noise, targetSN,
                           cvt=True, pixelsize=None, plot=True,
                       quiet=False, sn_func=None, wvt=True)
    plt.suptitle('Target_SN=' + str(i))
