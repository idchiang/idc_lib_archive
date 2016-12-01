from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
# execfile('IDC_astro.py')
"""
Ref:

emcee: 
    1. cite http://adsabs.harvard.edu/abs/2013PASP..125..306F
    2. consider adding your paper to the Testimonials list.
       http://dan.iel.fm/emcee/current/testimonials/#testimonials

Voronoi binning:
    1. acknowledgment to use of
       `the Voronoi binning method by Cappellari & Copin (2003)'.
       http://adsabs.harvard.edu/abs/2003MNRAS.342..345C

Corner:
    1. Cite the JOSS paper
       http://dx.doi.org/10.21105/joss.00024
"""
import matplotlib
matplotlib.use('Agg')
from astro_idchiang import Surveys, read_dust_file
from astro_idchiang import fit_dust_density as fdd
from astro_idchiang import imshowid
problem = ['NGC6946']
               
all_objects = ['IC2574', 'NGC0628', 'NGC0925', 'NGC2841', 'NGC2976', 'NGC3077', 
               'NGC3184', 'NGC3198', 'NGC3351', 'NGC3521', 'NGC3627', 
               'NGC4736', 'NGC5055', 'NGC5457', 'NGC7331']
all_surveys = ['THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250', 
               'PACS_160', 'PACS_100', 'HERACLES']
all_kernels = ['Gauss_25', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
MP2 = ['THINGS', 'HERACLES']
MP1 = ['SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
fine_surveys = ['THINGS', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 
                'PACS_100', 'HERACLES']

def generator():
    cmaps = Surveys(all_objects, all_surveys)
    cmaps.add_kernel(all_kernels, 'SPIRE_500')
    cmaps.matching_PSF_1step(all_objects, MP1, 'SPIRE_500')
    cmaps.matching_PSF_2step(all_objects, MP2, 'Gauss_25', 'SPIRE_500')
    cmaps.WCS_congrid(all_objects, fine_surveys, 'SPIRE_500')
    cmaps.save_data(all_objects)
    
def fitting(nwalkers=20, nsteps=150):
    for object_ in all_objects:
        fdd(object_, nwalkers=10, nsteps=500)
        read_dust_file(object_, bins=10, off=-22.5)

def test(n=10):
    # n = 10
    nwalkers = 20
    nsteps = 150
    ndim = 2
    logsigmas, Ts, lnprobs, topt, sopt, wl, sed_avg, inv_sigma2 = fdd('IC2574')
    from time import clock
    import numpy as np
    import emcee
    ## MCMC
    tic = clock()
    for i in range(n):
        print(i)
        temp0 = np.random.uniform(topt * 0.99, topt * 1.01, [nwalkers, 1])
        sigma0 = np.random.uniform(sopt * 0.99, sopt * 1.01, [nwalkers, 1])
        pos = np.concatenate([sigma0, temp0], axis=1)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _lnprob,
                                        args=(wl, sed_avg, inv_sigma2))
        sampler.run_mcmc(pos, nsteps)
        samples = sampler.chain[:, 50:, :].reshape(-1, ndim)
        sss = np.percentile(samples[:, 0], [16, 50, 84])
        sst = np.percentile(samples[:, 1], [16, 50, 84])
        """ Saving to results """
        ans = [max(sss[2]-sss[1], sss[1]-sss[0]), max(sst[2]-sst[1], sst[1]-sst[0])]
        del sampler
        del ans
    toc = clock()
    print("MCMC:", toc-tic)
    tic = clock()
    for i in range(n):
        print(i)
        mask = lnprobs > np.max(lnprobs) - 6
        lnprobs_cp, logsigmas_cp, Ts_cp = lnprobs[mask], logsigmas[mask], Ts[mask]
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
        ans = [max(sss[2]-sss[1], sss[1]-sss[0]), max(sst[2]-sst[1], sst[1]-sst[0])]
        del ans
    toc = clock()
    print("GRID:", toc-tic)