from __future__ import absolute_import, division, print_function, \
                       unicode_literals
from astro_idchiang import Surveys, read_dust_file
from astro_idchiang import fit_dust_density as fdd
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


all_objects = ['IC2574', 'NGC0628', 'NGC0925', 'NGC2841', 'NGC2976', 'NGC3077',
               'NGC3184', 'NGC3198', 'NGC3351', 'NGC3521', 'NGC3627',
               'NGC4736', 'NGC5055', 'NGC5457', 'NGC6946', 'NGC7331']
all_surveys = ['THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250',
               'PACS_160', 'PACS_100', 'HERACLES']
all_kernels = ['Gauss_25', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
MP2 = ['THINGS', 'HERACLES']
MP1 = ['SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
fine_surveys = ['THINGS', 'SPIRE_350', 'SPIRE_250', 'PACS_160',
                'PACS_100', 'HERACLES']


def generator(samples=all_objects):
    cmaps = Surveys(samples, all_surveys)
    cmaps.add_kernel(all_kernels, 'SPIRE_500')
    cmaps.matching_PSF_1step(samples, MP1, 'SPIRE_500')
    cmaps.matching_PSF_2step(samples, MP2, 'Gauss_25', 'SPIRE_500')
    cmaps.WCS_congrid(samples, fine_surveys, 'SPIRE_500')
    cmaps.save_data(samples)


def fitting(samples=all_objects, nwalkers=10, nsteps=500, bins=30, off=-22.5):
    samples = [samples] if type(samples) == str else samples
    for sample in samples:
        fdd(sample, nwalkers=nwalkers, nsteps=nsteps)
        read_dust_file(sample, bins=bins, off=off)


def misc(samples=['NGC0628'], nwalkers=10, nsteps=500, bins=30, off=-22.5,
         cmap0='gist_heat', dr25=0.025):
    samples = [samples] if type(samples) == str else samples
    for sample in samples:
        read_dust_file(sample, bins=bins, off=off, cmap0=cmap0, dr25=dr25)
