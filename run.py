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
from astro_idchiang import Surveys, read_dust_file, read_change_XCO, read_gases_dust
from astro_idchiang import fit_dust_density as fdd
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
        
def generator_change_XCO(objects=['NGC3198'], XCO_multiplier=0.5):
    surveys = ['THINGS', 'SPIRE_500', 'HERACLES']
    cmaps = Surveys(objects, surveys, XCO_multiplier)
    cmaps.add_kernel(['Gauss_25'], 'SPIRE_500')
    cmaps.matching_PSF_2step(objects, MP2, 'Gauss_25', 'SPIRE_500')
    cmaps.WCS_congrid(objects, MP2, 'SPIRE_500')
    cmaps.save_new_XCO(objects)

def generator_gases(objects=['NGC3198']):
    surveys = ['THINGS', 'SPIRE_500', 'HERACLES']
    cmaps = Surveys(objects, surveys)
    cmaps.add_kernel(['Gauss_25'], 'SPIRE_500')
    cmaps.matching_PSF_2step(objects, MP2, 'Gauss_25', 'SPIRE_500')
    cmaps.WCS_congrid(objects, MP2, 'SPIRE_500')
    cmaps.save_gases(objects)
    
def misc(objects = all_objects):
    for sample in objects:
        read_gases_dust(sample)
    pass