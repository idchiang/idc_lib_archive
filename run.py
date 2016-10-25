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
from astro_idchiang import Surveys, read_dust_file
from astro_idchiang import fit_dust_density as fdd
from astro_idchiang import imshowid
all_objects = ['DDO53', 'DDO154', 'HO_I', 'HO_II', 'IC_2574',
               'NGC_628', 'NGC_925', 'NGC_2841', 'NGC_2976', 'NGC_3077', 
               'NGC_3184', 'NGC_3198', 'NGC_3351', 'NGC_3521', 'NGC_3627', 
               'NGC_4736', 'NGC_5055', 'NGC_5457', 'NGC_6946', 'NGC_7331']
#all_objects = ['DDO53']
all_objects = ['NGC_628']
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
    
def fitting():
    for object_ in all_objects:
        fdd(object_, nwalkers=20, nsteps=200)
        read_dust_file(object_, bins=10, off=30)