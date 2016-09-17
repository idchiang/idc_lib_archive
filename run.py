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
from astro_idchiang import Surveys

def running():
    THINGS_objects = ['NGC 628', 'NGC 925', 'NGC 1569', 'NGC 2366', 'NGC 2403',
                      'Ho II', 'M81 DwA', 'DDO53', 'NGC 2841', 'NGC 2903',
                      'Ho I', 'NGC 2976', 'NGC 3031', 'NGC 3077', 'M81 DwB',
                      'NGC 3184', 'NGC 3198', 'IC 2574', 'NGC 3351', 
                      'NGC 3521', 'NGC 3621', 'NGC 3627', 'NGC 4214', 
                      'NGC 4449', 'NGC 4736', 'DDO154', 'NGC 4826', 'NGC 5055', 
                      'NGC 5194', 'NGC 5236', 'NGC 5457', 'NGC 6946', 
                      'NGC 7331', 'NGC 7793']
    objects = ['NGC 3198', 'NGC 628']
    all_surveys = ['THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250', 
                   'PACS_160', 'PACS_100', 'HERACLES']
    all_kernels = ['Gauss_25', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
    MP2 = ['THINGS', 'HERACLES']
    MP1 = ['SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
    fine_surveys = ['THINGS', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 
                    'PACS_100', 'HERACLES']
                
    cmaps = Surveys(objects, all_surveys)
    cmaps.add_kernel(all_kernels, 'SPIRE_500')
    cmaps.matching_PSF_1step(objects, MP1, 'SPIRE_500')
    cmaps.matching_PSF_2step(objects, MP2, 'Gauss_25', 'SPIRE_500')
    cmaps.WCS_congrid(objects, fine_surveys, 'SPIRE_500')
    cmaps.fit_dust_density(objects)