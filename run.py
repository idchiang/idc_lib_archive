# execfile('IDC_astro.py')

from astro_idchiang.io import Surveys

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

"""
self.matching_PSF_2step(objects, MP2, 'Gauss_25', 'SPIRE_500', False)
self.matching_PSF_1step(objects, MP1, 'SPIRE_500', False)
self.WCS_congrid(objects, fine_surveys, 'SPIRE_500', method = 'linear')
"""
# cmaps.fit_dust_density(objects)