from idc_lib.idc_io import MGS
import matplotlib.pyplot as plt
import numpy as np
"""
all_objects = ['IC2574', 'NGC0628', 'NGC0925', 'NGC2841', 'NGC2976', 'NGC3077',
               'NGC3184', 'NGC3198', 'NGC3351', 'NGC3521', 'NGC3627',
               'NGC4736', 'NGC5055', 'NGC5457', 'NGC6946', 'NGC7331']
SST = ['NGC0628', 'NGC3198']  # SST for "Small Scale Test"
SSST = ['NGC0628']  # SSST for "Super Small Scale Test"
"""
M101 = ['NGC5457']  # Currently focusing on NGC5457
all_surveys = ['THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250',
               'PACS_160', 'PACS_100', 'HERACLES', 'MIPS_24', 'GALEX_FUV',
               'IRAC_3.6']
all_kernels = ['Gauss_25', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100',
               'IRAC_3.6', 'MIPS_24', 'GALEX_FUV']
fine_surveys = ['THINGS', 'SPIRE_350', 'SPIRE_250', 'PACS_160',
                'PACS_100', 'HERACLES', 'IRAC_3.6', 'MIPS_24', 'GALEX_FUV']
crop_surveys = ['THINGS', 'HERACLES', 'HERSCHEL_011111', 'HERSCHEL_001111',
                'IRAC_3.6', 'MIPS_24', 'GALEX_FUV']
cut_surveys = ['RADIUS_KPC']
bkg_rm_surveys = ['THINGS', 'HERACLES', 'HERSCHEL_011111', 'HERSCHEL_001111',
                  'IRAC_3.6', 'MIPS_24', 'GALEX_FUV']
save_surveys = ['THINGS', 'HERACLES', 'HERSCHEL_011111', 'HERSCHEL_001111',
                'RADIUS_KPC', 'SFR', 'SMSD', 'TOTAL_GAS', 'DIST_MPC', 'PA_RAD',
                'cosINCL', 'R25_KPC', 'SPIRE_500_PS']

THINGS_Limit = 1.0E18
samples = ['NGC5457']
mgs = MGS(samples, all_surveys)
mgs.add_kernel(all_kernels, 'SPIRE_500')
mgs.matching_PSF(samples, fine_surveys, 'SPIRE_500')
mgs.WCS_congrid(samples, fine_surveys, 'SPIRE_500')

galex = mgs.df.loc['NGC5457']['GALEX_FUV']
tbkgmask = ~(mgs.df.loc['NGC5457']['GALEX_FUV'] > THINGS_Limit)
radius = mgs.df.loc['NGC5457']['RADIUS_KPC']

