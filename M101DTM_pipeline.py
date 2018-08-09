# from generating import generator
import os
from idc_lib.idc_io import MGS
from idc_lib.idc_fitting_old import fit_dust_density as fdd
from idc_lib.idc_fitting_old import kappa_calibration


os.system('clear')  # on linux / os x

reading = False
calibrating = False
fitting = True

nop = 10

name = 'NGC5457'
all_surveys = ['THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250',
               'PACS_160', 'PACS_100', 'HERACLES', 'MIPS_24', 'IRAC_3.6',
               'GALEX_FUV']
all_kernels = ['Gauss_25', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100',
               'IRAC_3.6', 'MIPS_24', 'GALEX_FUV']
fine_surveys = ['THINGS', 'SPIRE_350', 'SPIRE_250', 'PACS_160',
                'PACS_100', 'HERACLES', 'IRAC_3.6', 'MIPS_24', 'GALEX_FUV']
cut_surveys = ['RADIUS_KPC']
cov_surveys = ['HERSCHEL_011111', 'HERSCHEL_001111']
crop_surveys = ['THINGS', 'HERACLES', 'HERSCHEL_011111', 'HERSCHEL_001111',
                'IRAC_3.6', 'MIPS_24', 'GALEX_FUV']
save_surveys = ['THINGS', 'HERACLES', 'HERSCHEL_011111', 'HERSCHEL_001111',
                'RADIUS_KPC', 'SFR', 'SMSD', 'TOTAL_GAS', 'DIST_MPC', 'PA_RAD',
                'cosINCL', 'R25_KPC', 'SPIRE_500_PS']
all_methods = ['SE', 'FB', 'BE', 'WD', 'PL']
method_cali = all_methods
method_f = all_methods

if reading:
    samples = [name]
    mgs = MGS(samples, all_surveys)
    mgs.add_kernel(all_kernels, 'SPIRE_500')
    mgs.matching_PSF(samples, fine_surveys, 'SPIRE_500')
    mgs.WCS_congrid(samples, fine_surveys, 'SPIRE_500')
    mgs.covariance_matrix(samples, cov_surveys)
    mgs.crop_image(samples, crop_surveys)
    mgs.crop_image(samples, cut_surveys, unc=False)
    mgs.SFR(samples)
    mgs.SMSD(samples)
    mgs.total_gas(samples)
    mgs.save_data(samples, save_surveys)

if calibrating:
    for method_abbr in method_cali:
        quiet = False if method_abbr == 'PL' else True
        cov_mode = 5
        kappa_calibration(method_abbr, cov_mode=cov_mode, nop=nop,
                          quiet=quiet)

if fitting:
    for method_abbr in method_f:
        fdd(name, cov_mode=True, method_abbr=method_abbr, del_model=False,
            nop=nop)
