from idc_lib.idc_dust_fitting import read_dust_file as rdf
from idc_lib.idc_dust_fitting import plot_single_pixel as psp
"""
Ref:

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
SST = ['NGC0628', 'NGC3198']  # SST for "Small Scale Test"
SSST = ['NGC0628']  # SSST for "Super Small Scale Test"
M101 = ['NGC5457']  # Currently focusing on NGC5457
all_surveys = ['THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250',
               'PACS_160', 'PACS_100', 'HERACLES', 'KINGFISH_DUST']
all_kernels = ['Gauss_25', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
MP2 = ['THINGS', 'HERACLES']
MP1 = ['SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
fine_surveys = ['THINGS', 'SPIRE_350', 'SPIRE_250', 'PACS_160',
                'PACS_100', 'HERACLES', 'KINGFISH_DUST']


# Qt: Session management error: Could not open network socket
def read(test=0, samples=M101, bins=30, off=90., cmap0='gist_heat',
         dr25=0.025, ncmode=False, cmap2='seismic', fixed_beta=None):
    if test:
        samples = SSST
    elif type(samples) == str:
        samples = [samples]
    if fixed_beta is None:
        print('Fix beta? (1 for fix, 0 for varying)')
        fixed_beta = bool(int(input()))
    for sample in samples:
        rdf(sample, bins=bins, off=off, cmap0=cmap0, dr25=dr25, ncmode=ncmode,
            cmap2=cmap2, fixed_beta=fixed_beta)

read()
