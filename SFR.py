from idc_lib.idc_io import Surveys
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


SST = ['NGC0628', 'NGC3198']  # SST for "Small Scale Test"
SSST = ['NGC0628']  # SSST for "Super Small Scale Test"
M101 = ['NGC5457']  # Currently focusing on NGC5457
all_surveys = ['THINGS', 'GALEX_FUV', 'MIPS_24', 'SPIRE_500']
all_kernels = ['Gauss_25', 'GALEX_FUV', 'MIPS_24']
MP2 = ['THINGS']
MP1 = ['GALEX_FUV', 'MIPS_24']
fine_surveys = ['THINGS', 'GALEX_FUV', 'MIPS_24']


def generator(test=0, samples=M101):
    if test:
        samples = SST
    elif type(samples) == str:
        samples = [samples]

    cmaps = Surveys(samples, all_surveys)
    cmaps.add_kernel(all_kernels, 'SPIRE_500')
    cmaps.matching_PSF_1step(samples, MP1, 'SPIRE_500')
    cmaps.matching_PSF_2step(samples, MP2, 'Gauss_25', 'SPIRE_500')
    cmaps.WCS_congrid(samples, fine_surveys, 'SPIRE_500')
    cmaps.SFR_FUV_plus_24(samples)
    cmaps.save_SFR(samples)
    return cmaps.df

df = generator()

galex = df.loc[('NGC5457', 'GALEX_FUV')].MAP
mips24 = df.loc[('NGC5457', 'MIPS_24')].MAP
logSFR = df.loc[('NGC5457', 'GALEX_FUV')].logSFR
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
fig, ax = plt.subplots(1, 3)
cax = ax[0].imshow(galex, origin='lower', norm=LogNorm())
fig.colorbar(cax, ax=ax[0])
cax = ax[1].imshow(mips24, origin='lower', norm=LogNorm())
fig.colorbar(cax, ax=ax[1])
cax = ax[2].imshow(logSFR, origin='lower')
fig.colorbar(cax, ax=ax[2])
fig.savefig('haha.png')
plt.show()

