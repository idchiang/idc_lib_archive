from idc_lib.idc_fitting import fit_dust_density as fdd
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
M101 = ['NGC5457']  # Currently focusing on NGC5457


def fitting(samples=M101, cov_mode=True, fixed_beta=None, method='011111'):
    if (fixed_beta is None):
        print('Fixing beta? (1 for fix, 0 for varying)')
        fixed_beta = bool(int(input()))
    for sample in samples:
        fdd(sample, fixed_beta=fixed_beta, method=method)


if __name__ == "__main__":
    fitting()
