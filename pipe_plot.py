from idc_lib.idc_plot import Dust_Plots
# from idc_lib.idc_plot import plot_single_pixel as psp
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

"""
all_objects = ['IC2574', 'NGC0628', 'NGC0925', 'NGC2841', 'NGC2976', 'NGC3077',
               'NGC3184', 'NGC3198', 'NGC3351', 'NGC3521', 'NGC3627',
               'NGC4736', 'NGC5055', 'NGC5457', 'NGC6946', 'NGC7331']
"""

M101 = ['NGC5457']  # Currently focusing on NGC5457


# Qt: Session management error: Could not open network socket
def plot_dust(methods=['EF', 'FB', 'BEMFBFL', 'BEMFB', 'FBWD']):
    samples = ['NGC5457']
    #
    plots = Dust_Plots()
    plots.Load_and_Sum(samples, methods)


if __name__ == '__main__':
    plot_dust()
