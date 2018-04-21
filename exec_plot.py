import os
from idc_lib.idc_plot import Dust_Plots
# import numpy as np

os.system('clear')  # on linux / os x

name = 'NGC5457'

all_ = ['SE', 'FB', 'BE', 'WD', 'PL']
plots = Dust_Plots()
for m in all_:
    plots.Load_Data(name, m)
    # plots.pdf_profiles(name, m)
    # plots.chi2_experiments(method_abbr=m)
    # plots.residual_maps(name, m)
    # plots.corner_plots(name, m)
    # plots.STBC(name, m)
# plots.temperature_profiles_merge(name)
# plots.pdf_profiles_merge(name)
# plots.pdf_profiles(name, 'BE')
# plots.residual_trend()
# plots.residual_chi2(name)
# plots.metallicity_contour()
# plots.kappa160_fH2()
plots.example_model_merged(name)
# plots.residual_map_merged()
