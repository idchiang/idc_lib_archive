import os
from idc_lib.idc_plot_old import Dust_Plots
# import numpy as np

os.system('clear')  # on linux / os x

name = 'NGC5457'

# all_ = ['SE', 'FB', 'BE', 'WD', 'PL']
all_ = ['BE']

plots = Dust_Plots()
for m in all_:
    plots.Load_Data(name, m)
    # plots.pdf_profiles(name, m)
    # plots.corner_plots(name, m)
    # plots.STBC_uncver(name, m)
    # plots.realize_PDF(name, m)

# plots.example_model_merged(name)
# plots.pdf_profiles_merge(name)
# plots.temperature_profiles_merge(name)
# plots.temperature_profiles_talk(name)
# plots.pdf_profiles_talk(name)
# plots.residual_map_merged()
# plots.residual_chi2(name)

# plots.voronoi_plot()
# plots.pdf_profiles(name, 'BE')
# plots.kappa160_fH2()
# plots.residual_trend_half()

# plots.alpha_CO_test()
# plots.pdf_profiles_four_beta()
# plots.unbinned_and_binned_gas_masp()
# for i in range(20):
#     plots.realize_vs_simple_sum(rbin_num=i)
# plots.realize_vs_Mgas_PDF2()

# plots.D14_pdf_profiles_DUS19(name, 'BE')
plots.C18_datatable_DUS19()
