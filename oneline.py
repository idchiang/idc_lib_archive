from idc_lib.idc_fitting import fit_dust_density_Tmap
from idc_lib.idc_plot import plot_single_pixel
from idc_lib.idc_plot import plots_for_paper
from idc_lib.gal_data import gal_data

name = 'NGC5457'

# fit_dust_density_Tmap(name)
# plot_single_pixel(name=name, cmap0='gist_heat')
plots_for_paper()


# data = gal_data(name)
# print(data.field('DIST_MPC'), data.field('REF_DIST'), data.field('R25_DEG'))
