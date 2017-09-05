from idc_lib.idc_fitting import fit_dust_density_Tmap
from idc_lib.idc_plot import plot_single_pixel
from idc_lib.idc_plot import plots_for_paper, Residue_maps
from idc_lib.gal_data import gal_data
import idc_lib.idc_sandbox as sb


name = 'NGC5457'
method = '001111'

# fit_dust_density_Tmap(name)

# plot_single_pixel(name=name, cmap0='gist_heat')

# plots_for_paper(method=method)

# data = gal_data(name)

# print(data.field('DIST_MPC'), data.field('REF_DIST'), data.field('R25_DEG'))

Residue_maps(method=method)
Residue_maps()
