from idc_lib.idc_fitting import fit_dust_density_Tmap, fit_dust_density_Bmap
from idc_lib.idc_plot import plot_single_pixel
from idc_lib.idc_plot import plots_for_paper  # , residual_maps
from idc_lib.gal_data import gal_data
import idc_lib.idc_sandbox as sb
import idc_lib.idc_sandbox2 as sb2


name = 'NGC5457'
method = '011111'

# fit_dust_density_Bmap(name)
sb2.plots_for_paper()

# fit_dust_density_Bmap(name)
# fit_dust_density_Tmap(name)

# plot_single_pixel(name=name, cmap0='gist_heat')

# plots_for_paper(method=method)
# residual_maps(method=method)

# data = gal_data(name)

# print(data.field('DIST_MPC'), data.field('REF_DIST'), data.field('R25_DEG'))

# sb.plots_for_paper()
# sb.residual_maps()
