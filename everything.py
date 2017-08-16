from generating import generator
from fitting import fitting
from idc_lib.idc_fitting import fit_dust_density_Tmap
from idc_lib.idc_plot import plots_for_paper

name = 'NGC5457'

generator()
fitting(fixed_beta=True)
fitting(fixed_beta=False)
fit_dust_density_Tmap(name)
plots_for_paper()
