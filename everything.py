from generating import generator
from fitting import fitting
from idc_lib.idc_fitting import fit_dust_density_Tmap
from idc_lib.idc_plot import plots_for_paper
from idc_lib.idc_plot import Residue_maps


# generator()
# fitting(fixed_beta=True, method='001111')
fit_dust_density_Tmap(method='001111')
# fitting(fixed_beta=False, method='001111')
# fitting(fixed_beta=True)
fit_dust_density_Tmap()
# fitting(fixed_beta=False)
"""
Residue_maps(method='001111')
Residue_maps()
# plots_for_paper()
"""
