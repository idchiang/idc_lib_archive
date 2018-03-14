# from generating import generator
import os
from idc_lib.idc_plot import Dust_Plots
# from idc_lib.idc_fitting import fit_dust_density as fdd
# import numpy as np

os.system('clear')  # on linux / os x

name = 'NGC5457'

""" 10 random points with high PL r_chi^2
plots = Dust_Plots()
for m in ['BE', 'PL']:
    plots.Load_Data(name, m)
count = 0
d = plots.d[name]
xs, ys = np.meshgrid(np.arange(d['binmap'].shape[0]),
                     np.arange(d['binmap'].shape[1]))
range_ = [i for i in range(len(d['binlist']))]
np.random.shuffle(range_)
for i in range_:
    if (d['PL']['archi2'][i] > 4) & (d['BE']['archi2'][i] < 2):
        count += 1
        mask = d['binmap'] == d['binlist'][i]
        plots.x, plots.y = xs[mask][0], ys[mask][0]
        plots.example_model(name, 'BE')
        plots.example_model(name, 'PL')
    if count == 10:
        break
"""

all_ = ['SE', 'FB', 'BE', 'WD', 'PL']
plots = Dust_Plots()
for m in all_:
    plots.Load_Data(name, m)
    plots.STBC(name, m, use_mask=True)
# plots.temperature_profiles_merge(name)
# plots.example_model_merged(name)
# plots.pdf_profiles_merge(name)
# plots.pdf_profiles(name, 'BE')
# plots.kappa160_fH2()
# plots.metallicity_contour()
