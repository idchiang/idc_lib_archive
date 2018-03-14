# from generating import generator
import os
from idc_lib.idc_fitting import fit_dust_density as fdd
from idc_lib.idc_plot import Dust_Plots
from idc_lib.idc_fakegen import fake_generation
from idc_lib.idc_fitting import kappa_calibration


os.system('clear')  # on linux / os x

all_ = ['SE', 'FB', 'BE', 'WD', 'PL']
name = 'NGC5457'

cali = 0
f = 1
p = 1
fakegen = 0
fakefit = 0
fakeplot = 0

method_cali = all_
method_f = all_
method_p = all_
method_fg = ['PL']
method_ff = all_
method_fp = all_

nop = 10

if cali:
    for method_abbr in method_cali:
        quiet = False if method_abbr == 'PL' else True
        cov_mode = 5
        kappa_calibration(method_abbr, cov_mode=cov_mode, nop=nop,
                          quiet=quiet)

if f:
    for method_abbr in method_f:
        fdd(name, cov_mode=True, method_abbr=method_abbr, del_model=True,
            nop=nop)
print('')

if p:
    plots = Dust_Plots()
    # for m in method_p:
    #     plots.Load_Data(name, m)
    plots.Load_and_Sum([name], method_p)
    plots.extra_plots(name)

print('')
if fakegen:
    for method_abbr in method_fg:
        fake_generation(method_abbr=method_abbr)

print('')
if fakefit:
    for method_abbr in method_ff:
        fdd(name, cov_mode=True, method_abbr=method_abbr, del_model=True,
            fake=True)

print('')
if fakeplot:
    plots = Dust_Plots(fake=True)
    plots.Load_and_Sum([name], method_fp)
    plots.extra_plots(name)
print('')
