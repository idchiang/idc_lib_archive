# from generating import generator
import os
from idc_lib.idc_fitting import fit_dust_density as fdd
from idc_lib.idc_fitting import kappa_calibration


os.system('clear')  # on linux / os x

all_ = ['SE', 'FB', 'BE', 'WD', 'PL']
name = 'NGC5457'

cali = 0
f = 1

method_cali = all_
method_f = all_

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
