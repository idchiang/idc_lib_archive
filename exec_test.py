import os
from idc_lib.idc_fitting import SE_calibration_vs_G14

os.system('clear')  # on linux / os x

name = 'NGC5457'

df = SE_calibration_vs_G14()

"""
from astropy.modeling.blackbody import blackbody_nu, blackbody_lambda
from idc_lib.idc_functions import B_fast
from astropy.constants import c, N_A
import astropy.units as u
import numpy as np

wl_complete = np.linspace(80, 1000, 1000)
c_ums = c.to(u.um / u.s).value
T = 17.2
beta = 1.96
"""
