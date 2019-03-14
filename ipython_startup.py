import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
import astropy.units as u
import astropy.constants as const
from h5py import File

if False:  # Walk around syntax check
    np.nan
    pd.Series()
    fits.getdata()
    WCS()
    Angle()
    u.pc
    const.c
    File()

print('Loaded: os, numpy=np, pandas=pd, pyplot=plt, io.fits, wcs.WCS, ' +
      'coordinates.Angle, units=u, constants=const, ' +
      'h5py.File=File')

plt.ion()
print('plt set in interaction mode')

if platform.system() == 'Windows':
    os.chdir('C:\\Users\\jiang\\Documents\\Google_Drive\\Github\\idc_lib')
print('dir changed to idc_lib')
