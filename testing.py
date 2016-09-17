from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astro_idchiang import Surveys
import h5py

def run():
    THINGS_Limit = 1.0E18
    objects = ['NGC 3198']
    all_surveys = ['THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250', 
                   'PACS_160', 'PACS_100', 'HERACLES']
    all_kernels = ['Gauss_25', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 
                   'PACS_100']
    MP2 = ['THINGS', 'HERACLES']
    MP1 = ['SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
    fine_surveys = ['THINGS', 'SPIRE_350', 'SPIRE_250', 'PACS_160', \
                    'PACS_100', 'HERACLES']
    cmaps = Surveys(objects, all_surveys)
    cmaps.add_kernel(all_kernels, 'SPIRE_500')
    cmaps.matching_PSF_1step(objects, MP1, 'SPIRE_500')
    cmaps.matching_PSF_2step(objects, MP2, 'Gauss_25', 'SPIRE_500')
    cmaps.WCS_congrid(objects, fine_surveys, 'SPIRE_500')
    return cmaps


hf = h5py.File('output/dust_data.h5', 'r')

sig = 1.
mask = (popt[:,:,0] > sig*perr[:,:,0])*(popt[:,:,1] > sig*perr[:,:,1])
sexp = popt[:,:,0].copy()
sexp[~mask] = np.nan
texp = popt[:,:,1].copy()
texp[~mask] = np.nan