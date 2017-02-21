"""
PACKAGE NAME:
    idc_lib

AUTHOR:
    I-Da Chiang (jiang696@gmail.com)

PURPOSE:
    1. Calculate total flux and radial profile of a survey
    2. Convolve, matching, regridding, and cutting maps from different surveys

HISTORY:
    06/30/2016, ver. 0.0
        -- File created.
    07/01/2016, ver. 1.0
        -- Radial profile and mass calculation finished.
    07/06/2016, ver. 1.0.1
        -- Fixing mass calculation: including cos(Dec) correction in area.
    07/20/2016, ver. 1.1
        -- Adding matching WCS regridding function and cutting images
    07/27/2016, ver. 1.2
        -- Adding convolution, importing/creating kernel
        -- Multi-galaxies comparison possible
        -- Restructuring the entire code: comments and debugging
        -- Unify method for calculating pixel scale
    08/03/2016, ver. 1.2.1
        -- Fixing FWHM --> sigma errors
    ??/??/????, ver. ?.?.?
        -- Transform into package
        -- Finish dust fitting functions

CURRENT TO DO:
    1. Dust surface mass density fitting
    2. Method: generate kernel, convolution, regrid for all surveys
    3. Combine data files

FUTURE TO DO:
    1. Normalization of the regrid map if already in "density"
    2. add_kernel: checking centered
"""
from gal_data import *
from idc_dust_fitting import *
from idc_io import *
from idc_math import *
from idc_plot import *
from idc_regrid import *
from idc_voronoi import *
