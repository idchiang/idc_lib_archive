from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from astro_idchiang import Surveys, imshowid
from h5py import File

def run():
    objects = ['DDO53']
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

def read_dust_file(name='NGC 3198', sig=1.0, bins=10):
    # name = 'NGC 3198'
    # bins = 10
    # sig = 1
    hf = File('output/dust_data.h5', 'r')
    grp = hf[name]
    binmap = np.array(grp.get('Binmap'))
    sexp = np.array(grp.get('Dust_surface_density'))
    serr = np.array(grp.get('Dust_surface_density_err'))
    texp = np.array(grp.get('Dust_temperature'))
    terr = np.array(grp.get('Dust_temperature_err'))
    total_gas = np.array(grp.get('Total_gas'))
    readme = np.array(grp.get('readme'))
    hf.close()
    
    mask = (sexp < sig * serr) + (texp < sig * terr)
    sexp[mask], texp[mask], serr[mask], terr[mask] = \
        np.nan, np.nan, np.nan, np.nan
    dgr = sexp / total_gas
    
    plt.figure()
    plt.subplot(221)
    imshowid(sexp)
    plt.title('Surface density')
    plt.subplot(222)
    imshowid(texp)
    plt.title('Temperature')
    plt.subplot(223)
    imshowid(serr)
    plt.title('Surface density uncertainty')
    plt.subplot(224)
    imshowid(terr)
    plt.title('Temperature uncertainty')
    plt.suptitle(name)
    plt.savefig('output/' + name + '_fitting_results.png')
    
    plt.figure()
    plt.subplot(121)
    plt.hist(np.log10(sexp[sexp>0]), bins = bins)
    plt.title('Surface density distribution (log scale)')
    plt.subplot(122)
    plt.hist(texp[texp>0], bins = bins)
    plt.title('Temperature distribution')
    plt.suptitle(name)
    plt.savefig('output/' + name + '_fitting_hist.png')

    plt.figure()
    plt.subplot(121)
    imshowid(np.log10(dgr))
    plt.subplot(122)
    plt.hist(np.log10(dgr[dgr>0]), bins = bins)
    plt.suptitle(name + ' dust to gas ratio (log scale)')
    plt.savefig('output/' + name + '_fitting_DGR.png')