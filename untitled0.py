from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import numpy as np
from time import clock
import matplotlib.pyplot as plt
from h5py import File

hf = File('output/dust_data.h5', 'a')
hfn = File('output/dust_datan.h5', 'a')

all_objects = ['IC2574', 'NGC0628', 'NGC0925', 'NGC2841', 'NGC2976', 'NGC3077', 
               'NGC3184', 'NGC3198', 'NGC3351', 'NGC3521', 'NGC3627', 
               'NGC4736', 'NGC5055', 'NGC5457', 'NGC7331']

a = ['Binmap', 'Dust_temperature', 'Dust_temperature_err', 'Galaxy_center', 
     'Herschel_SED', 'Herschel_binned_bkg', 'PS', 'Radius_map', 'Total_gas']
n = ['Galaxy_distance', 'INCL', 'PA']
d = ['Dust_surface_density', 'Dust_surface_density_err']

for sample in all_objects:
    grpn = hfn.create_group(sample)
    grp = hf[sample]
    for temp in a:
          grpn.create_dataset(temp, data=np.array(grp[temp]))
    for temp in n:
          grpn.create_dataset(temp, data=float(np.array(grp[temp])))
    for temp in d:
          grpn.create_dataset(temp, data=np.pi * np.array(grp[temp]))    