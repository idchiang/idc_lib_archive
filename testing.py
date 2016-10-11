from __future__ import absolute_import, print_function, unicode_literals, \
                       division
range = xrange
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from astro_idchiang import Surveys, imshowid, read_dust_file
from h5py import File

################################

plt.figure()
titles = ['Surface density', 'Temperature']
for i in range(2):
    plt.subplot(1, 2, i + 1)
    for j in range(nwalkers):
        plt.plot(results[j, :, i], 'b')
    plt.title(titles[i])

r_index = 151
n = 50
alpha = 0.1
wlexp = np.linspace(70,520)
nuexp = (c / wlexp / u.um).to(u.Hz)
modelexp = _model(wlexp, sexp, texp, nuexp)

plt.figure()
list_ = np.random.randint(0, len(samples), n)
for i in xrange(n):
    model_i = _model(wlexp, samples[list_[i],0], samples[list_[i],1], 
                             nuexp)
    plt.plot(wlexp, model_i, alpha = alpha, c = 'g')
plt.plot(wlexp, modelexp, label='Expectation', c = 'b')
plt.errorbar(wl, sed_avg[r_index], yerr = bkgmap[60, 70], fmt='ro', \
                     label='Data')
plt.axis('tight')
plt.legend()
plt.title('NGC 3198 ['+str(yt)+','+str(xt)+']')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel('SED')
