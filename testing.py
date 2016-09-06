# log scale graph

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.figure()
plt.suptitle(r'NGC 3198 dust surface mass density (M$_\odot$/pc$^2$)')
plt.subplot(231)
plt.imshow(s_opt[40:150,40:145], norm=LogNorm(vmin=1.0E-4, vmax=1.0E2), origin='lower')
plt.colorbar()
plt.title('Fitted result')

s_nnan_array = s_opt[~np.isnan(s_opt)]
s_nnan_array = s_nnan_array[s_nnan_array > 0]
plt.subplot(232)
plt.hist(np.log10(s_nnan_array), bins=50)
plt.xlabel(r'Log$_{10}$ surface mass density')
plt.ylabel('Count')
plt.title(r'Log$_{10}$ surface mass density distribution')

mask_err = (s_opt > (3.0*s_err))
plt.subplot(234)
plt.imshow(mask_err[40:150,40:145], origin='lower')
plt.title('POPT > 3*PERR')
mask_err = (s_opt > (2.0*s_err))
plt.subplot(235)
plt.imshow(mask_err[40:150,40:145], origin='lower')
plt.title('POPT > 2*PERR')
mask_err = (s_opt > (s_err))
plt.subplot(236)
plt.imshow(mask_err[40:150,40:145], origin='lower')
plt.title('POPT > PERR')

#####Temperature######

plt.figure()
plt.suptitle('NGC 3198 dust temperature (K)')
plt.subplot(231)
plt.imshow(t_opt[40:150,40:145], vmin=0.0, vmax=3.0E2, origin='lower')
plt.colorbar()
plt.title('Fitted result')

t_nnan_array = t_opt[~np.isnan(t_opt)]
t_nnan_array = t_nnan_array[t_nnan_array > 0]
plt.subplot(232)
plt.hist(np.log10(t_nnan_array), bins=50)
plt.xlabel(r'Log$_{10}$ Temperature')
plt.ylabel('Count')
plt.title(r'Log$_{10}$ Temperature distribution')

mask_err = (t_opt > (3.0*t_err))
plt.subplot(234)
plt.imshow(mask_err[40:150,40:145], origin='lower')
plt.title('POPT > 3*PERR')

temp_copy = t_opt.copy()
temp_copy[~mask_err] = np.nan
plt.subplot(235)
plt.imshow(temp_copy[40:150,40:145], vmin=0.0, vmax=3.0E2, origin='lower')
plt.colorbar()
plt.title('SNR filtted fitted result')

temp_copy = t_opt.copy()
temp_copy[~mask_err] = np.nan
t_nnanerr_array = temp_copy[~np.isnan(temp_copy)]
t_nnanerr_array = t_nnanerr_array[t_nnanerr_array > 0]
plt.subplot(236)
plt.hist(np.log10(t_nnanerr_array), bins=50)
plt.xlabel(r'Log$_{10}$ Temperature')
plt.ylabel('Count')
plt.title(r'Log$_{10}$ SNR filtted distribution')