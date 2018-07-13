import os
from astropy.io import fits
from astropy.convolution import convolve_fft, convolve
import matplotlib.pyplot as plt
from time import clock
import numpy as np

os.system('clear')  # on linux / os x

survey = 'PACS160'
name = 'NGC5457'
name1 = 'NGC'
name2 = '5457'

filename = 'data/PACS/' + name1 + '_' + name2 + '_I_160um_k2011.fits.gz'
data, hdr = fits.getdata(filename, 0, header=True)
PACS160 = data[0]

filename = 'data/Kernels/Kernel_LoRes_PACS_160_to_SPIRE_500.fits.gz'
kernel, khdr = fits.getdata(filename, 0, header=True)

tic = clock()
r_fft = convolve_fft(PACS160, kernel)
r_fft[np.isnan(PACS160)] = np.nan
toc = clock()
fft_time = toc - tic
print('FFT done')
tic = clock()
r = convolve(PACS160, kernel)
r[np.isnan(PACS160)] = np.nan
toc = clock()
print('convolve done')
print('FFT time:', fft_time)
print('COV time:', toc - tic)
print('FFT sum:', np.nansum(r_fft))
print('COV sum:', np.nansum(r))

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].imshow(PACS160, origin='lower')
ax[0, 0].set_title('Original')
ax[0, 1].imshow(r_fft, origin='lower')
ax[0, 1].set_title('convolve_fft')
ax[1, 0].imshow(r, origin='lower')
ax[1, 0].set_title('convolve')
im = ax[1, 1].imshow((r - r_fft) / r_fft, origin='lower')
ax[1, 1].set_title('Relative residual')
plt.colorbar(im, ax=ax[1, 1])
fig.tight_layout()
fig.savefig('output/Ref017.png')
