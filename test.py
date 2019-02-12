import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from astropy.io import fits

m0 = fits.getdata('output/IC342.trial16.1.weak.m0.fits.gz')[0, 0]
m1 = fits.getdata('output/IC342.trial16.1.weak.m1.fits.gz')[0, 0]

cmap = cm.bwr_r
norm = mpl.colors.Normalize(vmin=m1.min(), vmax=m1.max())
m = cm.ScalarMappable(norm=norm, cmap=cmap)

mm = m.to_rgba(m1)[:, :, :3]

temp = np.sum(mm, axis=2)
for i in range(3):
    mm[:, :, i] /= temp

m0[m0 <= 0] = np.min(m0[m0 > 0])
m0 = np.log10(m0)
m0 -= m0.min()
m0 /= m0.max()

for i in range(3):
    mm[:, :, i] *= m0
mm /= mm.max()

fig1, ax1 = plt.subplots(figsize=(12, 10))
fig2, ax2 = plt.subplots()
mpb = ax2.imshow(m1, cmap='bwr_r')
ax1.imshow(mm, origin='lower')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
plt.colorbar(mpb, ax=ax1)
fig1.show()
