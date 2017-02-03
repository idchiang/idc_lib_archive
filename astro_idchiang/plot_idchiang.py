from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def imshowid(image):
    plt.imshow(image, origin='lower')
    plt.colorbar()

def imshow_log(image, vmin=None, vmax=None):
    if vmin == None:
        vmin = min(image[image>0].flatten())
    if vmax == None:
        vmax = max(image.flatten())
    plt.imshow(image, origin ='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar()
    
""" 
Might be useful in the future!!

fig = plt.figure(figsize=(8, 8))
gs = plt.GridSpec(3, 3)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :2])
ax3 = fig.add_subplot(gs[1:, 2])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])
"""