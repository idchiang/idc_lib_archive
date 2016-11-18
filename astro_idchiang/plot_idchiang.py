from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import matplotlib
matplotlib.use('Agg')
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