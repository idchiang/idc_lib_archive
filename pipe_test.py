from h5py import File
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS


def plot_PACS_100_negs():
    name = 'NGC5457'
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        sed = np.array(grp['HERSCHEL_011111'])[:, :, 0]
    with File('output/Voronoi_data.h5', 'r') as hf:
        grp = hf[name + '_011111']
        binlist = np.array(grp['BINLIST'])
        binmap = np.array(grp['BINMAP'])
        aSED = np.array(grp['Herschel_SED'])[:, 0]
    negs_mask = np.zeros_like(sed).astype(bool)
    for i in range(len(binlist)):
        if aSED[i] < 0:
            negs_mask[binmap == binlist[i]] = True
    plt.hist(sed[negs_mask], bins=25, range=(-4, 4))
    #
    fn = 'data/PACS/NGC_5457_I_100um_k2011.fits.gz'
    data, hdr = fits.getdata(fn, 0, header=True)
    w = WCS(hdr, naxis=2)
    return w, sed
