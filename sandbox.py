from __future__ import absolute_import, division, print_function, \
                       unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.colors import LogNorm
from h5py import File
from astropy.constants import c
from idc_lib import gal_data
range = xrange


def test():
    plt.ioff()
    name = 'NGC5457'
    bins = 50
    #
    with File('output/dust_data.h5', 'r') as hf:
        grp = hf[name]
        logsigmas = np.array(grp.get('logsigmas'))
        binmap = np.array(grp.get('Binmap'))
        radiusmap = np.array(grp.get('Radius_map'))
        total_gas = np.array(grp.get('Total_gas'))
        D = float(np.array(grp['Galaxy_distance']))
    """
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        total_gas = np.array(grp['Total_gas'])
        dp_radius = np.array(grp['DP_RADIUS'])
    """
    if len(logsigmas.shape) == 2:
        logsigmas = logsigmas[0]
    sigmas = 10**logsigmas
    #
    pdfs = pd.read_csv('output/' + name + '_pdf.csv', index_col=0)
    pdfs.index = pdfs.index.astype(int)
    #
    R25 = gal_data([name]).field('R25_DEG')[0]
    R25 *= (np.pi / 180.) * (D * 1E3)
    radiusmap /= R25
    #
    r, s, w = [], [], []
    for i in pdfs.index:
        bin_ = binmap == i
        temp_g = total_gas[bin_][0]
        temp_r = radiusmap[bin_][0]
        mask2 = (pdfs.iloc[i] > pdfs.iloc[i].max() / 1000).values
        temp_s = sigmas[mask2]
        temp_p = pdfs.iloc[i][mask2]
        temp_p /= np.sum(temp_p)
        for j in range(len(temp_s)):
            r.append(temp_r)
            s.append(temp_s[j] / temp_g)
            w.append(temp_g * temp_p[j])
        print('Current bin: No.' + str(i) + ' of ' + str(len(pdfs.index)))
    r, s, w = np.array(r), np.array(s), np.array(w)
    nanmask = np.isnan(r) + np.isnan(s) + np.isnan(w)
    r, s, w = r[~nanmask], s[~nanmask], w[~nanmask]
    rbins = np.linspace(np.min(r), np.max(r), bins)
    sbins = np.logspace(np.min(np.log10(s)), np.max(np.log10(s)), bins)
    counts, _, _ = np.histogram2d(r, s, bins=(rbins, sbins), weights=w)
    counts2, _, _ = np.histogram2d(r, s, bins=(rbins, sbins))
    counts = counts.T
    counts2 = counts2.T
    for i in range(len(counts)):
        if np.sum(counts2[:, i]) > 0:
            counts[:, i] /= np.sum(counts[:, i])
            counts2[:, i] /= np.sum(counts2[:, i])
    sbins = (sbins[:-1] + sbins[1:]) / 2
    rbins = (rbins[:-1] + rbins[1:]) / 2
    dgr_median = []
    # dgr_avg = []
    zeromask = [True] * len(counts[:])
    #
    for i in range(len(counts[:])):
        if np.sum(counts[:, i]) > 0:
            mask = counts[:, i] > (np.max(counts[:, i]) / 1000)
            smax = np.max(counts[mask, i])
            smin = np.min(counts[mask, i])
            csp = np.cumsum(counts[:, i])[:-1]
            csp = np.append(0, csp / csp[-1])
            sss = np.interp([0.16, 0.5, 0.84], csp, sbins)
            fig, ax = plt.subplots(2, 1)
            ax[0].semilogx([sss[0]] * len(counts[:, i]), counts[:, i],
                           label='16')
            ax[0].semilogx([sss[1]] * len(counts[:, i]), counts[:, i],
                           label='50')
            ax[0].semilogx([sss[2]] * len(counts[:, i]), counts[:, i],
                           label='84')
            # dgr_avg.append(np.sum(counts[:, i] * sbins))
            # ax[0].semilogx([dgr_avg[-1]] * len(counts[:, i]),
            #                counts[:, i], label='Exp')
            ax[0].semilogx(sbins, counts[:, i], label='PDF')
            ax[0].set_xlim([smin, smax])
            ax[0].legend()
            dgr_median.append(sss[1])
            ax[1].semilogx(sbins, counts2[:, i], label='Unweighted PDF')
            ax[1].set_xlim([smin, smax])
            ax[1].legend()
            fig.savefig('output/170311/' + str(i) + '_' + str(rbins[i]) +
                        '.png')
            plt.close('all')
        else:
            zeromask[i] = False
    zeromask = np.array(zeromask)
    #
    cmap = 'Reds'
    c_median = 'c'
    #
    plt.figure()
    plt.pcolormesh(rbins, sbins, counts, norm=LogNorm(), cmap=cmap)
    plt.yscale('log')
    plt.colorbar()
    plt.plot(rbins[zeromask], dgr_median, c_median, label='Median')
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'DGR', size=16)
    plt.title('Gas mass weighted DGR PDF', size=20)
    plt.savefig('output/hist2d_plt_pcolormesh.png')
