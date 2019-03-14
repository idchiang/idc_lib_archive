import os
import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from reproject import reproject_interp
solar_oxygen_bundance = 8.69
# IC342 R25 DEG = 0.16627185
# IC342 R25 arcsec = 598.57866
# IC342 calibration: KK04
r25_kpc = {'M101': 24.283176287707708, 'IC342': 6.645559814090436}
dist_mpc = {'M101': 6.960000038146973, 'IC342': 2.28999996}
# col2sur = 8.009006438434431e-21


def metallicity_from_oxygen_abd(oxygen_abundance):
    # oxygen_abundance in 12 + log(o/H)
    return 10**(oxygen_abundance - 12.0) * 16.0 / 1.008 / 0.51 / 1.36


def D14_DGR(oxygen_abundance):
    return 10**(oxygen_abundance - solar_oxygen_bundance) / 150


def DGR():
    plt.close('all')
    dpath = 'Projects/IC342_and_M101/'
    M101_dust = fits.getdata(dpath + 'M101_sigmad.fits.gz')
    IC342_dust = fits.getdata(dpath + 'IC342_sigmad.fits.gz')
    M101_edust = fits.getdata(dpath + 'M101_sigmad_err_dex.fits.gz')
    IC342_edust = fits.getdata(dpath + 'IC342_sigmad_err_dex.fits.gz')
    M101_gas = fits.getdata(dpath + 'M101_totalgas.fits.gz')
    IC342_gas = fits.getdata(dpath + 'IC342_totalgas.fits.gz')
    M101_metal = fits.getdata(dpath + 'M101_12+logOH.fits.gz')
    IC342_metal = fits.getdata(dpath + 'IC342_12+logOH.fits.gz')
    M101_fh2 = fits.getdata(dpath + 'M101_fh2.fits.gz')
    IC342_fh2 = fits.getdata(dpath + 'IC342_fh2.fits.gz')
    M101_z = metallicity_from_oxygen_abd(M101_metal)
    IC342_z = metallicity_from_oxygen_abd(IC342_metal)
    M101_DGR = M101_dust / M101_gas
    IC342_DGR = IC342_dust / IC342_gas
    #
    plt.ion()
    fig, ax = plt.subplots(2, 2)
    p, q = 0, 0
    ax[p, q].scatter(M101_metal, M101_DGR, s=1, label='M101')
    ax[p, q].scatter(IC342_metal, IC342_DGR, s=1, label='IC342')
    ax[p, q].set_yscale('log')
    ax[p, q].legend()
    ax[p, q].set_xlabel('12+log(O/H)')
    ax[p, q].set_ylabel('DGR')
    p, q = 0, 1
    ax[p, q].scatter(M101_fh2, M101_DGR, s=1, label='M101')
    ax[p, q].scatter(IC342_fh2, IC342_DGR, s=1, label='IC342')
    ax[p, q].set_xscale('log')
    ax[p, q].set_yscale('log')
    ax[p, q].legend()
    ax[p, q].set_xlabel(r'$\rm f_{H_2}$')
    ax[p, q].set_ylabel('DGR')
    p, q = 1, 0
    ax[p, q].scatter(M101_metal, M101_DGR / D14_DGR(M101_metal),
                     s=1, label='M101')
    ax[p, q].scatter(IC342_metal, IC342_DGR / D14_DGR(IC342_metal),
                     s=1, label='IC342')
    ax[p, q].set_yscale('log')
    ax[p, q].legend()
    ax[p, q].set_xlabel('12+log(O/H)')
    ax[p, q].set_ylabel('DTM / DTM(D14)')
    p, q = 1, 1
    ax[p, q].scatter(M101_fh2, M101_DGR / D14_DGR(M101_metal),
                     s=1, label='M101')
    ax[p, q].scatter(IC342_fh2, IC342_DGR / D14_DGR(IC342_metal),
                     s=1, label='IC342')
    ax[p, q].set_xscale('log')
    ax[p, q].set_yscale('log')
    ax[p, q].legend()
    ax[p, q].set_xlabel(r'$\rm f_{H_2}$')
    ax[p, q].set_ylabel('DTM / DTM(D14)')


def M101_sigmad_td():
    plt.close('all')
    dpath = 'Projects/IC342_and_M101/'
    M101_dust, hdr = fits.getdata(dpath + 'M101_sigmad.fits.gz', header=True)
    M101_td = fits.getdata(dpath + 'M101_Td.fits.gz', header=False)
    w = WCS(hdr, naxis=2)
    fig = plt.figure()
    ax = fig.add_subplot('111', projection=w)
    cax = ax.imshow(M101_dust, origin='lower', cmap='inferno')
    plt.colorbar(cax, ax=ax)
    ax.set_title(r'M101 $\log[\Sigma_d\, /\, (M_\odot\,pc^{-2})]$',
                 fontsize=14)
    # ax.set_xlabel('R.A. (2000)')
    ax.set_xlabel('R.A. (2000)')
    ax.set_ylabel('Dec. (2000)')
    # fig.tight_layout()
    fig.savefig('output/M101_sigmad.png')
    #
    fig = plt.figure()
    ax = fig.add_subplot('111', projection=w)
    cax = ax.imshow(M101_td, origin='lower', cmap='inferno')
    plt.colorbar(cax, ax=ax)
    ax.set_title(r'M101 $T_d (K)$',
                 fontsize=14)
    ax.set_xlabel('R.A. (2000)')
    ax.set_ylabel('Dec. (2000)')
    # fig.tight_layout()
    fig.savefig('output/M101_Td.png')


def M101_IC342_data():
    for i in range(2):
        if i == 0:
            HI, HIhdr = \
                fits.getdata('data/THINGS/NGC_5457_NA_MOM0_THINGS.FITS',
                             header=True)
            CO, COhdr = \
                fits.getdata('data/HERACLES/ngc5457_heracles_mom0.fits.gz',
                             header=True)
            P100, Phdr = \
                fits.getdata('data/KINGFISH_DR3/NGC5457_scanamorphos_v16.9' +
                             '_pacs100_0.fits', header=True)
            fn = 'output/M101_data.png'
            position = (1024, 1024)
            size = (1536, 1536)
        else:
            HI, HIhdr = \
                fits.getdata('data/EveryTHINGS/IC342.m0.fits.gz',
                             header=True)
            CO, COhdr = \
                fits.getdata('data/PHANGS/ic342_12co10-20kms-d-30m.mom0.fits',
                             header=True)
            P100, Phdr = \
                fits.getdata('data/KINGFISH_DR3/IC0342_scanamorphos_v16.9' +
                             '_pacs100_0.fits', header=True)
            fn = 'output/IC342_data.png'
            position = (900, 900)
            size = (900, 900)
            position2 = (450, 450)
            size2 = (300, 300)
        HI = HI[0, 0]
        P100 = P100[0]
        HIw = WCS(HIhdr, naxis=2)
        COw = WCS(COhdr, naxis=2)
        Pw = WCS(Phdr, naxis=2)
        #
        temp = Cutout2D(HI, position, size, HIw)
        HI, HIw = temp.data, temp.wcs
        CO, _ = reproject_interp((CO, COw), HIw, HI.shape)
        P100, _ = reproject_interp((P100, Pw), HIw, HI.shape)
        #
        COmask = np.isfinite(CO).astype(int)
        #
        fig = plt.figure(figsize=(16, 6))
        titles = [['FIR: KINGFISH', 'CO: HERACLES', 'HI: THINGS'],
                  ['FIR: KINGFISH', 'CO: PHANGS', 'HI: z0MGS-EveryTHINGS']]
        citations = [['Kennicutt et al. (2011)', 'Leroy et al. (2009)',
                      'Walter et al. (2008)'],
                     ['Kennicutt et al. (2011)', 'Schruba et al. in prep.',
                      'Chiang et al. in prep.']]
        for j in range(3):
            if j == 0:
                image = P100
                std = np.nanstd(image) / 2.0
                with np.errstate(invalid='ignore', divide='ignore'):
                    image[image < std] = std
                    image = np.log10(image)
                wcs = HIw
            elif j == 1:
                image = CO
                if i == 0:
                    wcs = HIw
                    COmask2 = COmask
                elif i == 1:
                    temp = Cutout2D(CO, position2, size2, HIw)
                    image, wcs = temp.data, temp.wcs
                    temp = Cutout2D(COmask, position2, size2, HIw)
                    COmask2 = temp.data
                with np.errstate(invalid='ignore', divide='ignore'):
                    std = np.abs(np.nanmin(image))
                    image[image < std] = std
                    image = np.log10(image)
            elif j == 2:
                image = HI
                wcs = HIw
            ax = fig.add_subplot('13' + str(j + 1), projection=wcs)
            ax.imshow(image, origin='lower', cmap='inferno')
            if j == 1:
                ax.contour(COmask2)
            else:
                ax.contour(COmask)
            ax.set_title(titles[i][j], fontsize=20)
            ax.text(x=0.95, y=0.05, s=citations[i][j], ha='right', va='bottom',
                    color='brown', transform=ax.transAxes)
            ax.set_xlabel('R.A. (2000)')
            if j == 0:
                ax.set_ylabel('Dec. (2000)')
        fig.savefig(fn, bbox_inches='tight')
    plt.close('all')


def VLA_and_Hershcel():  # tecolote only
    vpath = '/data/scratch/idchiang/'
    ppath = '/data/scratch/jchastenet/z0mgs_PACS_16oct18/Formatted/'
    # spath = '/data/scratch/jchastenet/z0mgs_SPIRE_15nov18/Scanamorphosed/'
    plt.ioff()

    names = {  # 'IC342': [[[515, 1286], [494, 1303]]],
             'NGC1961': [[[630, 811], [630, 811]]],
             # 'NGC2787': [w], (low SNR + RFI) (Can change velocity resolution)
             # 'NGC3227': [], (Weird shape + RFI)
             'NGC3898': [[[264, 377], [264, 377]]],
             'NGC3953': [[[483, 798], [483, 798]]],
             # 'NGC4038': [w], (Good; Expand spectral range; can add?)
             # 'NGC4374': [], (Bad cali; no PACS; clear image)
             'NGC4038': [[[503, 778], [503, 778]]],
             'NGC4496A': [[[524, 628], [524, 628]]],
             # 'NGC4501': [n], (Super strong RFI) (Can drop one block)
             'NGC4535': [[[472, 809], [472, 809]]]
             # 'NGC7479': [n] (Spectral range; RFI; ok image; can add?)
             }

    samples = list(names.keys())
    samples.sort()
    n = len(samples)

    # Generate plot structure
    fig, ax = plt.subplots(3, 4, figsize=(20, 14))
    fig.subplots_adjust(wspace=0, hspace=0.1)

    for i in range(n):
        # 1) p, q: location of VLA subplot; s: sample name
        p, q = i % 3, (i // 3) * 2
        s = samples[i]
        print(s)
        # 2) Hide ticks and labels
        ax[p, q].axis('off')
        ax[p, q+1].axis('off')
        # ax[p, q].patch.set_facecolor('black')
        # ax[p, q+1].patch.set_facecolor('black')
        # 3) VLA part
        # 3-1) Load VLA image
        VLA, Vhdr = fits.getdata(vpath + s + '/' + s + '.m0.fits.gz',
                                 header=True)
        VLA = VLA[0, 0]
        Vhdr.remove('NAXIS3')
        Vhdr.remove('NAXIS4')
        Vhdr['NAXIS'] = 2
        Vwcs = WCS(Vhdr, naxis=2)
        # 3-2) Cut VLA image
        if s == '':
            axissum = [0] * 2
            b = np.zeros([2, 2], dtype=int)
            for i in range(2):
                axissum[i] = np.nansum(VLA, axis=i, dtype=bool)
                for j in range(len(axissum[i])):
                    if axissum[i][j]:
                        b[i-1, 0] = j
                        break
                b[i-1, 1] = j + np.sum(axissum[i], dtype=int)
            print(s, b)
        else:
            b = np.array(names[s][0])
        # 3-3) Plot VLA image
        ax[p, q].imshow(VLA[b[0, 0]:b[0, 1], b[1, 0]:b[1, 1]], origin='lower',
                        cmap='hot')
        # [b[0, 0]:b[0, 1], b[1, 0]:b[1, 1]]
        ax[p, q].set_title('VLA HI', x=0.5, y=0.10, ha='center', va='bottom',
                           color='white', fontsize=16)
        ax[p, q].text(x=0.05, y=0.95, s=s, ha='left', va='top',
                      color='white', fontsize=20, transform=ax[p, q].transAxes)
        # 4) Hershcel part
        if s not in ['IC342']:
            # 4-1) Load Herschel images
            tempdir = ppath + s.lower() + '/'
            fns = os.listdir(tempdir)
            for tfn in fns:
                if tfn[-3:] == 'red':
                    fn = tempdir + tfn + '/' + s.lower() + \
                        '_scanamorphos_v25_pacs160_0.fits'
                    break
            pacs160, phdr = fits.getdata(fn, header=True)
            pacs160 = pacs160[0]
            phdr.remove('NAXIS3')
            phdr['NAXIS'] = 2
            pwcs = WCS(phdr, naxis=2)
            #
            """
            if s not in ['NGC4535']:
                fn = spath + s.lower() + '/' + s.lower() + \
                    '_scanamorphos_v25_spire350_0.fits'
                spire350, shdr = fits.getdata(fn, header=True)
                spire350 = spire350[0]
                shdr.remove('NAXIS3')
                shdr['NAXIS'] = 2
                swcs = WCS(shdr, naxis=2)
            """
            # 4-2) Reproject Herschel images
            pacs160, _ = reproject_interp((pacs160, pwcs), Vwcs, VLA.shape)
            # spire350, _ = reproject_interp((spire350, swcs), Vwcs, VLA.shape)
            # 4-3) Cut Herschel images
            pacs160 = pacs160[b[0, 0]:b[0, 1], b[1, 0]:b[1, 1]]
            sigma = np.nanstd(pacs160)
            pacs160[pacs160 <= 0.0] = 1.0 * sigma
            # spire350 = spire350[b[0, 0]:b[0, 1], b[1, 0]:b[1, 1]]
            # 4-4) Plot Herschel images
            ax[p, q+1].imshow(pacs160, origin='lower', cmap='Blues_r',
                              vmin=1.0 * sigma, norm=LogNorm())
            """
            ax[p, q+1].imshow(spire350, origin='lower', cmap='Reds_r',
                              alpha=0.5)
            """
            ax[p, q+1].set_title('PACS 160', x=0.5, y=0.10, ha='center',
                                 va='bottom', color='white', fontsize=16)
    # save and tight_layout
    fig.savefig('output/VLA+Herschel_thumbnails.png', bbox_inches='tight')


if platform.system() == 'Windows':
    M101_sigmad_td()
else:
    VLA_and_Hershcel()
