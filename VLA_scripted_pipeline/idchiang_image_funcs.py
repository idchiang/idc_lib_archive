import os
import sys
import gzip
import shutil
import imp
import numpy as np
# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
from shutil import rmtree
from tclean import tclean
from concat import concat
# from mstransform import mstransform
from exportfits import exportfits
from immoments import immoments
from imstat import imstat
from astropy.io import fits
pipepath = '/home/idchiang/VLA/VLA_scripted_pipeline/'
sys.path.append(pipepath + 'au/')
import analysisUtils as au
gal_data = \
    imp.load_source('gal_data', '/home/idchiang/idc_lib/idc_lib/gal_data.py')

# Constants for HI 21cm line
HIfreq = 1420.405752  # MHz
# w = 2.5
# vel_width = str(w) + 'km/s'
# startvel = str(w * nchan / (-2)) + 'km/s'


def target_info():
    # Read and return target info
    target = os.getcwd().split('/')[-1]
    subdirs = ['01', '02', '03']
    mss = []
    for subdir in subdirs:
        if os.path.isdir(subdir):
            fns = os.listdir(subdir)
            for fn in fns:
                temp = fn.split('.')
                if len(temp) > 4:
                    if temp[-1] == 'split':
                        mss.append(subdir + '/' + fn)
    # Calculate restfreq and phasecenter
    gdata = gal_data.gal_data(target)
    ra = str(round(gdata.field('RA_DEG')[0], 1)) + 'deg '
    dec = str(round(gdata.field('DEC_DEG')[0], 1)) + 'deg'
    glx_ctr = 'J2000 ' + ra + dec
    # target: name of target galaxy
    # mss: the calibrated ms files
    final_vis = target + '.ms'
    if os.path.isdir(final_vis):
        print("...Removing previous final vis file")
        rmtree(final_vis)
    print("...Concating " + str(len(mss)) + " ms files for " +
          target)
    concat(vis=mss, concatvis=final_vis)
    return target, mss, final_vis, glx_ctr


def image_dimensions(vis, oversamplerate=5):
    # Dimensions from analysisUtils
    au_cellsize, au_imsize, au_oversample = \
        au.pickCellSize(vis=vis, imsize=True, npix=oversamplerate)
    # Determine cellsize
    if au_cellsize < 1.0:
        cellsize_arcsec = 1.0
    else:
        cellsize_arcsec = int(au_cellsize * 2) / 2.0
    cellsize_str = str(cellsize_arcsec) + 'arcsec'
    # Determin image size
    valid_sizes = []
    for ii in range(10):
        for kk in range(3):
            for jj in range(3):
                valid_sizes.append(2**(ii+1)*5**(jj)*3**(kk))
    valid_sizes.sort()
    valid_sizes = np.array(valid_sizes)
    need_cells_x = au_imsize[0] * 1.2 * au_cellsize / cellsize_arcsec
    need_cells_y = au_imsize[1] * 1.2 * au_cellsize / cellsize_arcsec
    cells_x = np.min(valid_sizes[valid_sizes > need_cells_x])
    cells_y = np.min(valid_sizes[valid_sizes > need_cells_y])
    imsize = [cells_x, cells_y]
    return cellsize_str, cellsize_arcsec, imsize


def imaging(vis, trial_name, interactive, imsize, cellsize, glx_ctr, restfreq,
            specmode, outframe, veltype, restoringbeam, weighting, robust,
            scales, smallscalebias, dogrowprune,
            growiterations, noisethreshold, minbeamfrac, sidelobethreshold,
            gridder, pbmask, pblimit, threshold, niter_in, nsigma, cyclefactor,
            minpsffraction, gain, w, nchan):
    #
    titles = ['.dirty', '', '.2.strong']
    #
    for i in range(2):
        # niter
        niter = 0 if i == 0 else niter_in
        # masking method
        usemask = 'auto-multithresh' if i == 2 else 'pb'
        # deconvolver
        deconvolver = 'multiscale'
        #
        tclean(vis=vis,
               imagename=trial_name,
               interactive=interactive,
               intent='*TARGET*',
               #
               datacolumn='data',
               nchan=nchan,
               start=str(w * nchan / (-2)) + 'km/s',
               width=str(w) + 'km/s',
               # Image dimension
               imsize=imsize,
               cell=cellsize,
               phasecenter=glx_ctr,
               restfreq=restfreq,
               specmode=specmode,
               outframe=outframe,
               veltype=veltype,
               # Restore to common beam?
               restoringbeam=restoringbeam,
               # Weighting
               weighting=weighting,
               robust=robust,
               # Methods
               deconvolver=deconvolver,
               scales=scales,
               gain=gain,
               smallscalebias=smallscalebias,
               usemask=usemask,
               dogrowprune=dogrowprune,
               growiterations=growiterations,
               noisethreshold=noisethreshold,
               minbeamfrac=minbeamfrac,
               sidelobethreshold=sidelobethreshold,
               gridder=gridder,
               pbmask=pbmask,
               pblimit=pbmask,
               # Stopping criteria
               threshold=threshold,
               niter=niter,
               nsigma=nsigma,
               cyclefactor=cyclefactor,
               minpsffraction=minpsffraction)
        #
        cubename = trial_name + '.image'
        tempcube = trial_name + titles[i] + '.cube.fits'
        residualname = trial_name + '.residual'
        tempresidual = trial_name + titles[i] + '.residual.fits'
        m0name = cubename + '.integrated'
        tempm0 = trial_name + titles[i] + '.m0.fits'
        m1name = cubename + '.weighted_coord'
        tempm1 = trial_name + titles[i] + '.m1.fits'
        m2name = cubename + '.weighted_dispersion_coord'
        tempm2 = trial_name + titles[i] + '.m2.fits'
        for mxname in [m0name, m1name, m2name]:
            if os.path.isdir(mxname):
                rmtree(mxname)
        #
        print("...Calculating moment maps")
        rms = imstat(cubename)['rms'][0]
        excludepix = -1 if i == 0 else [-1000, (2 * rms)]
        immoments(imagename=cubename, moments=[0, 1, 2], axis='spectral',
                  excludepix=excludepix)
        print("#\n")
        print("...Exporting fits files of " + titles[i])
        for (mxname, fitsname) in zip([cubename, residualname, m0name, m1name,
                                       m2name],
                                      [tempcube, tempresidual, tempm0, tempm1,
                                       tempm2]):
            if (mxname in [residualname, m1name, m2name]) and i == 0:
                continue
            exportfits(imagename=mxname, fitsimage=fitsname)
            if mxname == cubename:
                data = fits.getdata(tempcube)
                nanmask = np.any(np.isnan(data), axis=(0, 1))
                del data
            if mxname in [m0name, m1name, m2name]:
                print("...Setting " + m0name + " NaNs to zeros")
                data, hdr = fits.getdata(fitsname, header=True)
                data[np.isnan(data)] = 0.0
                data[0, 0][nanmask] = np.nan
                fits.writeto(fitsname, data, hdr, overwrite=True)
            with open(fitsname, 'rb') as f_in:
                with gzip.open(fitsname + '.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(fitsname)
            if mxname in [m0name, m1name, m2name]:
                rmtree(mxname)
        """
        print("...Plotting weighted m1 image")
        plt.ioff()
        m0 = fits.getdata(tempm0 + '.gz')[0, 0]
        m1 = fits.getdata(tempm1 + '.gz')[0, 0]
        #
        cmap = cm.bwr_r
        norm = mpl.colors.Normalize(vmin=np.nanmin(m1),
                                    vmax=np.nanmax(m1))
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        with np.errstate(invalid='ignore'):
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
        #
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        fig2, ax2 = plt.subplots()
        mpb = ax2.imshow(m1, cmap='bwr_r', origin='lower')
        ax1.imshow(mm, origin='lower')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.colorbar(mpb, ax=ax1)
        fig1.savefig(trial_name + titles[i] + '.m1.m0-weighted.png')
        plt.close('all')
        """
