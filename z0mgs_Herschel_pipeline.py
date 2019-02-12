# from generating import generator
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
import astropy.units as u
import idc_lib.idc_io as idcio
from idc_lib.idc_fitting import fit_dust_density as fdd
from idc_lib.idc_functions import save_fits_gz, error_msg
from idc_lib.gal_data import gal_data
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.wcs import WCS


preprocessing = 1
fitting = 1
bkgcov_generating = 1

all_objects = 0
selected_objects = ['ic10']

os.system('clear')  # on linux / os x
project_name = 'z0mgs_Herschel'
dpath = 'data/z0mgs_Herschel/'
dpath2 = 'data/z0mgs_Herschel_RGD/'
kpath = 'data/Kernels/'
ppath = 'Projects/z0mgs_Herschel/'

aobjects = []
objects = aobjects if all_objects else selected_objects
# Name: [Capital name, Native Gaussian, Beam area]
# http://herschel.esac.esa.int/Docs/SPIRE/spire_handbook.pdf Table 5.2
band_char = {'pacs100': ['PACS_100', 'Gauss_15', 15.0, np.nan],
             'pacs160': ['PACS_160', 'Gauss_15', 15.0, np.nan],
             'spire250': ['SPIRE_250', 'Gauss_30', 30.0, 469.35],
             'spire350': ['SPIRE_350', 'Gauss_30', 30.0, 831.27],
             'spire500': ['SPIRE_500', 'Gauss_41', 41.0, 1804.31]}
bands = list(band_char.keys())


def mp_regrid(o, bands):
    nbands = len(bands)
    tfn = dpath + o + '_scanamorphos_v25_' + bands[nbands - 1] + '_0.fits.gz'
    tdata, thdr = fits.getdata(tfn, header=True)
    tdata = tdata[0]
    #
    # -1. Comment data
    #
    # -1-0. Grab information from gal_data
    try:
        gdata = gal_data(o)
        ra = Angle(gdata.field('RA_DEG')[0] * u.deg)
        dec = Angle(gdata.field('DEC_DEG')[0] * u.deg)
        posang = Angle(gdata.field('POSANG_DEG')[0] * u.deg)
        if np.isnan(posang):
            error_msg('output/z0mgs_Herschel_fitting_error.txt',
                      o + ': POSANG is NaN.')
            posang = Angle(0.0 * u.deg)
        incl = Angle(gdata.field('INCL_DEG')[0] * u.deg)
        r25 = Angle(gdata.field('R25_DEG')[0] * u.deg).arcsec
    except KeyError:
        if o == 'ic10':  # LEDA
            ra = Angle('00h20m23.11s')
            dec = Angle('+59d17m34.9s')
            posang = Angle(129.0 * u.deg)
            incl = Angle(31.1 * u.deg)
            r25 = Angle((10**1.83 * 0.1 / 60 / 2) * u.deg).arcsec
        else:
            print(o, 'not implemented yet')
            sys.exit()
    #
    for bi in range(nbands):
        band = bands[bi]
        job_name = o + '.' + band
        #
        # 0. Load data
        #
        print('\n##', job_name + ': Load image')
        fn = dpath + o + '_scanamorphos_v25_' + band + \
            '_0.fits.gz'
        print('file name:', fn)
        data, hdr = fits.getdata(fn, header=True)
        data = data[0]
        ps = hdr['PFOV']
        # 0-1. Correct PACS unit and SPIRE dimension to MJy/sr
        if band[:4] == 'pacs':
            if hdr['BUNIT'] == 'Jy/pixel':
                print('\n##', job_name + ': PACS unit conversion')
                data *= (np.pi / 36 / 18)**(-2) / ps**2
            else:
                print('Unit conversion not implemented yet')
                print('Unit:', hdr['BUNIT'])
                sys.exit()
        elif band[:5] == 'spire':
            if hdr['BUNIT'] == 'Jy/beam':
                data *= (np.pi / 36 / 18)**(-2) / band_char[band][3]
            elif hdr['ZUNITS'] == 'MJy/sr':  # This will go wrong.
                pass
            else:
                print('Unit conversion not implemented yet')
                print('Unit:', hdr['BUNIT'])
                sys.exit()
        #
        # 2. Masking
        #
        # 2-0. Frame of the current image
        shape = data.shape
        w = WCS(hdr, naxis=2)
        # 2-1. Generate radius map (in arcsec or deg)
        radius_map = idcio.radius_arcsec(shape, w, ra, dec, posang, incl)
        # 2-2. Create mask with radius map and r25.
        diskmask = radius_map < (3.0 * r25)
        #
        # 3. Background subtraction
        #
        # 3-1. Subtract bkg level: median(pixels in bkg with AD <= 3 MAD)
        # 3-2. Iterate 3-1. until difference < 1%
        bkgmask = ~diskmask * np.isfinite(data)
        assert np.sum(bkgmask) > 200
        # Will need to adjust all iterations if one of the bands meet assertion
        # error.
        bkgs = data[bkgmask]
        M = np.median(bkgs)
        prev_bkg = 1000.0
        for i in range(5):
            AD = np.abs(bkgs - M)
            MAD = np.median(AD)
            M = np.median(bkgs[AD <= 3 * MAD])
            if np.abs(prev_bkg - M) / prev_bkg < 0.01:
                break
            else:
                prev_bkg = M
        data -= M
        print('\n##', job_name + ': Background level =', round(M, 3))
        # 3-3. Tilted plane from my old code
        print('\n##', job_name + ': Tilted plane background removal')
        data = idcio.bkg_removal(data, diskmask, job_name=job_name)
        #
        # 4. Convolution
        #
        # 4-1. Convolve image and mask
        print('\n##', job_name + ': Load spire500 kernel')
        kernel, khdr = idcio.load_kernel(kpath, band_char[band][0],
                                         'SPIRE_500')
        kernel /= np.nansum(kernel)
        print('\n##', job_name + ': Convolve to circular SPIRE500')
        data = idcio.convolve_map(data, hdr, kernel, khdr,
                                  job_name=job_name)
        if bi != (nbands - 1):
            # 4-2. Regrid image
            print('\n##', job_name + ': Regrid')
            data = idcio.regrid(data, hdr, tdata, thdr, exact=False,
                                job_name=job_name)
        else:
            fn = dpath2 + o + '_diskmask.fits'
            save_fits_gz(fn, diskmask.astype(int), thdr)
            fn = dpath2 + o + '_radius.arcsec.fits'
            save_fits_gz(fn, radius_map, thdr)
        bitpix = abs(int(hdr['BITPIX']))
        if bitpix == 32:
            data = data.astype(np.float32)
        elif bitpix == 16:
            data = data.astype(np.float16)
        fn = dpath2 + o + '_' + band + '_bgsub.fits'
        save_fits_gz(fn, data, thdr)


def mp_fit(o, bands):
    observe_fns = []
    for band in bands:
        observe_fns.append(dpath2 + o + '_' + band + '_bgsub.fits.gz')
    mask_fn = dpath2 + o + '_diskmask.fits.gz'
    #
    fdd(o, method_abbr='FB', del_model=False,
        nop=10, beta_f=2.0, Voronoi=False, save_pdfs=False,
        project_name='z0mgs_Herschel', observe_fns=observe_fns,
        mask_fn=mask_fn, subdir='', notes='',
        bands=bands, rand_cube=True, better_bkgcov=None,
        import_beta=False, galactic_integrated=False)


if __name__ == "__main__":
    #
    if preprocessing:
        for o in objects:
            mp_regrid(o, bands)
    if fitting:
        for o in objects:
            mp_fit(o, bands)
