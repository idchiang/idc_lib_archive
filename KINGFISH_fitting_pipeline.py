# from generating import generator
import os
# import sys
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
selected_objects = ['IC0342']


os.system('clear')  # on linux / os x
project_name = 'KINGFISH18'
dpath = 'data/KINGFISH_DR3/'
dpath2 = 'data/KINGFISH_DR3_RGD/'
kpath = 'data/Kernels/'
ppath = 'Projects/KINGFISH18/'

# NGC0584
aobjects = ['DDO053', 'DDO154', 'DDO165', 'HoI', 'HoII', 'IC0342', 'IC2574',
            'M81dwB', 'NGC0337', 'NGC0628', 'NGC0855', 'NGC0925',
            'NGC1097', 'NGC1266', 'NGC1291', 'NGC1316', 'NGC1377', 'NGC1404',
            'NGC1482', 'NGC1512', 'NGC2146', 'NGC2798', 'NGC2841', 'NGC2915',
            'NGC2976', 'NGC3049', 'NGC3077', 'NGC3184', 'NGC3190', 'NGC3198',
            'NGC3265', 'NGC3351', 'NGC3521', 'NGC3621', 'NGC3627', 'NGC3773',
            'NGC3938', 'NGC4236', 'NGC4254', 'NGC4321', 'NGC4536', 'NGC4559',
            'NGC4569', 'NGC4579', 'NGC4594', 'NGC4625', 'NGC4631', 'NGC4725',
            'NGC4736', 'NGC4826', 'NGC5055', 'NGC5398', 'NGC5408', 'NGC5457',
            'NGC5474', 'NGC5713', 'NGC5866', 'NGC6946', 'NGC7331', 'NGC7793']
objects = aobjects if all_objects else selected_objects
# Name: [Capital name, Native Gaussian]
band_char = {'pacs100': ['PACS_100', 'Gauss_15', 15.0],
             'pacs160': ['PACS_160', 'Gauss_15', 15.0],
             'spire250': ['SPIRE_250', 'Gauss_30', 30.0],
             'spire350': ['SPIRE_350', 'Gauss_30', 30.0],
             'spire500': ['SPIRE_500', 'Gauss_41', 41.0]}
bands = list(band_char.keys())
# bands = bands[:4] 


def mp_regrid(o, bands):
    nbands = len(bands)
    if nbands < 3:
        tfn = dpath + o + '_scanamorphos_v16.9_' + bands[nbands - 1] + \
            '_0.fits'
    else:
        tfn = dpath + o + '_kingfish_' + bands[nbands - 1] + '_v3-0_scan.fits'
    tdata, thdr = fits.getdata(tfn, header=True)
    #
    # -1. Comment data
    #
    # -1-0. Grab information from gal_data
    gdata = gal_data(o)
    ra = Angle(gdata.field('RA_DEG')[0] * u.deg)
    dec = Angle(gdata.field('DEC_DEG')[0] * u.deg)
    posang = Angle(gdata.field('POSANG_DEG')[0] * u.deg)
    if np.isnan(posang):
        error_msg('output/KINGFISH_fitting_error.txt',
                  o + ': POSANG is NaN.')
        posang = Angle(0.0 * u.deg)
    incl = Angle(gdata.field('INCL_DEG')[0] * u.deg)
    r25 = Angle(gdata.field('R25_DEG')[0] * u.deg).arcsec
    for bi in range(nbands):
        band = bands[bi]
        job_name = o + '.' + band
        #
        # 0. Load data
        #
        print('\n##', job_name + ': Load image')
        if band[:4] == 'pacs':
            fn = dpath + o + '_scanamorphos_v16.9_' + band + '_0.fits'
        else:
            fn = dpath + o + '_kingfish_' + band + '_v3-0_scan.fits'
        data, hdr = fits.getdata(fn, header=True)
        ps = hdr['PFOV']
        # 0-1. Correct PACS unit and SPIRE dimension
        if band[:4] == 'pacs':
            print('\n##', job_name + ': PACS unit conversion')
            data = data[0]
            data *= (np.pi / 36 / 18)**(-2) / ps**2
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
        if bi != (nbands - 1):
            # 4-1. Convolve image and mask
            print('\n##', job_name + ': Load spire500 kernel')
            kernel, khdr = idcio.load_kernel(kpath, band_char[band][0],
                                             'SPIRE_500')
            kernel /= np.nansum(kernel)
            print('\n##', job_name + ': Convolve to SPIRE500')
            data = idcio.convolve_map(data, hdr, kernel, khdr,
                                      job_name=job_name)
            # 4-2. Regrid image
            print('\n##', job_name + ': Regrid')
            data = idcio.regrid(data, hdr, tdata, thdr, job_name=job_name)
        else:
            bitpix = abs(int(hdr['BITPIX']))
            if bitpix == 32:
                data = data.astype(np.float32)
            elif bitpix == 16:
                data = data.astype(np.float16)
            fn = dpath2 + o + '_diskmask.fits'
            save_fits_gz(fn, diskmask.astype(int), thdr)
            fn = dpath2 + o + '_radius.arcsec.fits'
            save_fits_gz(fn, radius_map, thdr)
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
        project_name='KINGFISH18', observe_fns=observe_fns,
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
