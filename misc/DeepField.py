import os
import numpy as np
from astropy.io import fits
import idc_lib.idc_io as idcio
from idc_lib.idc_functions import save_fits_gz
from idc_lib.idc_math import Gaussian_Kernel_C1

generating = 0

target_fwhmss = \
    {'lmc': [53.4, 123.3, 205.4, 410.9, 686.2, 1232.7, 2054.4, 4108.9],
     'm31': [46.3, 83.2, 138.6, 277.2, 554.5, 1109.0, 2217.9, 4158.6],
     'm33': [41.0, 73.7, 122.8, 245.6, 491.1, 982.2, 1964.4, 3683.3],
     'smc': [43.2, 99.6, 166.1, 332.1, 554.7, 996.4, 1660.7, 3321.5, 6643.0,
             13286.0]}
target_fwhms = target_fwhmss['smc']
# target_fwhms = [] 

dpath = 'data/NGP/'
kpath = 'data/Kernels/'
ws = [500, 350, 250, 160, 100]
ws = [100, 160, 250, 350, 500]
bands = {100: 'PACS_100', 160: 'PACS_160', 250: 'SPIRE_250', 350: 'SPIRE_350',
         500: 'SPIRE_500'}
beam_area = {100: np.nan, 160: np.nan, 250: 469.35, 350: 831.27, 500: 1804.31}

if generating:
    tfn = dpath + 'HATLAS_NGP_DR2_BACKSUB500.FITS'
    tdata, thdr = fits.getdata(tfn, header=True)
    tdata = np.zeros([2000, 2000])
    thdr['NAXIS1'] = 2000
    thdr['NAXIS2'] = 2000
    thdr['CDELT1'] = -(41 / 2.5) / 3600
    thdr['CDELT2'] = (41 / 2.5) / 3600
    thdr['CRVAL1'] = 199.3129459955868
    thdr['CRPIX1'] = 1000
    thdr['CRVAL2'] = 29.2233931273550
    thdr['CRPIX2'] = 1000
    thdr['UNITS'] = 'MJy/sr'
    for w in ws:
        fn = dpath + 'HATLAS_NGP_DR2_BACKSUB' + str(w) + '.FITS'
        data, hdr = fits.getdata(fn, header=True)
        #
        cut = int(data.shape[1] / 5)
        data = data[:, cut:(-cut)]
        hdr['NAXIS1'] -= 2 * cut
        hdr['CRPIX1'] -= cut
        cut = int(data.shape[0] / 5)
        data = data[cut:(-cut)]
        hdr['NAXIS2'] -= 2 * cut
        hdr['CRPIX2'] -= cut
        #
        # 0. Unit conversion
        #
        if w < 200:
            print('\n##', bands[w] + ': PACS unit conversion')
            ps = np.abs(hdr['CD1_1']) * 3600
            data *= (np.pi / 36 / 18)**(-2) / ps**2
        else:
            print('\n##', bands[w] + ': SPIRE unit conversion')
            data *= (np.pi / 36 / 18)**(-2) / beam_area[w]
        #
        # 1. Convolve to Gauss41.
        #
        print('\n##', bands[w] + ': Load spire500 kernel')
        kernel, khdr = idcio.load_kernel(kpath, bands[w],
                                         'Gauss_41')
        kernel /= np.nansum(kernel)
        print('\n##', bands[w] + ': Convolve to Gauss_41')
        data = idcio.convolve_map(data, hdr, kernel, khdr,
                                  job_name=bands[w])
        #
        # 2. Regrid to smaller SPIRE500
        #
        print('\n##', bands[w] + ': Regrid')
        data = idcio.regrid(data, hdr, tdata, thdr, exact=False,
                            job_name=bands[w])
        #
        # 3. Save
        #
        fn = dpath + bands[w] + '.fits'
        save_fits_gz(fn, data, thdr)
    del tdata, thdr

#
# 4. Load from saved files.
#
sed = []
for w in ws:
    fn = dpath + bands[w] + '.fits.gz'
    data, hdr = fits.getdata(fn, header=True)
    sed.append(data)
sed = np.array(sed)
ps = [np.abs(hdr['CDELT1']) * 3600] * 2
print('Pixel size:', ps)

for fwhm in target_fwhms:
    #
    # 5. Convolve to resolution we want
    #
    if fwhm > 41.0:
        cdata = np.full_like(sed, np.nan)
        kernel = Gaussian_Kernel_C1(ps, 0, 41.0, 41.0, fwhm)
        for i in range(5):
            cdata[i] = idcio.convolve_map(sed[i], None, kernel, None,
                                          job_name=str(fwhm) + '-' + str(i))
    else:
        cdata = np.copy(sed)
    finite_mask = np.all(np.isfinite(cdata), axis=0)
    #
    # 6. Reject outlier. Calculate covariance.
    #
    outlier_mask = np.zeros_like(sed[0], dtype=bool)
    for i in range(5):
        AD = np.abs(cdata[i] - np.nanmedian(cdata[i]))
        MAD = np.nanmedian(AD)
        with np.errstate(invalid='ignore'):
            outlier_mask += AD > 3 * MAD
    bkgcov = np.cov(cdata[:, (~outlier_mask) * finite_mask])
    if not os.path.isdir(dpath + str(fwhm)):
        os.mkdir(dpath + str(fwhm))
    fn = dpath + str(fwhm) + '/smc_bkgcov_df.fits'
    save_fits_gz(fn, bkgcov, None)

"""
target_fwhms = np.logspace(np.log10(50), np.log10(16000), 25) 
bkgcovs = [] 
for fwhm in target_fwhms:
    #
    # 5. Convolve to resolution we want
    #
    if fwhm > 41.0:
        cdata = np.full_like(sed, np.nan)
        kernel = Gaussian_Kernel_C1(ps, 0, 41.0, 41.0, fwhm)
        for i in range(5):
            cdata[i] = idcio.convolve_map(sed[i], None, kernel, None,
                                          job_name=str(fwhm) + '-' + str(i))
    else:
        cdata = np.copy(sed)
    finite_mask = np.all(np.isfinite(cdata), axis=0)
    #
    # 6. Reject outlier. Calculate covariance.
    #
    outlier_mask = np.zeros_like(sed[0], dtype=bool)
    for i in range(5):
        AD = np.abs(cdata[i] - np.nanmedian(cdata[i]))
        MAD = np.nanmedian(AD)
        with np.errstate(invalid='ignore'):
            outlier_mask += AD > 3 * MAD
    bkgcov = np.cov(cdata[:, (~outlier_mask) * finite_mask])
    bkgcovs.append(bkgcov)
bkgcovs = np.array(bkgcovs)
stds = np.sqrt(np.diagonal(bkgcovs, axis1=1, axis2=2))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(5, 5, figsize=(10, 10))
fig2, ax2 = plt.subplots(5, 5, figsize=(10, 10))
plt.ion()
ts = ['100', '160', '250', '350', '500']
for i in range(5):
    for j in range(5):
        ax[i, j].loglog(target_fwhms, bkgcovs[:, i, j])
        ax[i, j].set_title(ts[i] + '-' + ts[j])
        ax2[i, j].semilogx(target_fwhms, bkgcovs[:, i, j] / stds[:, i] /
                           stds[:, j])
        ax2[i, j].set_title(ts[i] + '-' + ts[j])
fig.savefig('output/deepfield_cov.png')
fig2.savefig('output/deepfield_corr.png')
"""
