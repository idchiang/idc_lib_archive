import numpy as np
from astropy.coordinates import Angle
from astropy.io import fits
import astropy.units as u
import idc_lib.idc_io as idcio
from idc_lib.idc_functions import save_fits_gz
from idc_lib.idc_math import Gaussian_Kernel_C1


Do_regrid_gas_maps = True


#
col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value
band_char = {'pacs100': ['PACS_100', 'Gauss_15', 15.0],
             'pacs160': ['PACS_160', 'Gauss_15', 15.0],
             'spire250': ['SPIRE_250', 'Gauss_30', 30.0],
             'spire350': ['SPIRE_350', 'Gauss_30', 30.0],
             'spire500': ['SPIRE_500', 'Gauss_41', 41.0]}
bands = list(band_char.keys())
kpath = 'data/Kernels/'
#


def regrid_gas_maps(target='IC342', HI=True, CO=True,
                    RES=['spire350', 'spire500']):
    #
    for bi in range(2):
        #
        # 0. Load data
        #
        if bi == 0:
            job_name = target + ' HI'
            if HI:
                fn = 'data/EveryTHINGS/' + target + '.m0.fits.gz'
            else:
                continue
        elif bi == 1:
            job_name = target + ' CO'
            if CO:
                fn = 'data/PHANGS/' + target.lower() + \
                    '_12co10-20kms-d-30m.mom0.fits'
            else:
                continue
        print('\n##', job_name + ': Load image')
        data, hdr = fits.getdata(fn, header=True)
        # 0-1. Correct data dimension and units
        if bi == 0:
            data = data[0, 0]
            bmaj = Angle(hdr['BMAJ'] * u.deg).arcsec
            bmin = Angle(hdr['BMIN'] * u.deg).arcsec
            data *= col2sur * 1.823E18 * 6.07E5 / bmaj / bmin
        #
        # 4. Convolution
        #
        # 4-0. Convolve to Gauss 25 first.
        print('\n##', job_name + ': Build Gauss 25 Kernel')
        bmaj = Angle(hdr['BMAJ'] * u.deg).arcsec
        bmin = Angle(hdr['BMIN'] * u.deg).arcsec
        bpa = hdr['BPA']
        ps_temp = np.abs(Angle(hdr['CDELT1'] * u.deg).arcsec)
        ps = [ps_temp, ps_temp]
        gauss_kernel = Gaussian_Kernel_C1(ps, bpa, bmaj, bmin, FWHM=25)
        print('\n##', job_name + ': Convolve to Gauss 25')
        data = idcio.convolve_map(data, hdr, gauss_kernel, None,
                                  job_name=job_name)
        #
        tname = 'IC0342' if target == 'IC342' else target
        for tn in RES:
            tfn = 'data/KINGFISH_DR3/' + tname + '_kingfish_' + tn + \
                '_v3-0_scan.fits'
            tdata, thdr = fits.getdata(tfn, header=True)
            if tn == 'spire350':
                tnc = 'SPIRE_350'
            elif tn == 'spire500':
                tnc = 'SPIRE_500'
            else:
                tnc = ''
            # 4-1. Convolve image and mask
            print('\n##', job_name + ': Load', tn, 'kernel')
            kernel, khdr = idcio.load_kernel(kpath, 'Gauss_25', tnc)
            kernel /= np.nansum(kernel)
            print('\n##', job_name + ': Convolve to', tnc)
            cdata = idcio.convolve_map(data, hdr, kernel, khdr,
                                       job_name=job_name)
            # 4-2. Regrid image
            print('\n##', job_name + ': Regrid')
            rdata = idcio.regrid(cdata, hdr, tdata, thdr, job_name=job_name)
            if bi == 0:
                fn = 'data/EveryTHINGS/' + target + '.' + tn + '.HI.fits'
                thdr['DATAUNIT'] = 'Msun/pc2'
            elif bi == 1:
                fn = 'data/PHANGS/' + target + '.' + tn + '.CO.fits'
                thdr['DATAUNIT'] = 'K*km/s'
            save_fits_gz(fn, rdata, thdr)


if __name__ == "__main__":
    #
    if Do_regrid_gas_maps:
        regrid_gas_maps()
