import os
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
from sklearn import linear_model
from reproject import reproject_exact
from reproject import reproject_interp
from h5py import File
from time import clock
from . import idc_regrid as ric
from .gal_data import gal_data
from .idc_math import reasonably_close as cls
from .idc_math import Gaussian_Kernel_C1 as GK1
import matplotlib.pyplot as plt
plt.ioff()
# plt.ion()
col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value
H2HaHe = 1.36


class MGS(object):
    def __init__(self, names=[], surveys=[]):
        """ Initialize the class; Can take optional initial inputs """
        self.df = pd.DataFrame()
        self.kernels = pd.DataFrame()
        self.new = np.full(len(names), np.nan, object)
        if names and surveys:
            self.add_galaxies(names, surveys)

    def add_galaxies(self, names, surveys):
        """ Import fits files """
        names = [names] if type(names) == str else names
        surveys = [surveys] if type(surveys) == str else surveys
        for n in ['THINGS', 'SPIRE_500']:
            if n not in surveys:
                surveys.append(n)
        for name in names:
            self.add_galaxy(name, surveys)

    def add_galaxy(self, name, surveys, filenames=None, uncfn=None,
                   rmbkg=True):
        """  Import fits files from one galaxy """
        print('Importing', len(surveys), 'fits files of', name + '...')
        tic = clock()
        name = name.replace('_', '').replace(' ', '').upper()
        try:
            s = self.df.loc[name]
            self.df = self.df.drop(name)
        except KeyError:
            s = pd.Series(name=name)
            temp_data = gal_data(name, galdata_dir='data/gal_data')
            s['DIST_MPC'] = temp_data.field('DIST_MPC')[0]
            s['RA_RAD'] = Angle(temp_data.field('RA_DEG')[0] * u.deg).rad
            s['DEC_RAD'] = Angle(temp_data.field('DEC_DEG')[0] * u.deg).rad
            s['PA_RAD'] = Angle(temp_data.field('POSANG_DEG')[0] * u.deg).rad
            s['cosINCL'] = np.cos(Angle(temp_data.field('INCL_DEG')[0] *
                                        u.deg).rad)
            s['R25_KPC'] = Angle(temp_data.field('R25_DEG')[0] * u.deg).rad * \
                (s['DIST_MPC'] * 1E3)
            del temp_data

        if name[:3] == 'NGC':
            name1, name2 = 'NGC', name[3:]
        elif name == 'UGC05139':
            name1, name2 = 'HO', 'I'
        elif name == 'UGC04305':
            name1, name2 = 'HO', 'II'
        elif name == 'PGC023521':
            name1, name2 = 'M81', 'dwA'
        elif name == 'UGC05423':
            name1, name2 = 'M81', 'dwB'
        elif name == 'UGC04459':
            name1, name2 = 'DDO', '053'
        elif name == 'NGC4789A':
            name1, name2 = 'DDO', '154'
        elif name[:2] == 'IC':
            name1, name2 = 'IC', name[2:]
        else:
            raise ValueError(name + ' not in database.')

        for survey in surveys:
            print(' --' + survey)
            if name1 == 'M81':
                if survey == 'THINGS':
                    filename = 'data/THINGS/M81_' + \
                        name2.lstrip('0').upper() + '_NA_MOM0_THINGS.FITS'
                elif survey[:5] == 'SPIRE':
                    filename = 'data/SPIRE/M81' + name2 + '_I_' + \
                        survey[6:] + 'um_scan_k2011.fits.gz'
                elif survey[:4] == 'PACS':
                    filename = 'data/PACS/M81' + name2 + '_I_' + \
                        survey[5:] + 'um_k2011.fits.gz'
                elif survey == 'HERACLES':
                    filename = 'data/HERACLES/m81' + name2.lower() + \
                        '_heracles_mom0.fits.gz'
                else:
                    raise ValueError(survey + ' not supported.')
            elif name1 == 'DDO' and survey == 'THINGS':
                filename = 'data/THINGS/' + name1 + name2.lstrip('0') + \
                    '_NA_MOM0_THINGS.FITS'
            elif name1 == 'HO' or name1.upper() == 'HOLMBERG':
                name2 = name2.upper()
                if survey[:5] == 'SPIRE':
                    name1 = 'Holmberg'
                elif survey[:4] == 'PACS':
                    name1 = 'HOLMBERG'
                elif survey == 'HERACLES':
                    name2 = name2.lower()

            uncfn = None
            if not filenames:
                if survey == 'THINGS':
                    filename = 'data/THINGS/' + name1 + '_' + \
                        name2.lstrip('0') + '_NA_MOM0_THINGS.FITS'
                elif survey[:5] == 'SPIRE':
                    filename = 'data/SPIRE/' + name1 + '_' + name2 + '_I_' + \
                        survey[6:] + 'um_scan_k2011.fits.gz'
                    uncfn = 'data/SPIRE/' + name1 + '_' + name2 + '-I-' + \
                        survey[6:] + 'um_s_unc-k2011.fits'
                elif survey[:4] == 'PACS':
                    filename = 'data/PACS/' + name1 + '_' + name2 + '_I_' + \
                        survey[5:] + 'um_k2011.fits.gz'
                elif survey == 'HERACLES':
                    filename = 'data/HERACLES/' + name1.lower() + name2 + \
                        '_heracles_mom0.fits.gz'
                    uncfn = 'data/HERACLES/' + name1.lower() + name2 + \
                        '_heracles_emom0.fits.gz'
                elif survey == 'KINGFISH_DUST':
                    filename = 'data/KINGFISH/' + name1 + name2 + \
                               '_S500_110_SSS_111_Model_SurfBr_Mdust.fits.gz'
                    uncfn = 'data/KINGFISH/' + name1 + name2 + \
                            '_S500_110_SSS_111_Model_SurfBr_Mdust_unc.fits.gz'
                elif survey[:5] == 'GALEX':
                    filename = 'data/GALEX/' + name1 + '_' + name2 + \
                        '_I_' + survey[6:] + '_d2009.fits.gz'
                elif survey[:4] == 'MIPS':
                    filename = 'data/MIPS/' + name1 + '_' + name2 + \
                        '_I_MIPS' + survey[5:] + '_d2009.fits.gz'
                elif survey[:4] == 'IRAC':
                    filename = 'data/IRAC/' + name1 + '_' + name2 + \
                        '_I_IRAC_' + survey[5:] + '_d2009.fits.gz'
                else:
                    raise ValueError(survey + ' not supported.')

            # Start to read data
            data, hdr = fits.getdata(filename, 0, header=True)
            hdr['NAXIS'] = 3
            hdr['NAXIS3'] = 2
            hdr['BUNIT'] = 'MJy/sr'
            hdr.comments['BUNIT'] = '[IDC modified]'
            hdr['CTYPE3'] = 'Signal map / Error map'
            hdr.comments['CTYPE3'] = '[IDC modified]'
            try:
                idx = 4
                while(1):
                    hdr.remove('NAXIS' + str(idx))
                    idx += 1
            except KeyError:
                pass
            if uncfn:
                uncdata = fits.getdata(uncfn, 0, header=False)
            elif survey[:4] == 'PACS':
                uncdata = data[1]
            else:
                uncdata = np.zeros(data.shape)

            if survey in ['THINGS', 'HERACLES']:
                temp_s = pd.read_csv('data/Tables/galaxy_data.csv',
                                     index_col=0).loc[name]
                if survey == 'THINGS':
                    # THINGS: Raw data in JY/B*M/s. Change to
                    # column density 1/cm^2
                    data = data[0, 0]
                    data *= 1.823E18 * 6.07E5 / 1.0E3 / temp_s['TBMAJ'] / \
                        temp_s['TBMIN']
                    uncdata = data * 0.05
                    hdr['BUNIT'] = '1/cm^2'
                elif survey == 'HERACLES':
                    # HERACLES: Raw data in K*km/s. Change to
                    # surface density M_\odot/pc^2
                    # He included
                    # This is a calculated parameter by fitting HI to H2 mass
                    R21 = 1 / 0.7
                    data *= R21 * temp_s['ACO']
                    uncdata *= R21 * temp_s['ACO']
                    hdr['BUNIT'] = 'Solar_mass/pc^2'
                del temp_s
            elif survey[:4] == 'PACS':
                data = data[0]
                # print survey + " not supported for density calculation!!"
            # Extended sources correction
            elif survey == 'SPIRE_500':
                data *= 0.9195
                uncdata *= 0.9195
            elif survey == 'SPIRE_350':
                data *= 0.9351
                uncdata *= 0.9351
            elif survey == 'SPIRE_250':
                data *= 0.9282
                uncdata *= 0.9282
            elif survey == 'IRAC_3.6':
                data *= 0.91
                uncdata *= 0.91
            w = WCS(hdr, naxis=2)
            s[survey + '_WCS'] = w
            s[survey + '_UNCMAP'] = uncdata
            ctr = np.array(data.shape) // 2
            ps = np.zeros(2)
            xs, ys = \
                w.wcs_pix2world([ctr[0] - 1, ctr[0] + 1, ctr[0], ctr[0]],
                                [ctr[1], ctr[1], ctr[1] - 1, ctr[1] + 1], 1)
            ps[0] = np.abs(xs[0] - xs[1]) / 2 * \
                np.cos(Angle((ys[0] + ys[1]) * u.deg).rad / 2)
            ps[1] = np.abs(ys[3] - ys[2]) / 2
            ps *= u.degree.to(u.arcsec)
            if survey in ['PACS_160', 'PACS_100']:
                # Converting Jy/pixel to MJy/sr
                data *= (np.pi / 36 / 18)**(-2) / ps[0] / ps[1]
            elif (survey[:5] == 'GALEX'):
                data *= 1.073E-10 * (np.pi / 3600 / 180)**(-2) / ps[0] / \
                    ps[1]
                filename2 = 'data/GALEX/' + name1 + '_' + name2 + \
                    '-fd-rrhr.fits.gz'
                mask = fits.getdata(filename2, 0, header=False)
                data[mask < 0] = np.nan
                """
                xcm, ycm = float(hdr['CRPIX1']), float(hdr['CRPIX1'])
                coords = np.array(np.meshgrid(np.arange(data.shape[1]),
                                              np.arange(data.shape[0]))
                                  ).T.astype(float)
                # 1470 pixels from center to radius
                # 2000 from center to right-most boarder
                # But the initial max is at the corners...
                zeroCount = 0
                ii = 0
                while(True):
                    ii += 1
                    radii = np.sqrt((coords[:, :, 0] - xcm)**2 +
                                    (coords[:, :, 1] - ycm)**2)
                    r = np.max(radii) / np.sqrt(2)
                    with np.errstate(invalid='ignore'):
                        rs = np.linspace(0, r, 3000)
                        for i in range(len(rs))[::-1]:
                            if np.sum(data[radii > rs[i]] > 0) > 0:
                                r = rs[i]
                                currentCount = np.sum(radii > rs[i + 1])
                                if currentCount > zeroCount:
                                    zeroCount = currentCount
                                    currentMask = radii > rs[i + 1]
                                break
                        DeltaX = (np.mean(coords[:, :, 0][(radii > r) *
                                          (data > 0)]) - xcm) / r
                        DeltaY = (np.mean(coords[:, :, 1][(radii > r) *
                                          (data > 0)]) - ycm) / r
                        if (DeltaX**2 + DeltaY**2 < 0.4**2) | \
                                (currentCount < zeroCount):
                            break
                        xcm += DeltaX
                        ycm += DeltaY
                        r *= 1.05
                data[currentMask] = np.nan
                """
            s[survey + '_PS'] = ps
            s[survey + '_HDR'] = hdr
            s[survey + '_CVL'] = True if survey == 'SPIRE_500' else False
            s[survey + '_RGD'] = True if survey == 'SPIRE_500' else False
            s[survey + '_BITPIX'] = abs(int(hdr['BITPIX']))
            s[survey] = data
        s['RADIUS_KPC'] = self.dp_radius(s)
        s['RADIUS_KPC_RGD'] = True
        # Update DataFrame
        self.df = self.df.append(s)
        for survey in surveys:
            self.df[survey + '_CVL'] = self.df[survey + '_CVL'].astype(bool)
            self.df[survey + '_RGD'] = self.df[survey + '_RGD'].astype(bool)
            if (survey not in ['THINGS', 'HERACLES', 'DYAS18']) and rmbkg:
                self.bkg_removal(name, survey)
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def add_kernel(self, name1s, name2):
        """ Ad Kernels """
        FWHM = {'SPIRE_500': 36.09, 'SPIRE_350': 24.88, 'SPIRE_250': 18.15,
                'Gauss_25': 25, 'PACS_160': 11.18, 'PACS_100': 7.04,
                'HERACLES': 13, 'GALEX_FUV': 4.48, 'MIPS_24': 6.43,
                'IRAC_3.6': 1.9}
        # Note: pixel scale of SPIRE 500 ~ 14.00
        name1s = [name1s] if type(name1s) == str else name1s

        print("Importing", len(name1s), "kernel files...")
        tic = clock()
        for name1 in name1s:
            s = pd.Series()
            filename = 'data/Kernels/Kernel_LoRes_' + name1 + '_to_' + \
                name2 + '.fits.gz'
            s['KERNEL'], hdr = fits.getdata(filename, 0, header=True)
            s['KERNEL'] /= np.nansum(s['KERNEL'])
            assert hdr['CD1_1'] == hdr['CD2_2']
            assert hdr['CD1_2'] == hdr['CD2_1'] == 0
            assert hdr['CD1_1'] % 2
            s['PS'] = np.array([hdr['CD1_1'] * u.degree.to(u.arcsec),
                                hdr['CD2_2'] * u.degree.to(u.arcsec)])
            s['FWHM1'], s['FWHM2'], s['NAME1'], s['NAME2'] = \
                FWHM[name1], FWHM[name2], name1, name2
            self.kernels = \
                self.kernels.append(s.to_frame().T.
                                    set_index(['NAME1', 'NAME2']))
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def dp_radius(self, s, survey='SPIRE_500'):
        """ Calculate the radius at each point """
        shape = np.array(s[survey].shape)
        cosPA, sinPA = np.cos(s['PA_RAD']), np.sin(s['PA_RAD'])
        cosINCL = s['cosINCL']
        w = s[survey + '_WCS']
        xcm, ycm = s['RA_RAD'], s['DEC_RAD']
        dp_coords = np.zeros([shape[0], shape[1], 2])
        # Original coordinate is (y, x)
        # :1 --> x, RA --> the one needed to be divided by cos(incl)
        # :0 --> y, Dec
        dp_coords[:, :, 0], dp_coords[:, :, 1] = \
            np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # Now, value inside dp_coords is (x, y)
        # :0 --> x, RA --> the one needed to be divided by cos(incl)
        # :1 --> y, Dec
        for i in range(shape[0]):
            dp_coords[i] = Angle(w.wcs_pix2world(dp_coords[i], 1) * u.deg).rad
        dp_coords[:, :, 0] = 0.5 * (dp_coords[:, :, 0] - xcm) * \
            (np.cos(dp_coords[:, :, 1]) + np.cos(ycm))
        dp_coords[:, :, 1] -= ycm
        # Now, dp_coords is (dx, dy) in the original coordinate
        # cosPA*dy-sinPA*dx is new y
        # cosPA*dx+sinPA*dy is new x
        if survey[:5] == 'GALEX':
            return np.sqrt((cosPA * dp_coords[:, :, 1] +
                            sinPA * dp_coords[:, :, 0])**2 +
                           ((cosPA * dp_coords[:, :, 0] -
                             sinPA * dp_coords[:, :, 1]))**2) * \
                s['DIST_MPC'] * 1.0E3  # Radius in kpc
        else:
            return np.sqrt((cosPA * dp_coords[:, :, 1] +
                            sinPA * dp_coords[:, :, 0])**2 +
                           ((cosPA * dp_coords[:, :, 0] -
                             sinPA * dp_coords[:, :, 1]) / cosINCL)**2) * \
                s['DIST_MPC'] * 1.0E3  # Radius in kpc

    def matching_PSF(self, names, survey1s, survey2):
        """
        Inputs:
            names, survey1s: <str>
                Object names and survey names to be convolved.
            survey2: <str>
                Name of target PSF.
        """
        names = [names] if type(names) == str else names
        survey1s = [survey1s] if type(survey1s) == str else survey1s
        for survey1 in survey1s:
            print('Convoling', len(names), 'galaxies in', survey1 + '...')
            tic = clock()
            ps = self.df.loc[names[0]][survey1 + '_PS']
            if survey1 == 'HERACLES':
                bmaj, bmin, bpa, FWHM1, FWHM2 = 13.0, 13.0, 0.0, 13.0, 25.0
                try:
                    kernel = self.kernels.loc[survey1, 'Gauss_25']['KERNEL']
                except KeyError:
                    s = pd.Series()
                    Gkernel = GK1(ps, bpa, bmaj, bmin, FWHM2)
                    s['KERNEL'], s['PS'] = Gkernel, ps
                    s['FWHM1'], s['FWHM2'], s['NAME1'], s['NAME2'] = \
                        FWHM1, FWHM2, survey1, 'Gauss_25'
                    self.kernels = \
                        self.kernels.append(s.to_frame().T.
                                            set_index(['NAME1', 'NAME2']))
                for name in names:
                    map0 = self.df.loc[name][survey1]
                    uncmap0 = self.df.loc[name][survey1 + '_UNCMAP']
                    cvl_image, cvl_unc = \
                        ric.matching_PSF(Gkernel, FWHM1, FWHM2, map0, uncmap0)
                    self.df.at[name, survey1] = cvl_image
                    self.df.at[name, survey1 + '_UNCMAP'] = cvl_unc

            survey_load = 'Gauss_25' if survey1 in ['THINGS', 'HERACLES'] \
                else survey1
            kernel = self.kernels.loc[survey_load, survey2]['KERNEL']
            FWHM1 = self.kernels.loc[survey_load, survey2]['FWHM1']
            FWHM2 = self.kernels.loc[survey_load, survey2]['FWHM2']
            if not cls(ps, self.kernels.loc[survey_load, survey2]['PS'], 2.0):
                ps_old = self.kernels.loc[survey_load, survey2]['PS']
                self.kernels.at[(survey_load, survey2), 'PS'] = ps
                kernel = ric.Kernel_regrid(kernel, ps, ps_old)
                self.kernels.at[(survey_load, survey2), 'KERNEL'] = kernel

            for name in names:
                fn = 'data/PROCESSED/' + name + '/' + survey1 + '_RGD.fits'
                try:
                    data = fits.getdata(fn, 0, header=False)
                    del data
                except FileNotFoundError:
                    map0 = self.df.loc[name][survey1]
                    uncmap0 = self.df.loc[name][survey1 + '_UNCMAP']
                    if survey1 == 'THINGS':
                        temp_s = pd.read_csv('data/Tables/galaxy_data.csv',
                                             index_col=0).loc[name]
                        bmaj, bmin, bpa = \
                            temp_s['TBMAJ'], temp_s['TBMIN'], temp_s['TBPA']
                        Gkernel = GK1(ps, bpa, bmaj, bmin, 25.0)
                        map0, uncmap0 = \
                            ric.matching_PSF(Gkernel, np.sqrt(bmaj * bmin),
                                             25.0, map0, uncmap0)
                    cvl_image, cvl_unc = \
                        ric.matching_PSF(kernel, FWHM1, FWHM2, map0, uncmap0)
                    self.df.at[name, survey1] = cvl_image
                    self.df.at[name, survey1 + '_UNCMAP'] = cvl_unc
                    self.df.at[name, survey1 + '_CVL'] = True
            print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def WCS_congrid(self, names, fine_surveys, course_survey, method='linear'):
        names = [names] if type(names) == str else names
        fine_surveys = [fine_surveys] if type(fine_surveys) == str else \
            fine_surveys
        for fine_survey in fine_surveys:
            print('Regridding', len(names), 'galaxies in', fine_survey + '...')
            tic = clock()
            for name in names:
                fn = 'data/PROCESSED/' + name + '/' + fine_survey + '_RGD.fits'
                try:
                    rgd_image, rgd_unc = fits.getdata(fn, 0, header=False)
                    if self.df.loc[name][fine_survey + '_BITPIX'] == 32:
                        rgd_image, rgd_unc = rgd_image.astype(np.float32), \
                            rgd_unc.astype(np.float32)
                    elif self.df.loc[name][fine_survey + '_BITPIX'] == 16:
                        rgd_image, rgd_unc = rgd_image.astype(np.float16), \
                            rgd_unc.astype(np.float16)
                except FileNotFoundError:
                    assert self.df.loc[name][fine_survey + '_CVL']
                    """
                    # Old code
                    rgd_image, rgd_unc = \
                        ric.WCS_congrid(self.df.loc[name][fine_survey],
                                        self.df.loc[name][fine_survey +
                                                          '_UNCMAP'],
                                        self.df.loc[name][fine_survey +
                                                          '_WCS'],
                                        self.df.loc[name][course_survey +
                                                          '_WCS'],
                                        self.df.loc[name][course_survey].shape,
                                        method)
                    """
                    # New code
                    s_out = self.df.loc[name][course_survey].shape
                    w_out = self.df.loc[name][course_survey + '_WCS']
                    w_in = self.df.loc[name][fine_survey + '_WCS']
                    rgd_image, _ = \
                        reproject_exact((self.df.loc[name][fine_survey],
                                         w_in), w_out, s_out)
                    temp_unc = np.abs(self.df.loc[name][fine_survey +
                                                        '_UNCMAP'])**2
                    rgd_unc, _ = \
                        reproject_exact((temp_unc, w_in), w_out, s_out)
                    rgd_unc = np.sqrt(rgd_unc)
                    if self.df.loc[name][fine_survey + '_BITPIX'] == 32:
                        rgd_image, rgd_unc = rgd_image.astype(np.float32), \
                            rgd_unc.astype(np.float32)
                    elif self.df.loc[name][fine_survey + '_BITPIX'] == 16:
                        rgd_image, rgd_unc = rgd_image.astype(np.float16), \
                            rgd_unc.astype(np.float16)
                    hdu = fits.PrimaryHDU(np.array([rgd_image, rgd_unc]),
                                          header=self.df.loc[name]
                                          [fine_survey + '_HDR'])
                    hdu.header.comments['NAXIS'] = \
                        '[IDC modified] Number of data axes.'
                    hdu.header.comments['NAXIS3'] = \
                        '[IDC modified] Signal / Error'
                    hdu.writeto(fn)
                self.df.at[name, fine_survey] = rgd_image
                self.df.at[name, fine_survey + '_UNCMAP'] = rgd_unc
                print(self.df[fine_survey + '_RGD'])
                self.df.at[name, fine_survey + '_RGD'] = True
            print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def SFR(self, names, mode=1):
        """
        mode 1: GALEX FUV + MIPS 24 --> SFR
        """
        d = {1: 'GALEX FUV + MIPS 24...'}
        names = [names] if type(names) == str else names
        print('Calculating SFR of', len(names), 'galaxies from', d[mode])
        tic = clock()
        for name in names:
            if mode == 1:
                # SFR in Solar mass / kpc^2 / yr
                SFR = self.df.loc[name]['cosINCL'] * \
                    (0.081 * self.df.loc[name]['GALEX_FUV'] +
                     0.0032 * self.df.loc[name]['MIPS_24'])
            try:
                self.df.at[name, 'SFR'] = SFR
            except ValueError:
                self.df['SFR'] = self.new
                self.df.at[name, 'SFR'] = SFR
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def SMSD(self, names):
        """
        Convert IRAC 3.6 micron to stellar mass density according to Leroy+08
        """
        names = [names] if type(names) == str else names
        print('Calculating stellar mass surface density of', len(names),
              'galaxies from IRAC 3.6...')
        tic = clock()
        for name in names:
            # SMD in solar mass / pc^2
            smsd = self.df.loc[name]['IRAC_3.6'] * \
                self.df.loc[name]['cosINCL'] * 350
            try:
                self.df.at[name, 'SMSD'] = smsd
            except ValueError:
                self.df['SMSD'] = self.new
                self.df.at[name, 'SMSD'] = smsd
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def total_gas(self, names):
        names = [names] if type(names) == str else names
        print('Calculating total gas mass surface density of', len(names),
              'galaxies from THINGS and HERACLES...')
        tic = clock()
        for name in names:
            things, heracles = self.df.loc[name]['THINGS'], \
                self.df.loc[name]['HERACLES']
            things_unc, heracles_unc = self.df.loc[name]['THINGS_UNCMAP'], \
                self.df.loc[name]['HERACLES_UNCMAP']
            nan_things, nan_heracles = np.isnan(things), np.isnan(heracles)
            things[nan_things], heracles[nan_heracles] = 0.0, 0.0
            things_unc[nan_things], heracles_unc[nan_heracles] = 0.0, 0.0
            total_gas = col2sur * H2HaHe * things + heracles
            total_gas_unc = col2sur * H2HaHe * things_unc + heracles_unc
            total_gas[nan_things * nan_heracles] = np.nan
            total_gas_unc[nan_things * nan_heracles] = np.nan
            try:
                self.df.at[name, 'TOTAL_GAS'] = total_gas
                self.df.at[name, 'TOTAL_GAS_UNCMAP'] = total_gas_unc
            except ValueError:
                self.df['TOTAL_GAS'] = self.new
                self.df['TOTAL_GAS_UNCMAP'] = self.new
                self.df.at[name, 'TOTAL_GAS'] = total_gas
                self.df.at[name, 'TOTAL_GAS_UNCMAP'] = total_gas_unc
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def BDR_cal(self, names, course_survey='SPIRE_500'):
        names = [names] if type(names) == str else names
        print(' --Calculating boundary cutting of', len(names),
              'galaxies from THINGS...')
        tic = clock()
        for name in names:
            if not self.df.loc[name]['THINGS_RGD']:
                print('THINGS map not regridded yet!!')
                return 0
            things = self.df.loc[name]['THINGS']
            axissum = [0] * 2
            lc = np.zeros([2, 2], dtype=int)
            for i in range(2):
                axissum[i] = np.nansum(things, axis=i, dtype=bool)
                for j in range(len(axissum[i])):
                    if axissum[i][j]:
                        lc[i-1, 0] = j
                        break
                lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)
            try:
                self.df.at[name, 'BDR'] = lc
            except ValueError:
                self.df['BDR'] = self.new
                self.df.at[name, 'BDR'] = lc
            hdr_in = self.df.loc[name][course_survey + '_HDR']
            hdr_in['CRPIX1'] -= lc[1, 0]
            hdr_in['CRPIX2'] -= lc[0, 0]
            fn = 'data/PROCESSED/' + name + '/' + course_survey + '_CRP.fits'
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
            hdu = fits.PrimaryHDU(self.df.loc[name][course_survey]
                                  [lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]],
                                  header=hdr_in)
            hdu.writeto(fn)
        print("   --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
        return lc

    def crop_image(self, names, surveys, unc=True):
        names = [names] if type(names) == str else names
        print('Cropping', len(names), 'galaxies according to THINGS...')
        tic = clock()
        surveys = [surveys] if type(surveys) == str else surveys
        for survey in surveys:
            for name in names:
                try:
                    lc = self.df.loc[name]['BDR']
                except KeyError:
                    lc = self.BDR_cal(name)
                # assert self.df.loc[name][survey + '_RGD']
                data = self.df.\
                    loc[name][survey][lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
                self.df.at[name, survey] = data
                if unc:
                    uncmap = self.df.\
                        loc[name][survey + '_UNCMAP'][lc[0, 0]:lc[0, 1],
                                                      lc[1, 0]:lc[1, 1]]
                    self.df.at[name, survey + '_UNCMAP'] = uncmap
                if survey == 'THINGS':
                    with np.errstate(invalid='ignore'):
                        diskmask = data > self.THINGS_Limit
                    try:
                        self.df.at[name, 'DISKMASK'] = diskmask
                    except ValueError:
                        self.df['DISKMASK'] = self.new
                        self.df.at[name, 'DISKMASK'] = diskmask
                elif survey[:9] == 'HERSCHEL_':
                    t = self.df.loc[name][survey + '_DISKMASK']
                    t = t[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
                    self.df.at[name, survey + '_DISKMASK'] = t
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def bkg_removal(self, names, surveys, THINGS_Limit=1.0E18, bdr=0.5):
        self.THINGS_Limit = THINGS_Limit
        names = [names] if type(names) == str else names
        surveys = [surveys] if type(surveys) == str else surveys
        print('Removing background of', len(names), 'galaxies in',
              len(surveys), 'surveys...')
        tic = clock()
        for name in names:
            with np.errstate(invalid='ignore'):
                disk = (self.df.loc[name]['THINGS'] > THINGS_Limit).astype(int)
                tw = self.df.loc[name]['THINGS_WCS']
            for survey in surveys:
                data = self.df.loc[name][survey]
                s_out = self.df.loc[name][survey].shape
                w_out = self.df.loc[name][survey + '_WCS']
                #
                # Defining background region
                rgd_disk, _ = reproject_interp((disk, tw), w_out, s_out)
                rgd_disk[np.isnan(rgd_disk)] = 0
                rgd_disk[rgd_disk > bdr] = 1
                bkg_mask = (~(rgd_disk.astype(int).astype(bool))) * \
                    (~np.isnan(data))
                """
                #
                # Selecting regions for plotting local histogram
                bkg_mask_copy = np.copy(bkg_mask).astype(int)
                bkg_mask_copy -= 1
                p1s, p0s = np.meshgrid(np.arange(bkg_mask.shape[1]),
                                       np.arange(bkg_mask.shape[0]))
                p0s, p1s = p0s[bkg_mask], p1s[bkg_mask]
                local_masks = []
                num_of_locals = 4
                width = int(np.sqrt((np.sum(bkg_mask) / 20)))
                ii = 0
                while(ii < num_of_locals):
                    jj = np.random.randint(0, len(p0s))
                    p0, p1 = p0s[jj], p1s[jj]
                    if (p0 < bkg_mask.shape[0] - width) & \
                            (p1 < bkg_mask.shape[1] - width):
                        temp_local_mask = np.zeros(bkg_mask.shape, dtype=bool)
                        for jj in range(p0, p0 + width):
                            for kk in range(p1, p1 + width):
                                temp_local_mask[jj, kk] = True
                        if np.sum(bkg_mask_copy[temp_local_mask]) == 0:
                            local_masks.append(temp_local_mask)
                            ii += 1
                            bkg_mask_copy[temp_local_mask] = ii
                """
                """
                #
                # Definition of some plotting parameters
                rs = {'GALEX_FUV': (-0.001, 0.008), 'IRAC_3.6': (-0.1, 0.2),
                      'MIPS_24': (-0.05, 0.15), 'PACS_100': (-5.0, 5.0),
                      'PACS_160': (-5.0, 5.0), 'SPIRE_250': (-0.70, 0.90),
                      'SPIRE_350': (-0.70, 0.90), 'SPIRE_500': (-0.70, 0.90)}
                bins = 30
                """
                if survey[:5] != 'GALEX':
                    #
                    # tilted plane new
                    bkg_plane, coef_ = self.bkg_tilted_plane(data, bkg_mask)
                    for i in range(5):
                        full_diff = data - bkg_plane
                        full_AD = np.abs(full_diff -
                                         np.nanmedian(full_diff[bkg_mask]))
                        MAD = np.nanmedian(full_AD[bkg_mask])
                        with np.errstate(invalid='ignore'):
                            MAD_mask = (full_AD <= 3 * MAD) * bkg_mask
                        bkg_plane, coef_ = \
                            self.bkg_tilted_plane(data, MAD_mask)
                    data -= bkg_plane
                else:
                    #
                    # GALEX: just use the mean
                    coef_ = [np.nan] * 3
                    data -= np.nanmean(data[bkg_mask])
                self.df.at[name, survey] = data
                try:
                    self.df.at[name, survey + '_BKG_COEF'] = coef_
                except ValueError:
                    self.df[survey + '_BKG_COEF'] = self.new
                    self.df.at[name, survey + '_BKG_COEF'] = coef_
            print(' --' + name, 'in', survey + ':', bkg_mask.sum(),
                  'effective background pixels.')
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def bkg_tilted_plane(self, image, bkgmask):
        m, n = image.shape
        q, p = np.meshgrid(np.arange(n), np.arange(m))
        #
        regr = linear_model.LinearRegression()
        DataX = np.array([p[bkgmask], q[bkgmask]]).T
        regr.fit(DataX, image[bkgmask])
        bkg_plane = \
            regr.predict(np.array([p.flatten(), q.flatten()]).T).reshape(m, n)
        coef = np.append(regr.coef_, regr.intercept_)
        return bkg_plane, coef

    def covariance_matrix(self, names, surveys):
        names = [names] if type(names) == str else names
        surveys = [surveys] if type(surveys) == str else surveys
        print('Calculating covariance matrix of ', len(names), 'galaxies in',
              len(surveys))
        tic = clock()
        for name in names:
            with np.errstate(invalid='ignore'):
                tbkgmask = ~(self.df.loc[name]['THINGS'] > self.THINGS_Limit)
            for survey in surveys:
                if survey[:9] == 'HERSCHEL_':
                    all_ = np.array(['PACS_70', 'PACS_100', 'PACS_160',
                                     'SPIRE_250', 'SPIRE_350', 'SPIRE_500'])
                    slc = np.array([bool(int(s)) for s in survey[9:]])
                    sub_surveys = all_[slc]
                    s = self.df.loc[name]['THINGS'].shape
                    data = np.zeros([s[0], s[1], len(sub_surveys)])
                    uncmap = np.empty_like(data)
                    for i in range(len(sub_surveys)):
                        data[:, :, i] = self.df.loc[name][sub_surveys[i]]
                        uncmap[:, :, i] = \
                            self.df.loc[name][sub_surveys[i] + '_UNCMAP']
                    bkgmask = tbkgmask * (~np.sum(np.isnan(data), axis=2,
                                                  dtype=bool))
                    bkgcov = np.cov(data[bkgmask].T)
                    try:
                        self.df.at[name, survey] = data
                        self.df.at[name, survey + '_UNCMAP'] = uncmap
                        self.df.at[name, survey + '_BKGCOV'] = bkgcov
                        self.df.at[name, survey + '_DISKMASK'] = ~bkgmask
                    except ValueError:
                        self.df[survey] = self.new
                        self.df[survey + '_UNCMAP'] = self.new
                        self.df[survey + '_BKGCOV'] = self.new
                        self.df[survey + '_DISKMASK'] = self.new
                        self.df.at[name, survey] = data
                        self.df.at[name, survey + '_UNCMAP'] = uncmap
                        self.df.at[name, survey + '_BKGCOV'] = bkgcov
                        self.df.at[name, survey + '_DISKMASK'] = ~bkgmask
                print(' --' + name, 'in', survey + ':', np.sum(bkgmask),
                      'effective background pixels.')
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def save_data(self, names, surveys):
        """
        Inputs:
            names: <list of str | str>
                Object names to be saved.
        """
        names = [names] if type(names) == str else names
        surveys = [surveys] if type(surveys) == str else surveys
        no_unc = ['RADIUS_KPC', 'SFR', 'SMSD', 'DIST_MPC', 'PA_RAD', 'cosINCL',
                  'R25_KPC', 'SPIRE_500_PS']
        for name in names:
            print('Saving', name, 'data...')
            tic = clock()
            try:
                with File('hdf5_MBBDust/' + name + '.h5', 'a') as hf:
                    pass
            except OSError:
                os.mkdir('hdf5_MBBDust')
            with File('hdf5_MBBDust/' + name + '.h5', 'a') as hf:
                grp = hf.require_group('Regrid')
                for survey in surveys:
                    try:
                        del grp[survey]
                    except KeyError:
                        pass
                    grp[survey] = self.df.loc[name][survey]
                    if survey not in no_unc:
                        try:
                            del grp[survey + '_UNCMAP']
                        except KeyError:
                            pass
                        grp[survey + '_UNCMAP'] = \
                            self.df.loc[name][survey + '_UNCMAP']
                    if survey[:9] == 'HERSCHEL_':
                        try:
                            del grp[survey + '_BKGCOV']
                            del grp[survey + '_DISKMASK']
                        except KeyError:
                            pass
                        grp[survey + '_BKGCOV'] = \
                            self.df.loc[name][survey + '_BKGCOV']
                        grp[survey + '_DISKMASK'] = \
                            self.df.loc[name][survey + '_DISKMASK']
            print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")


"""
def metallicity_to_coordinate(names=['NGC5457']):
    names = [names] if type(names) == str else names
    # Reading & convolving
    mgs = MGS(names, ['THINGS', 'SPIRE_500'])
    mgs.add_kernel('Gauss_25', 'SPIRE_500')
    mgs.matching_PSF(names, 'THINGS', 'SPIRE_500')
    mgs.WCS_congrid(names, 'THINGS', 'SPIRE_500')
    mgs.BDR_cal(names)
    for name in names:
        # Extracting some parameters for calculating radius
        lc = mgs.df.loc[name]['BDR']
        w = mgs.df.loc[name]['SPIRE_500_WCS']
        # Calculating radius
        df = pd.read_csv('data/Tables/' + name + '_Z_modified.csv')
        coords = np.empty([len(df), 2])
        for i in range(len(df)):
            ra = df['RAJ2000'].iloc[i].strip().split(' ')
            dec = df['DEJ2000'].iloc[i].strip().split(' ')
            temp1 = ra[0] + 'h' + ra[1] + 'm' + ra[2] + 's ' + dec[0] + 'd' + \
                dec[1] + 'm' + dec[2] + 's'
            temp2 = SkyCoord(temp1)
            coords[i] = np.array([temp2.ra.degree, temp2.dec.degree])
        # Now, save DEC(y) in axis 0, RA(x) in axis 1.
        coords[:, 0], coords[:, 1] = w.wcs_world2pix(coords[:, 0],
                                                     coords[:, 1], 1)
        coords[:, 0] -= lc[0, 0]
        coords[:, 1] -= lc[1, 0]
        df['new_c1'] = coords[:, 0]
        df['new_c2'] = coords[:, 1]
        df.to_csv('output/Metal_' + name + '.csv')
"""
