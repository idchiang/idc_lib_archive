import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from h5py import File
from time import clock
from . import idc_regrid as ric
from .gal_data import gal_data
from .idc_math import reasonably_close as cls
from .idc_math import Gaussian_Kernel_C1 as GK1
col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value
H2HaHe = 1.36


class MGS(object):
    def __init__(self, names=None, surveys=None):
        """ Initialize the class; Can take optional initial inputs """
        self.df = pd.DataFrame()
        self.kernels = pd.DataFrame()
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

    def add_galaxy(self, name, surveys, filenames=None, uncfn=None):
        """  Import fits files from one galaxy """
        print('Importing', len(surveys), 'fits files of', name + '...')
        tic = clock()
        name = name.replace('_', '').replace(' ', '').upper()
        try:
            s = self.df.loc[name]
            self.df = self.df.drop(name)
        except KeyError:
            s = pd.Series(name=name)
            temp_data = gal_data(name)
            s['DIST_MPC'] = temp_data.field('DIST_MPC')[0]
            s['RA_RAD'] = temp_data.field('RA_DEG')[0] * np.pi / 180
            s['DEC_RAD'] = temp_data.field('DEC_DEG')[0] * np.pi / 180
            s['PA_RAD'] = temp_data.field('POSANG_DEG')[0] * np.pi / 180
            s['cosINCL'] = np.cos(temp_data.field('INCL_DEG')[0] * np.pi / 180)
            s['R25_KPC'] = temp_data.field('R25_DEG')[0] * (np.pi / 180.) * \
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
                elif survey == 'HERACLES':
                    # HERACLES: Raw data in K*km/s. Change to
                    # surface density M_\odot/pc^2
                    # He included
                    # This is a calculated parameter by fitting HI to H2 mass
                    R21 = 0.7
                    data *= R21 * temp_s['ACO']
                    uncdata *= R21 * temp_s['ACO']
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

            w = wcs.WCS(hdr, naxis=2)
            s[survey + '_WCS'], s[survey] = w, data
            s[survey + '_UNCMAP'] = uncdata
            ctr = np.array(data.shape) // 2
            ps = np.zeros(2)
            xs, ys = \
                w.wcs_pix2world([ctr[0] - 1, ctr[0] + 1, ctr[0], ctr[0]],
                                [ctr[1], ctr[1], ctr[1] - 1, ctr[1] + 1], 1)
            ps[0] = np.abs(xs[0] - xs[1]) / 2 * np.cos((ys[0] + ys[1]) *
                                                       np.pi / 2 / 180)
            ps[1] = np.abs(ys[3] - ys[2]) / 2
            ps *= u.degree.to(u.arcsec)
            if survey in ['PACS_160', 'PACS_100']:
                # Converting Jy/pixel to MJy/sr
                data *= (np.pi / 36 / 18)**(-2) / ps[0] / ps[1]
            elif survey[:5] == 'GALEX':
                data *= 1.073E-10 * (np.pi / 3600 / 180)**(-2) / ps[0] / ps[1]
            s[survey + '_PS'] = ps
            s[survey + '_CVL'] = True if survey == 'SPIRE_500' else False
            s[survey + '_RGD'] = True if survey == 'SPIRE_500' else False

        s['RADIUS_KPC'] = self.dp_radius(s)
        s['RADIUS_KPC_RGD'] = True
        # Update DataFrame
        self.df = self.df.append(s)
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

    def dp_radius(self, s):
        """ Calculate the radius at each point """
        l = np.array(s['SPIRE_500'].shape)
        cosPA, sinPA = np.cos(s['PA_RAD']), np.sin(s['PA_RAD'])
        cosINCL = s['cosINCL']
        w = s['SPIRE_500_WCS']
        xcm, ycm = s['RA_RAD'], s['DEC_RAD']
        dp_coords = np.zeros([l[0], l[1], 2])
        # Original coordinate is (y, x)
        # :1 --> x, RA --> the one needed to be divided by cos(incl)
        # :0 --> y, Dec
        dp_coords[:, :, 0], dp_coords[:, :, 1] = \
            np.meshgrid(np.arange(l[1]), np.arange(l[0]))
        # Now, value inside dp_coords is (x, y)
        # :0 --> x, RA --> the one needed to be divided by cos(incl)
        # :1 --> y, Dec
        for i in range(l[0]):
            dp_coords[i] = w.wcs_pix2world(dp_coords[i], 1) * np.pi / 180
        dp_coords[:, :, 0] = 0.5 * (dp_coords[:, :, 0] - xcm) * \
            (np.cos(dp_coords[:, :, 1]) + np.cos(ycm))
        dp_coords[:, :, 1] -= ycm
        # Now, dp_coords is (dx, dy) in the original coordinate
        # cosPA*dy-sinPA*dx is new y
        # cosPA*dx+sinPA*dy is new x
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
                    self.df.set_value(name, survey1, cvl_image)
                    self.df.set_value(name, survey1 + '_UNCMAP', cvl_unc)

            survey_load = 'Gauss_25' if survey1 in ['THINGS', 'HERACLES'] \
                else survey1
            kernel = self.kernels.loc[survey_load, survey2]['KERNEL']
            FWHM1 = self.kernels.loc[survey_load, survey2]['FWHM1']
            FWHM2 = self.kernels.loc[survey_load, survey2]['FWHM2']
            if not cls(ps, self.kernels.loc[survey_load, survey2]['PS'], 2.0):
                ps_old = self.kernels.loc[survey_load, survey2]['PS']
                self.kernels.set_value((survey_load, survey2), 'PS', ps)
                kernel = ric.Kernel_regrid(kernel, ps, ps_old)
                self.kernels.set_value((survey_load, survey2), 'KERNEL',
                                       kernel)

            for name in names:
                map0 = self.df.loc[name][survey1]
                uncmap0 = self.df.loc[name][survey1 + '_UNCMAP']
                if survey1 == 'THINGS':
                    temp_s = pd.read_csv('data/Tables/galaxy_data.csv',
                                         index_col=0).loc[name]
                    bmaj, bmin, bpa = \
                        temp_s['TBMAJ'], temp_s['TBMIN'], temp_s['TBPA']
                    Gkernel = GK1(ps, bpa, bmaj, bmin, 25.0)
                    map0, uncmap0 = \
                        ric.matching_PSF(Gkernel, np.sqrt(bmaj * bmin), 25.0,
                                         map0, uncmap0)
                cvl_image, cvl_unc = \
                    ric.matching_PSF(kernel, FWHM1, FWHM2, map0, uncmap0)
                self.df.set_value(name, survey1, cvl_image)
                self.df.set_value(name, survey1 + '_UNCMAP', cvl_unc)
                self.df.set_value(name, survey1 + '_CVL', True)
            print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def WCS_congrid(self, names, fine_surveys, course_survey, method='linear'):
        names = [names] if type(names) == str else names
        fine_surveys = [fine_surveys] if type(fine_surveys) == str else \
            fine_surveys
        for fine_survey in fine_surveys:
            print('Regridding', len(names), 'galaxies in', fine_survey + '...')
            tic = clock()
            for name in names:
                assert self.df.loc[name][fine_survey + '_CVL']
                rgd_image, rgd_unc = \
                    ric.WCS_congrid(self.df.loc[name][fine_survey],
                                    self.df.loc[name][fine_survey + '_UNCMAP'],
                                    self.df.loc[name][fine_survey + '_WCS'],
                                    self.df.loc[name][course_survey + '_WCS'],
                                    self.df.loc[name][course_survey].shape,
                                    method)
                self.df.set_value(name, fine_survey, rgd_image)
                self.df.set_value(name, fine_survey + '_UNCMAP', rgd_unc)
                self.df.set_value(name, fine_survey + '_RGD', True)
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
                self.df.set_value(name, 'SFR', SFR)
            except ValueError:
                self.df['SFR'] = np.full(len(self.df), np.nan, object)
                self.df.set_value(name, 'SFR', SFR)
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
                self.df.loc[name]['cosINCL'] * 280
            try:
                self.df.set_value(name, 'SMSD', smsd)
            except ValueError:
                self.df['SMSD'] = np.full(len(self.df), np.nan, object)
                self.df.set_value(name, 'SMSD', smsd)
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
                self.df.set_value(name, 'TOTAL_GAS', total_gas)
                self.df.set_value(name, 'TOTAL_GAS_UNCMAP', total_gas_unc)
            except ValueError:
                temp = np.full(len(self.df), np.nan, object)
                self.df['TOTAL_GAS'] = temp
                self.df['TOTAL_GAS_UNCMAP'] = temp
                self.df.set_value(name, 'TOTAL_GAS', total_gas)
                self.df.set_value(name, 'TOTAL_GAS_UNCMAP', total_gas_unc)
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def BDR_cal(self, names):
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
                self.df.set_value(name, 'BDR', lc)
            except ValueError:
                self.df['BDR'] = np.full(len(self.df), np.nan, object)
                self.df.set_value(name, 'BDR', lc)
        print("   --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

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
                    self.BDR_cal(name)
                    lc = self.df.loc[name]['BDR']
                assert self.df.loc[name][survey + '_RGD']
                data = self.df.loc[name][survey]
                self.df.set_value(name, survey,
                                  data[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]])
                if unc:
                    uncmap = self.df.loc[name][survey + '_UNCMAP']
                    self.df.set_value(name, survey + '_UNCMAP',
                                      uncmap[lc[0, 0]:lc[0, 1],
                                             lc[1, 0]:lc[1, 1]])
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")

    def bkg_removal(self, names, surveys, THINGS_Limit=1.0E17):
        names = [names] if type(names) == str else names
        surveys = [surveys] if type(surveys) == str else surveys
        print('Removing background of', len(names), 'galaxies in',
              len(surveys), 'surveys...')
        tic = clock()
        for name in names:
            try:
                diskmask = self.df.loc[name]['DISKMASK']
            except KeyError:
                assert self.df.loc[name]['THINGS_RGD']
                with np.errstate(invalid='ignore'):
                    diskmask = self.df.loc[name]['THINGS'] > THINGS_Limit
                try:
                    self.df.set_value(name, 'DISKMASK', diskmask)
                except ValueError:
                    self.df['DISKMASK'] = np.full(len(self.df), np.nan, object)
                    self.df.set_value(name, 'DISKMASK', diskmask)
            for survey in surveys:
                if survey == 'HERSCHEL_011111':
                    sub_surveys = ['PACS_100', 'PACS_160', 'SPIRE_250',
                                   'SPIRE_350', 'SPIRE_500']
                    lc = self.df.loc[name]['BDR']
                    data = np.zeros([lc[0, 1] - lc[0, 0], lc[1, 1] - lc[1, 0],
                                     5])
                    uncmap = np.empty_like(data)
                    for i in range(len(sub_surveys)):
                        assert self.df.loc[name][sub_surveys[i] + '_RGD']
                        data[:, :, i] = self.df.loc[name][sub_surveys[i]]
                        uncmap[:, :, i] = \
                            self.df.loc[name][sub_surveys[i] + '_UNCMAP']
                        bkg = np.nanmedian(data[~diskmask][i])
                        data[:, :, i] -= bkg
                    n_nanmask = ~np.sum(np.isnan(data), axis=2, dtype=bool)
                    bkgmask = n_nanmask * (~diskmask)
                    bkgcov = np.cov(data[bkgmask].T)
                    try:
                        self.df.set_value(name, 'HERSCHEL_011111', data)
                        self.df.set_value(name, 'HERSCHEL_011111_UNCMAP',
                                          uncmap)
                        self.df.set_value(name, 'HERSCHEL_011111_BKGCOV',
                                          bkgcov)
                        self.df.set_value(name, 'HERSCHEL_011111_DISKMASK',
                                          ~bkgmask)
                    except ValueError:
                        temp = np.full(len(self.df), np.nan, object)
                        self.df['HERSCHEL_011111'] = temp
                        self.df['HERSCHEL_011111_UNCMAP'] = temp
                        self.df['HERSCHEL_011111_BKGCOV'] = temp
                        self.df['HERSCHEL_011111_DISKMASK'] = temp
                        self.df.set_value(name, 'HERSCHEL_011111', data)
                        self.df.set_value(name, 'HERSCHEL_011111_UNCMAP',
                                          uncmap)
                        self.df.set_value(name, 'HERSCHEL_011111_BKGCOV',
                                          bkgcov)
                        self.df.set_value(name, 'HERSCHEL_011111_DISKMASK',
                                          ~bkgmask)
                else:
                    assert self.df.loc[name][survey + '_RGD']
                    data = self.df.loc[name][survey]
                    n_nanmask = ~np.isnan(data)
                    bkgmask = n_nanmask * (~diskmask)
                    bkg = np.median(data[bkgmask]) if np.sum(bkgmask) else 0.0
                    self.df.set_value(name, survey, data - bkg)
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

            with File('output/RGD_data.h5', 'a') as hf:
                grp = hf.create_group(name)
                for survey in surveys:
                    unc = False if survey in no_unc else True
                    grp.create_dataset(survey, data=self.df.loc[name][survey])
                    if unc:
                        grp.create_dataset(survey + '_UNCMAP',
                                           data=self.df.loc[name][survey +
                                                                  '_UNCMAP'])
                    if survey == 'HERSCHEL_011111':
                        grp.create_dataset('HERSCHEL_011111_BKGCOV',
                                           data=self.df.loc[name]
                                           ['HERSCHEL_011111_BKGCOV'])
                        grp.create_dataset('HERSCHEL_011111_DISKMASK',
                                           data=self.df.loc[name]
                                           ['HERSCHEL_011111_DISKMASK'])
            print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")


""" This part will not be working due to major change of MGS """
"""
def metallicity_to_coordinate(name='NGC5457'):
    # Reading & convolving
    cmaps = MGS(name, ['THINGS', 'SPIRE_500'])
    cmaps.add_kernel('Gauss_25', 'SPIRE_500')
    cmaps.matching_PSF_2step(name, 'THINGS', 'Gauss_25', 'SPIRE_500')
    cmaps.WCS_congrid(name, 'THINGS', 'SPIRE_500')
    things = cmaps.df.loc[(name, 'THINGS')].RGD_MAP
    s = cmaps.df.loc[(name, 'SPIRE_500')]
    del cmaps
    # Calculating HI boundary
    axissum = [0] * 2
    lc = np.zeros([2, 2], dtype=int)
    for i in range(2):
        axissum[i] = np.nansum(things, axis=i, dtype=bool)
        for j in range(len(axissum[i])):
            if axissum[i][j]:
                lc[i-1, 0] = j
                break
        lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)
    # Extracting some parameters for calculating radius
    cosPA = np.cos((s.PA) * np.pi / 180)
    sinPA = np.sin((s.PA) * np.pi / 180)
    cosINCL = np.cos(s.INCL * np.pi / 180)
    w, ctr = s.WCS, SkyCoord(s.CMC)
    R25 = gal_data([name]).field('R25_DEG')[0] * (np.pi / 180.)
    # Calculating radius
    df = pd.read_csv('data/Tables/' + name + '_Z_modified.csv')
    coords = np.empty([len(df), 2])
    dp_coords = np.empty([len(df), 2])
    xcm, ycm = ctr.ra.radian, ctr.dec.radian
    for i in range(len(df)):
        ra = df['RAJ2000'].iloc[i].strip().split(' ')
        dec = df['DEJ2000'].iloc[i].strip().split(' ')
        temp1 = ra[0] + 'h' + ra[1] + 'm' + ra[2] + 's ' + dec[0] + 'd' + \
            dec[1] + 'm' + dec[2] + 's'
        temp2 = SkyCoord(temp1)
        coords[i] = np.array([temp2.ra.degree, temp2.dec.degree])
        dp_coords[i] = np.array([temp2.ra.radian, temp2.dec.radian])
    dp_coords[:, 0] = 0.5 * (dp_coords[:, 0] - xcm) * \
        (np.cos(dp_coords[:, 1]) + np.cos(ycm))
    dp_coords[:, 1] -= ycm
    dp_radius = np.sqrt((cosPA * dp_coords[:, 1] +
                        sinPA * dp_coords[:, 0])**2 +
                        ((cosPA * dp_coords[:, 0] -
                          sinPA * dp_coords[:, 1]) / cosINCL)**2) / R25
    # Now, save DEC(y) in axis 0, RA(x) in axis 1.
    coords[:, 0], coords[:, 1] = w.wcs_world2pix(coords[:, 0], coords[:, 1], 1)
    coords[:, 0] -= lc[0, 0]
    coords[:, 1] -= lc[1, 0]
    df['r25'] = dp_radius
    df['new_c1'] = coords[:, 0]
    df['new_c2'] = coords[:, 1]
    df.to_csv('output/' + name + '_metal.csv')
"""
