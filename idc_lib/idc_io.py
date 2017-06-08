import os
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from h5py import File
from time import clock
# from .plot_idchiang import imshowid
from . import idc_regrid as ric
from .gal_data import gal_data

col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value
H2HaHe = 1.36


def read_h5(filename):
    """
    Inputs:
        filename: <str>
         Filename of the file to be read.
    Outputs:
        d: <dict>
         Dictionary with {keys, values} as described in the input file.
    """
    with File('filename', 'r') as hf:
        d = {}
        for key in hf.keys():
            d[key] = np.array(hf.get(key))
    return d


class Surveys(object):
    """ Storaging maps, properties, kernels. """
    def __init__(self, names, surveys, auto_import=True):
        """
        Inputs:
            names: <list of str | str>
              Names of objects to be read.
            surveys: <list of str | str>
              Names of surveys to be read.
            auto_import: <bool>
              Set if we want to import survey maps automatically (default True)
        """
        self.df = pd.DataFrame()
        self.kernels = pd.DataFrame()

        this_dir, this_filename = os.path.split(__file__)
        DATA_PATH = os.path.join(this_dir, "data_table/galaxy_data.csv")
        self.galaxy_data = pd.read_csv(DATA_PATH)
        self.galaxy_data.index = self.galaxy_data.OBJECT.values

        if auto_import:
            print("Importing", len(names) * len(surveys), "fits files...")
            tic = clock()
            self.add_galaxies(names, surveys)
            print("Done. Elapsed time:", round(clock()-tic, 3), "s.")

    def add_galaxies(self, names, surveys, filenames=None):
        """
        Inputs:
            names: <list of str | str>
             Names of objects to be read.
            surveys: <list of str | str>
             Names of surveys to be read.
            filenames: <list of str | str>
                Filenames of files to be read (default None)
        """
        names = [names] if type(names) == str else names
        surveys = [surveys] if type(surveys) == str else surveys
        if filenames:
            filenames = [filenames]
            assert len(names) == len(surveys) == len(filenames), \
                "Input lengths are not equal!!"
            for i in range(len(filenames)):
                print("Warning: BMAJ, BMIN, BPA not supported now!!")
                self.add_galaxy(names[i], surveys[i], filenames[i])
        else:
            for survey in surveys:
                if survey == 'THINGS':
                    self.galaxy_data['BMAJ'] = self.galaxy_data['TBMAJ'].copy()
                    self.galaxy_data['BMIN'] = self.galaxy_data['TBMIN'].copy()
                    self.galaxy_data['BPA'] = self.galaxy_data['TBPA'].copy()
                elif survey == 'HERACLES':
                    self.galaxy_data['BMAJ'] = [13.0] * len(self.galaxy_data)
                    self.galaxy_data['BMIN'] = [13.0] * len(self.galaxy_data)
                    self.galaxy_data['BPA'] = [0.0] * len(self.galaxy_data)
                else:
                    self.galaxy_data['BMAJ'] = [1.0] * len(self.galaxy_data)
                    self.galaxy_data['BMIN'] = [1.0] * len(self.galaxy_data)
                    self.galaxy_data['BPA'] = [0.0] * len(self.galaxy_data)
                for name in names:
                    self.add_galaxy(name, survey)

    def add_galaxy(self, name, survey, filename=None, uncfn=None):
        """
        Inputs:
            name: <str>
             Name of object to be read.
            survey: <str>
             Name of survey to be read.
            filename: <str>
                Filename of file to be read (default None)
        """
        name = name.replace('_', '').replace(' ', '').upper()
        if name == 'UGC05139':
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
        elif name[:3] == 'NGC':
            name1, name2 = 'NGC', name[3:]
            if survey == 'GALEX_FUV' and name2 == '5457':
                filename = 'data/GALEX/ngc5457_clean_FUV.fits'
        elif name[:2] == 'IC':
            name1, name2 = 'IC', name[2:]
        else:
            raise ValueError(name + ' not in database.')
        if survey == 'THINGS':
            name2 = name2.lstrip('0')

        if name1 == 'M81':
            if survey == 'THINGS':
                filename = 'data/THINGS/M81_' + name2.upper() + \
                    '_NA_MOM0_THINGS.FITS'
            elif survey == 'SPIRE_500':
                filename = 'data/SPIRE/M81' + name2 + \
                    '_I_500um_scan_k2011.fits.gz'
            elif survey == 'SPIRE_350':
                filename = 'data/SPIRE/M81' + name2 + \
                    '_I_350um_scan_k2011.fits.gz'
            elif survey == 'SPIRE_250':
                filename = 'data/SPIRE/M81' + name2 + \
                    '_I_250um_scan_k2011.fits.gz'
            elif survey == 'PACS_160':
                filename = 'data/PACS/M81' + name2 + '_I_160um_k2011.fits.gz'
            elif survey == 'PACS_100':
                filename = 'data/PACS/M81' + name2 + '_I_100um_k2011.fits.gz'
            elif survey == 'HERACLES':
                filename = 'data/HERACLES/m81' + name2.lower() + \
                    '_heracles_mom0.fits.gz'
            else:
                raise ValueError(survey + ' not supported.')
        elif name1 == 'DDO' and survey == 'THINGS':
            filename = 'data/THINGS/' + name1 + name2 + '_NA_MOM0_THINGS.FITS'
        elif survey in ['SPIRE_500', 'SPIRE_350', 'SPIRE_250', 'PACS_160',
                        'PACS_100', 'HERACLES']:
            if name1 == 'HO':
                if survey in ['SPIRE_500', 'SPIRE_350', 'SPIRE_250']:
                    name1 = 'Holmberg'
                elif survey in ['PACS_160', 'PACS_100']:
                    name1 = 'HOLMBERG'
                else:
                    name2 = name2.lower()

        if not filename:
            if survey == 'THINGS':
                filename = 'data/THINGS/' + name1 + '_' + name2 + \
                           '_NA_MOM0_THINGS.FITS'
            elif survey == 'SPIRE_500':
                filename = 'data/SPIRE/' + name1 + '_' + name2 + \
                           '_I_500um_scan_k2011.fits.gz'
                uncfn = 'data/SPIRE/' + name1 + '_' + name2 + \
                        '-I-500um_s_unc-k2011.fits'
            elif survey == 'SPIRE_350':
                filename = 'data/SPIRE/' + name1 + '_' + name2 + \
                           '_I_350um_scan_k2011.fits.gz'
                uncfn = 'data/SPIRE/' + name1 + '_' + name2 + \
                        '-I-350um_s_unc-k2011.fits'
            elif survey == 'SPIRE_250':
                filename = 'data/SPIRE/' + name1 + '_' + name2 + \
                           '_I_250um_scan_k2011.fits.gz'
                uncfn = 'data/SPIRE/' + name1 + '_' + name2 + \
                        '-I-250um_s_unc-k2011.fits'
            elif survey == 'PACS_160':
                filename = 'data/PACS/' + name1 + '_' + name2 + \
                           '_I_160um_k2011.fits.gz'
            elif survey == 'PACS_100':
                filename = 'data/PACS/' + name1 + '_' + name2 + \
                           '_I_100um_k2011.fits.gz'
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
            elif survey == 'GALEX_FUV':
                print('I haven"t writen codes for GALEX FUV except M101...')
                return 0
            elif survey == 'MIPS_24':
                filename = 'data/MIPS/' + name1 + '_' + name2 + \
                    '_I_MIPS24_d2009.fits.gz'
            elif survey == 'IRAC_3.6':
                filename = 'data/IRAC/' + name1 + '_' + name2 + \
                    '_I_IRAC_3.6_d2009.fits.gz'
            else:
                raise ValueError(survey + ' not supported.')

        try:
            s = self.galaxy_data.loc[name].copy()
            del s['TBMIN'], s['TBMAJ'], s['TBPA']
            data, hdr = fits.getdata(filename, 0, header=True)

            if uncfn:
                uncdata = fits.getdata(uncfn, 0, header=False)
            elif survey in ['PACS_100', 'PACS_160']:
                uncdata = data[1]
            else:
                uncdata = np.zeros(data.shape)

            if survey == 'THINGS':
                # THINGS: Raw data in JY/B*M/s. Change to
                # column density 1/cm^2
                data = data[0, 0]
                data *= 1.823E18 * 6.07E5 / 1.0E3 / s.BMAJ / s.BMIN
                uncdata = data * 0.05
            elif survey == 'HERACLES':
                # HERACLES: Raw data in K*km/s. Change to
                # surface density M_\odot/pc^2
                # He included
                # This is a calculated parameter by fitting HI to H2 mass
                R21 = 0.7
                data *= R21 * s['ACO']
                uncdata *= R21 * s['ACO']
            elif survey in ['PACS_160', 'PACS_100']:
                data = data[0]
                # print survey + " not supported for density calculation!!"
            elif survey == 'SPIRE_500':
                # In MJy/sr
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
                uncdata += 0.91

            w = wcs.WCS(hdr, naxis=2)
            # add the generated data to dataframe
            s['WCS'], s['MAP'], s['L'] = w, data, np.array(data.shape)
            s['UNCMAP'] = uncdata
            ctr = s.L // 2
            ps = np.zeros(2)
            xs, ys = \
                w.wcs_pix2world([ctr[0] - 1, ctr[0] + 1, ctr[0], ctr[0]],
                                [ctr[1], ctr[1], ctr[1] - 1, ctr[1] + 1],
                                1)
            ps[0] = np.abs(xs[0] - xs[1]) / 2 * np.cos((ys[0] + ys[1]) *
                                                       np.pi / 2 / 180)
            ps[1] = np.abs(ys[3] - ys[2]) / 2
            ps *= u.degree.to(u.arcsec)
            if survey in ['PACS_160', 'PACS_100']:
                # Converting Jy/pixel to MJy/sr
                data *= (np.pi / 36 / 18)**(-2) / ps[0] / ps[1]
            s['PS'] = ps
            s['CVL_MAP'] = np.zeros([1, 1])
            s['CVL_UNC'] = np.zeros([1, 1])
            if survey in ['KINGFISH_DUST']:
                s['CVL_MAP'] = data
                s['CVL_UNC'] = uncdata
            s['RGD_MAP'] = np.zeros([1, 1])
            s['RGD_UNC'] = np.zeros([1, 1])
            s['CAL_MASS'] = 0
            s['DP_RADIUS'] = self.dp_radius(s) if \
                (survey == 'SPIRE_500') else np.zeros([1, 1])
            s['RVR'] = np.zeros([1, 1])
            s['logSFR'] = np.zeros([1, 1])
            """
            if cal:
                print "Calculating " + name + "..."
                s['CAL_MASS'] = self.total_mass(s)
                s['RVR'] = self.Radial_prof(s)
            """
            # Update DataFrame
            self.df = \
                self.df.append(s.to_frame().T.set_index([[name], [survey]]))
        except KeyError:
            print("Warning:", name, "not in csv database!!")
        except IOError:
            print("Warning:", filename, "doesn't exist!!")

    def add_kernel(self, name1s, name2, FWHM1s=[], FWHM2=None, filenames=[]):
        """
        Inputs:
            name1s: <list of str | str>
                Name1's of kernels to be read.
            name2: <str>
                Name2 of kernels to be read.
            FWHM1s: <list of float | float>
                FWHM1's of kernels to be read. (default [])
            FWHM2: <float>
                FWHM2 of kernels to be read. (default None)
            filenames: <str>
                Filenames of files to be read (default [])
        """
        name1s = [name1s] if type(name1s) == str else name1s
        if not filenames:
            for name1 in name1s:
                filenames.append('data/Kernels/Kernel_LoRes_' + name1 +
                                 '_to_' + name2 + '.fits.gz')
        FWHM = {'SPIRE_500': 36.09, 'SPIRE_350': 24.88, 'SPIRE_250': 18.15,
                'Gauss_25': 25, 'PACS_160': 11.18, 'PACS_100': 7.04,
                'HERACLES': 13, 'GALEX_FUV': 4.48, 'MIPS_24': 6.43}
        # Note: pixel scale of SPIRE 500 ~ 14.00
        if not FWHM1s:
            for name1 in name1s:
                FWHM1s.append(FWHM[name1])
        FWHM2 = FWHM[name2] if (not FWHM2) else FWHM2
        assert len(filenames) == len(name1s)
        assert len(FWHM1s) == len(name1s)

        print("Importing", len(name1s), "kernel files...")
        tic = clock()
        for i in range(len(name1s)):
            s = pd.Series()
            try:
                s['KERNEL'], hdr = fits.getdata(filenames[i], 0, header=True)
                s['KERNEL'] /= np.nansum(s['KERNEL'])
                assert hdr['CD1_1'] == hdr['CD2_2']
                assert hdr['CD1_2'] == hdr['CD2_1'] == 0
                assert hdr['CD1_1'] % 2
                s['PS'] = np.array([hdr['CD1_1'] * u.degree.to(u.arcsec),
                                    hdr['CD2_2'] * u.degree.to(u.arcsec)])
                s['FWHM1'], s['FWHM2'], s['NAME1'], s['NAME2'] = \
                    FWHM1s[i], FWHM2, name1s[i], name2
                s['RGDKERNEL'], s['RGDPS'] = np.zeros([1, 1]), np.zeros(2)
                self.kernels = \
                    self.kernels.append(s.to_frame().T.
                                        set_index(['NAME1', 'NAME2']))
            except IOError:
                print("Warning:", filenames[i], "doesn't exist!!")
        print("Done. Elapsed time:", round(clock()-tic, 3), "s.")

    def matching_PSF_1step(self, names, survey1s, survey2):
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
            for name in names:
                cvl_image, cvl_unc, new_ps, new_kernel = \
                    ric.matching_PSF_1step(self.df, self.kernels, name,
                                           survey1, survey2)
                self.df.set_value((name, survey1), 'CVL_MAP', cvl_image)
                self.df.set_value((name, survey1), 'CVL_UNC', cvl_unc)
                self.kernels.set_value((survey1, survey2), 'RGDKERNEL',
                                       new_kernel)
                self.kernels.set_value((survey1, survey2), 'RGDPS',
                                       new_ps)

    def matching_PSF_2step(self, names, survey1s, k2_survey1, k2_survey2):
        """
        Inputs:
            names: <list of str | str>
                Object names to be convolved.
            survey1s: <list of str | str>
                Survey names to be convolved.
            k2_survey1, k2_survey2: <str>
                Names of second kernel.
        """
        names = [names] if type(names) == str else names
        survey1s = [survey1s] if type(survey1s) == str else survey1s
        for survey1 in survey1s:
            for name in names:
                cvl_image, cvl_unc, new_ps, new_kernel = \
                    ric.matching_PSF_2step(self.df, self.kernels, name,
                                           survey1, k2_survey1, k2_survey2)
                self.df.set_value((name, survey1), 'CVL_MAP', cvl_image)
                self.df.set_value((name, survey1), 'CVL_UNC', cvl_unc)
                self.kernels.set_value((k2_survey1, k2_survey2), 'RGDKERNEL',
                                       new_kernel)
                self.kernels.set_value((k2_survey1, k2_survey2), 'RGDPS',
                                       new_ps)

    def WCS_congrid(self, names, fine_surveys, course_survey,
                    method='linear'):
        """
        Inputs:
            names, fine_surveys: <list of str | str>
                Object names and fine survey names to be regrdidded.
            course_survey: <str>
                Course survey name to be regridded.
            method: <str>
                Fitting method. 'linear', 'nearest', 'cubic'
        """
        names = [names] if type(names) == str else names
        fine_surveys = [fine_surveys] if type(fine_surveys) == str else \
            fine_surveys
        for fine_survey in fine_surveys:
            for name in names:
                rgd_image, rgd_unc = ric.WCS_congrid(self.df, name,
                                                     fine_survey,
                                                     course_survey, method)
                self.df.set_value((name, fine_survey), 'RGD_MAP', rgd_image)
                self.df.set_value((name, fine_survey), 'RGD_UNC', rgd_unc)

    def dp_radius(self, s):
        """
        Inputs:
            s: <pandas.Series>
                Series containting all information needed.
        """
        l = s.L
        cosPA = np.cos((s.PA) * np.pi / 180)
        sinPA = np.sin((s.PA) * np.pi / 180)
        cosINCL = np.cos(s.INCL * np.pi / 180)
        w = s.WCS
        ctr = SkyCoord(s.CMC)
        xcm, ycm = ctr.ra.radian, ctr.dec.radian
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
            s.D * 1.0E3  # Radius in kpc

    def SFR_FUV_plus_24(self, names):
        names = [names] if type(names) == str else names
        logCx = 43.35
        for name in names:
            print('Calculating ', name, ' SFR...')
            fuv = self.df.loc[(name, 'GALEX_FUV')].RGD_MAP
            mips24 = self.df.loc[(name, 'MIPS_24')].RGD_MAP
            ps = self.df.loc[(name, 'SPIRE_500')].PS
            nanmask1, nanmask2 = np.isnan(fuv), np.isnan(mips24)
            fuv[nanmask1], mips24[nanmask2] = 0.0, 0.0
            nanmask = nanmask1 * nanmask1
            logfuv_corr = np.log10(fuv + 3.89 * mips24)
            logfuv_corr[nanmask] = np.nan
            """ fuv in MJy / sr """
            logfuv_corr += 6  # Now in Jy / sr
            logfuv_corr += np.log10(ps[0] * ps[1] / (np.pi / 3600 / 180)**(-2))
            # Now in Jy, or (ergs s-1) / Hz / cm2 / 10^23
            # GALEX main lense diameter = 50cm
            logfuv_corr += np.log10(1934144E9) + np.log10(np.pi * 25**2) + 23
            # Lx in ergs s-1
            logSFR = logfuv_corr - logCx
            self.df.set_value((name, 'GALEX_FUV'), 'logSFR', logSFR)

    def save_data(self, names):
        """
        Inputs:
            names: <list of str | str>
                Object names to be saved.
        """
        names = [names] if type(names) == str else names
        for name in names:
            print('Saving', name, 'data...')
            things = self.df.loc[(name, 'THINGS')].RGD_MAP
            things_unc = self.df.loc[(name, 'THINGS')].RGD_UNC
            # Cutting off the nan region of THINGS map.
            # [lc[0,0]:lc[0,1],lc[1,0]:lc[1,1]]
            axissum = [0] * 2
            lc = np.zeros([2, 2], dtype=int)
            for i in range(2):
                axissum[i] = np.nansum(things, axis=i, dtype=bool)
                for j in range(len(axissum[i])):
                    if axissum[i][j]:
                        lc[i-1, 0] = j
                        break
                lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)

            sed = np.zeros([things.shape[0], things.shape[1], 5])
            sed_unc = np.zeros([things.shape[0], things.shape[1], 5])

            heracles = self.df.loc[(name, 'HERACLES')].RGD_MAP
            kingfish = self.df.loc[(name, 'KINGFISH_DUST')].RGD_MAP
            sed[:, :, 0] = self.df.loc[(name, 'PACS_100')].RGD_MAP
            sed[:, :, 1] = self.df.loc[(name, 'PACS_160')].RGD_MAP
            sed[:, :, 2] = self.df.loc[(name, 'SPIRE_250')].RGD_MAP
            sed[:, :, 3] = self.df.loc[(name, 'SPIRE_350')].RGD_MAP
            sed[:, :, 4] = self.df.loc[(name, 'SPIRE_500')].MAP
            heracles_unc = self.df.loc[(name, 'HERACLES')].RGD_UNC
            kingfish_unc = self.df.loc[(name, 'KINGFISH_DUST')].RGD_UNC
            sed_unc[:, :, 0] = self.df.loc[(name, 'PACS_100')].RGD_UNC
            sed_unc[:, :, 1] = self.df.loc[(name, 'PACS_160')].RGD_UNC
            sed_unc[:, :, 2] = self.df.loc[(name, 'SPIRE_250')].RGD_UNC
            sed_unc[:, :, 3] = self.df.loc[(name, 'SPIRE_350')].RGD_UNC
            sed_unc[:, :, 4] = self.df.loc[(name, 'SPIRE_500')].UNCMAP

            dp_radius = self.df.loc[(name, 'SPIRE_500')].DP_RADIUS

            # Using the variance of non-galaxy region as uncertainty
            nanmask = ~np.sum(np.isnan(sed), axis=2, dtype=bool)
            bkgcov = None
            THINGS_Limit = 1.0E17
            while(bkgcov is None):
                THINGS_Limit *= 10
                with np.errstate(invalid='ignore'):
                    glxmask = (things > THINGS_Limit)
                diskmask = glxmask * nanmask
                # Covariance matrix begins
                bkgmask = (~glxmask) * nanmask
                N = np.sum(bkgmask)
                if N > 100:
                    print("Available bkg pixels:", N)
                    bkgcov = np.cov(sed[bkgmask].T)
                # Covariance matrix ends

            for i in range(5):
                temp = sed[:, :, i][bkgmask]
                sed[:, :, i] -= np.median(temp)

            # Cut the images and masks!!!
            things = things[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            things_unc = things_unc[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            heracles = heracles[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            heracles_unc = heracles_unc[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            # To avoid np.nan in H2 + signal in HI
            mask = np.isnan(heracles)
            heracles_unc[mask] = 0
            heracles[mask] = 0
            total_gas = col2sur * H2HaHe * things + heracles
            total_gas_unc = col2sur * H2HaHe * things_unc + heracles_unc
            heracles_unc[mask] = np.nan
            heracles[mask] = np.nan
            kingfish = kingfish[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            kingfish_unc = \
                kingfish_unc[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            sed = sed[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1], :]
            sed_unc = sed_unc[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1], :]
            diskmask = diskmask[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            dp_radius = dp_radius[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            assert diskmask.shape == dp_radius.shape
            total_gas[~diskmask] = np.nan
            sed[~diskmask] = np.nan
            dp_radius[~diskmask] = np.nan

            # Create some parameters for calculating radial distribution
            ctr = SkyCoord(self.df.loc[name].CMC[0])
            xcm, ycm = ctr.ra.degree, ctr.dec.degree
            w = self.df.loc[(name, 'SPIRE_500')].WCS
            x_ctr, y_ctr = w.wcs_world2pix(xcm, ycm, 1)
            glx_ctr = np.array([y_ctr - lc[0, 0], x_ctr - lc[1, 0]])
            with File('output/RGD_data.h5', 'a') as hf:
                grp = hf.create_group(name)
                grp.create_dataset('Total_gas', data=total_gas)
                grp.create_dataset('Total_gas_unc', data=total_gas_unc)
                grp.create_dataset('THINGS', data=things)
                grp.create_dataset('THINGS_unc', data=things_unc)
                grp.create_dataset('HERACLES', data=heracles)
                grp.create_dataset('HERACLES_unc', data=heracles_unc)
                grp.create_dataset('KINGFISH', data=kingfish)
                grp.create_dataset('KINGFISH_unc', data=kingfish_unc)
                grp.create_dataset('Herschel_SED', data=sed)
                grp.create_dataset('Herschel_SED_unc', data=sed_unc)
                grp.create_dataset('Herschel_bkgcov', data=bkgcov)
                grp.create_dataset('Herschel_bkg_N', data=N)
                grp.create_dataset('Diskmask', data=diskmask)
                grp.create_dataset('Galaxy_center', data=glx_ctr)
                grp.create_dataset('Galaxy_distance',
                                   data=self.df.loc[name].D[0])
                grp.create_dataset('INCL', data=self.df.loc[name].INCL[0])
                grp.create_dataset('PA', data=self.df.loc[name].PA[0])
                grp.create_dataset('PS',
                                   data=self.df.loc[(name, 'SPIRE_500')].PS)
                grp.create_dataset('THINGS_LIMIT', data=THINGS_Limit)
                grp.create_dataset('DP_RADIUS', data=dp_radius)  # kpc
            """
            plt.figure()
            plt.subplot(2, 4, 1)
            imshowid(np.log10(total_gas))
            plt.title('Total gas (log); TL = ' + str(THINGS_Limit))
            for i in range(5):
                plt.subplot(2, 4, i + 2)
                imshowid(sed[:, :, i])
                plt.title('Herschel SED: ' + str(i))
            plt.subplot(2, 4, 7)
            imshowid(diskmask)
            plt.title('Diskmask')
            plt.suptitle(name)
            plt.savefig('output/RGD_data/' + name + '.png')
            plt.clf()
            plt.close()
            """
        print('All data saved.')

    def save_SFR(self, names):
        """
        Inputs:
            names: <list of str | str>
                Object names to be saved.
        """
        names = [names] if type(names) == str else names
        for name in names:
            print('Saving', name, 'data...')
            things = self.df.loc[(name, 'THINGS')].RGD_MAP
            logsfr = self.df.loc[(name, 'GALEX_FUV')].logSFR
            # Cutting off the nan region of THINGS map.
            # [lc[0,0]:lc[0,1],lc[1,0]:lc[1,1]]
            axissum = [0] * 2
            lc = np.zeros([2, 2], dtype=int)
            for i in range(2):
                axissum[i] = np.nansum(things, axis=i, dtype=bool)
                for j in range(len(axissum[i])):
                    if axissum[i][j]:
                        lc[i-1, 0] = j
                        break
                lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)

            # Cut the images and masks!!!
            logsfr = logsfr[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]

            with File('output/RGD_data.h5', 'a') as hf:
                grp = hf[name]
                grp.create_dataset('logSFR', data=logsfr)
        print('All data saved.')


def metallicity_to_coordinate(name='NGC5457'):
    """ Reading & convolving """
    cmaps = Surveys(name, ['THINGS', 'SPIRE_500'])
    cmaps.add_kernel('Gauss_25', 'SPIRE_500')
    cmaps.matching_PSF_2step(name, 'THINGS', 'Gauss_25', 'SPIRE_500')
    cmaps.WCS_congrid(name, 'THINGS', 'SPIRE_500')
    things = cmaps.df.loc[(name, 'THINGS')].RGD_MAP
    s = cmaps.df.loc[(name, 'SPIRE_500')]
    del cmaps
    """ Calculating HI boundary """
    axissum = [0] * 2
    lc = np.zeros([2, 2], dtype=int)
    for i in range(2):
        axissum[i] = np.nansum(things, axis=i, dtype=bool)
        for j in range(len(axissum[i])):
            if axissum[i][j]:
                lc[i-1, 0] = j
                break
        lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)
    """ Extracting some parameters for calculating radius """
    cosPA = np.cos((s.PA) * np.pi / 180)
    sinPA = np.sin((s.PA) * np.pi / 180)
    cosINCL = np.cos(s.INCL * np.pi / 180)
    w, ctr = s.WCS, SkyCoord(s.CMC)
    R25 = gal_data.gal_data([name]).field('R25_DEG')[0] * (np.pi / 180.)
    """ Calculating radius """
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
