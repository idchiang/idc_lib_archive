from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from h5py import File
from time import clock
from .plot_idchiang import imshowid
from . import regrid_idchiang as ric

col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value

"""
HOI UGC05139
HOII UGC04305
M81DWA PGC023521
M81DWB UGC05423
DDO053 UGC04459
DDO154 NGC4789A
"""

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
    def __init__(self, names, surveys, XCO_multiplier=1.0, auto_import=True):
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
        self.XCO_multiplier = XCO_multiplier
        
        if auto_import:
            print("Importing", len(names) * len(surveys), "fits files...")        
            tic = clock()
            self.add_galaxies(names, surveys, XCO_multiplier)
            print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
                
    def add_galaxies(self, names, surveys, XCO_multiplier=1.0, filenames=None):
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
                self.add_galaxy(names[i], surveys[i], XCO_multiplier, 
                                filenames[i])
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
                    self.add_galaxy(name, survey, XCO_multiplier)
		
    def add_galaxy(self, name, survey, XCO_multiplier=1.0, filename=None):
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
        elif name[:2] == 'IC':
            name1, name2 = 'IC', name[2:]
        else:
            raise ValueError(name + ' not in database.')
        if survey == 'THINGS':
            name2 = name2.lstrip('0')

        if name1 == 'M81':
            if survey == 'THINGS':
                filename = 'data/THINGS/M81_' + name2.upper() + '_NA_MOM0_THINGS.FITS'
            elif survey == 'SPIRE_500':
                filename = 'data/SPIRE/M81' + name2 + '_I_500um_scan_k2011.fits.gz'
            elif survey == 'SPIRE_350':
                filename = 'data/SPIRE/M81' + name2 + '_I_350um_scan_k2011.fits.gz'
            elif survey == 'SPIRE_250':
                filename = 'data/SPIRE/M81' + name2 + '_I_250um_scan_k2011.fits.gz'
            elif survey == 'PACS_160':
                filename = 'data/PACS/M81' + name2 + '_I_160um_k2011.fits.gz'
            elif survey == 'PACS_100':
                filename = 'data/PACS/M81' + name2 + '_I_100um_k2011.fits.gz'
            elif survey == 'HERACLES': 
                filename = 'data/HERACLES/m81' + name2.lower() + '_heracles_mom0.fits.gz'
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
            elif survey == 'SPIRE_350':
                filename = 'data/SPIRE/' + name1 + '_' + name2 + \
                           '_I_350um_scan_k2011.fits.gz'
            elif survey == 'SPIRE_250':
                filename = 'data/SPIRE/' + name1 + '_' + name2 + \
                           '_I_250um_scan_k2011.fits.gz'
            elif survey == 'PACS_160':
                filename = 'data/PACS/' + name1 + '_' + name2 + \
                           '_I_160um_k2011.fits.gz'
            elif survey == 'PACS_100':
                filename = 'data/PACS/' + name1 + '_' + name2 + \
                           '_I_100um_k2011.fits.gz'
            elif survey == 'HERACLES': 
                filename = 'data/HERACLES/' + name1.lower() + name2 + \
                           '_heracles_mom0.fits.gz'
            else:
                raise ValueError(survey + ' not supported.')

        try:
            s = self.galaxy_data.loc[name].copy()
            del s['TBMIN'], s['TBMAJ'], s['TBPA']
            data, hdr = fits.getdata(filename, 0, header=True)

            if survey == 'THINGS':
                # THINGS: Raw data in JY/B*M/s. Change to
                # column density 1/cm^2    
                data = data[0, 0]
                data *= 1.823E18 * 6.07E5 / 1.0E3 / s.BMAJ / s.BMIN                
            elif survey == 'HERACLES':
                # HERACLES: Raw data in K*km/s. Change to
                # column density 1/cm^2
                # This is a calculated parameter by fitting HI to H2 mass
                R21 = 0.8
                XCO = 2.0E20 * XCO_multiplier
                data *= XCO * (R21 / 0.8)
            else:
                if survey in ['PACS_160', 'PACS_100']:
                    data = data[0]
                # print survey + " not supported for density calculation!!"

            w = wcs.WCS(hdr, naxis=2)
            # add the generated data to dataframe
            s['WCS'], s['MAP'], s['L'] = w, data, np.array(data.shape)
            ctr = s.L // 2
            ps = np.zeros(2)
            xs, ys = \
                w.wcs_pix2world([ctr[0] - 1, ctr[0] + 1, ctr[0], ctr[0]],
                                [ctr[1], ctr[1], ctr[1] - 1, ctr[1] + 1], 
                                1)
            ps[0] = np.abs(xs[0] - xs[1]) / 2 * \
                    np.cos((ys[0] + ys[1]) * np.pi / 2 / 180)
            ps[1] = np.abs(ys[3] - ys[2]) / 2
            ps *= u.degree.to(u.arcsec)
            if survey in ['PACS_160', 'PACS_100']:
                # Converting Jy/pixel to MJy/sr
                data *= (np.pi / 36 / 18)**(-2) / ps[0] / ps[1]
            s['PS'] = ps
            s['CVL_MAP'] = np.zeros([1, 1])
            s['RGD_MAP'] = np.zeros([1, 1])
            s['CAL_MASS'] = 0
            s['DP_RADIUS'] = self.dp_radius(s) if \
                (survey == 'SPIRE_500') else np.zeros([1, 1])
            s['RVR'] = np.zeros([1, 1])
            """
            if cal:
                print "Calculating " + name + "..."
                s['CAL_MASS'] = self.total_mass(s)
                s['RVR'] = self.Radial_prof(s)
            """
            # Update DataFrame
            self.df = \
                self.df.append(s.to_frame().T.set_index([[name],[survey]]))
        except KeyError:
            print("Warning:", name, "not in csv database!!")
        except IOError:
            print("Warning:", filename ,"doesn't exist!!")
				
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
                'HERACLES': 13}
        if not FWHM1s:
            for name1 in name1s:
                FWHM1s.append(FWHM[name1])
        FWHM2 = FWHM[name2] if (not FWHM2) else FWHM2
        print(filenames, len(filenames))
        print(name1s, len(name1s))
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
                                        set_index(['NAME1','NAME2']))
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
                cvl_image, new_ps, new_kernel = \
                    ric.matching_PSF_1step(self.df, self.kernels, name, 
                                           survey1, survey2)
                self.df.set_value((name, survey1), 'CVL_MAP', cvl_image)
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
                cvl_image, new_ps, new_kernel = \
                    ric.matching_PSF_2step(self.df, self.kernels, name, 
                                           survey1, k2_survey1, k2_survey2)
                self.df.set_value((name, survey1), 'CVL_MAP', cvl_image)
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
        fine_surveys = [fine_surveys] if type(fine_surveys) == str \
                       else fine_surveys
        for fine_survey in fine_surveys:
            for name in names:
                rgd_image = ric.WCS_congrid(self.df, name, fine_survey, 
                                            course_survey, method)
                self.df.set_value((name, fine_survey), 'RGD_MAP', rgd_image)

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
        for i in xrange(l[0]):
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
                   s.D * 1.0E3 # Radius in kpc

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
            # Cutting off the nan region of THINGS map.
            # [lc[0,0]:lc[0,1],lc[1,0]:lc[1,1]]
            axissum = [0] * 2
            lc = np.zeros([2,2], dtype=int)
            for i in xrange(2):
                axissum[i] = np.nansum(things, axis=i, dtype=bool)
                for j in xrange(len(axissum[i])):
                    if axissum[i][j]:
                        lc[i-1, 0] = j
                        break
                lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)

            sed = np.zeros([things.shape[0], things.shape[1], 5])            
            heracles = self.df.loc[(name, 'HERACLES')].RGD_MAP
            sed[:, :, 0] = self.df.loc[(name, 'PACS_100')].RGD_MAP
            sed[:, :, 1] = self.df.loc[(name, 'PACS_160')].RGD_MAP
            sed[:, :, 2] = self.df.loc[(name, 'SPIRE_250')].RGD_MAP
            sed[:, :, 3] = self.df.loc[(name, 'SPIRE_350')].RGD_MAP
            sed[:, :, 4] = self.df.loc[(name, 'SPIRE_500')].MAP
            dp_radius = self.df.loc[(name, 'SPIRE_500')].DP_RADIUS
                
            # Using the variance of non-galaxy region as uncertainty
            nanmask = ~np.sum(np.isnan(sed), axis=2, dtype=bool)
            bkgerr = np.full(5, np.nan)
            THINGS_Limit = 1.0E17
            while(np.sum(np.isnan(bkgerr))):
                THINGS_Limit *= 10
                temp = []
                glxmask = (things > THINGS_Limit)
                diskmask = glxmask * nanmask
                for i in range(5):
                    inv_glxmask2 = ~(np.isnan(sed[:, :, i]) + glxmask)
                    temp.append(sed[inv_glxmask2, i])
                    temp[i] = temp[i][np.abs(temp[i]) < (3 * np.std(temp[i]))]
                    bkgerr[i] = np.std(temp[i])

            for i in range(5):                
                sed[:, :, i] -= np.median(temp[i])
                
            # Cut the images and masks!!!
            things = things[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            heracles = heracles[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            # To avoid np.nan in H2 + signal in HI
            heracles[np.isnan(heracles)] = 0 
            total_gas = col2sur * (2 * heracles + things)
            sed = sed[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1], :]
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
                grp.create_dataset('THINGS', data=things)
                grp.create_dataset('HERACLES', data=heracles)
                grp.create_dataset('Herschel_SED', data=sed)
                grp.create_dataset('Herschel_bkgerr', data=bkgerr)
                grp.create_dataset('Diskmask', data=diskmask)
                grp.create_dataset('Galaxy_center', data=glx_ctr)
                grp.create_dataset('Galaxy_distance', 
                                   data=self.df.loc[name].D[0])
                grp.create_dataset('INCL', data=self.df.loc[name].INCL[0])
                grp.create_dataset('PA', data=self.df.loc[name].PA[0])
                grp.create_dataset('PS', 
                                   data=self.df.loc[(name, 'SPIRE_500')].PS)
                grp.create_dataset('THINGS_LIMIT', data=THINGS_Limit)
                grp.create_dataset('DP_RADIUS', data=dp_radius) # kpc
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
        print('All data saved.')
        
    def save_new_XCO(self, names):    
        """
        Inputs:
            names: <list of str | str>
                Object names to be saved.
        """
        names = [names] if type(names) == str else names
        for name in names:
            print('Saving', name, 'data...')
            things = self.df.loc[(name, 'THINGS')].RGD_MAP
            heracles = self.df.loc[(name, 'HERACLES')].RGD_MAP
            # Cutting off the nan region of THINGS map.
            # [lc[0,0]:lc[0,1],lc[1,0]:lc[1,1]]
            axissum = [0] * 2
            lc = np.zeros([2,2], dtype=int)
            for i in xrange(2):
                axissum[i] = np.nansum(things, axis=i, dtype=bool)
                for j in xrange(len(axissum[i])):
                    if axissum[i][j]:
                        lc[i-1, 0] = j
                        break
                lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)

            # Cut the images and masks!!!
            things = things[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            heracles = heracles[lc[0, 0]:lc[0, 1], lc[1, 0]:lc[1, 1]]
            # To avoid np.nan in H2 + signal in HI
            heracles[np.isnan(heracles)] = 0 
            total_gas = col2sur * (2 * heracles + things)

            # Create some parameters for calculating radial distribution
            with File('output/RGD_data.h5', 'a') as hf:
                grp = hf[name]
                diskmask = np.array(grp['Diskmask'])
                total_gas[~diskmask] = np.nan
                grp.create_dataset('Total_gas_XCOM=' + 
                                   str(round(self.XCO_multiplier, 2)), 
                                   data=total_gas)
        print('All data saved.')