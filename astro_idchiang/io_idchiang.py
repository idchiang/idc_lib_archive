from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
import os
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from h5py import File
from time import clock
from .dustfit_idchiang import fit_dust_density as fdd
from . import regrid_idchiang as ric

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
		
    def add_galaxy(self, name, survey, filename=None):
        """   
        Inputs:
            name: <str>
		        Name of object to be read.
            survey: <str>
		        Name of survey to be read.
            filename: <str>
                Filename of file to be read (default None)
        """
        continuing = True
        
        for i in range(len(name)):
            if name[i] in list(' _0'):
                name1 = name[:i]
                name2 = name[i+1:]
                break
            elif name[i] in list('123456789'):
                name1 = name[:i]
                name2 = name[i:]
                break
        name = name1.upper() + ' ' + name2

        if survey in ['SPIRE_500', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 
                      'PACS_100', 'HERACLES']:
            if name1.upper() == 'NGC' and len(name2) == 3:
                name2 = '0' + name2

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
                continuing = False
                print("Warning: Survey not supported yet!!", 
                      "Please check or enter file name directly.")

        if continuing:
            try:
                s = self.galaxy_data.loc[name].copy()
                del s['TBMIN'], s['TBMAJ'], s['TBPA']
                data, hdr = fits.getdata(filename, 0, header=True)

                if survey == 'THINGS':
                    # THINGS: Raw data in JY/B*M. Change to
                    # column density 1/cm^2    
                    data = data[0, 0]
                    data *= 1.823E18 * 6.07E5 / 1.0E3 / s.BMAJ / s.BMIN                
                elif survey == 'HERACLES':
                    # HERACLES: Raw data in K*km/s. Change to
                    # column density 1/cm^2
                    # This is a calculated parameter by fitting HI to H2 mass
                    R21 = 0.8
                    XCO = 2.0E20
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
                s['CAL_MASS'] = np.zeros([1, 1])
                s['DP_RADIUS'] = np.zeros([1, 1])
                s['RVR'] = np.zeros([1, 1])
                """
                if cal:
                    print "Calculating " + name + "..."
                    s['CAL_MASS'] = self.total_mass(s)
                    s['DP_RADIUS'] = self.dp_radius(s)
                    s['RVR'] = self.Radial_prof(s)
                else:
                    s['CAL_MASS'] = np.zeros([1,1])
                    s['DP_RADIUS'] = np.zeros([1,1])
                    s['RVR'] = np.zeros([1,1])
                """
                # Update DataFrame
                self.df = \
                    self.df.append(s.to_frame().T.set_index([[name],[survey]]))
            except KeyError:
                print("Warning:", name, "not in", self.name ,"csv database!!")
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
                
    def fit_dust_density(self, names, nwalkers=10, nsteps=200):
        """
        Inputs:
            names: <list of str | str>
                Object names to be calculated.
            nwalkers: <int>
                Number of 'walkers' in the mcmc algorithm
            nsteps: <int>
                Number of steps in the mcm algorithm
        Outputs:
        """
        names = [names] if type(names) == str else names
        for name in names:
            fdd(self.df, name, nwalkers, nsteps)