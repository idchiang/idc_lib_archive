"""
NAME:
    IDC_astro.py

PURPOSE:
    1. Calculate total flux and radial profile of a survey
    2. Convolve, matching, regridding, and cutting maps from different surveys

CLASSES:
    Compare_images(object)
        Class designed for comparing maps of the same object from different surveys.
    Galaxies_M0(object)
        Class designed for manipulating maps from the same survey

FUNCTIONS:
    Gaussian_rot(x_in, y_in, x_c, y_c, bpa, bmaj, bmin, ps)
        Generate a elliptical, rotated Gaussian PSF.
    Gaussian_Kernel_C1(ps, bpa, bmaj, bmin, FWHM=25.0):
        Generate Kernel Case 1: from a elliptical, rotated Gaussian to a normal Gaussian.

VARIABLES:
    THINGS_objects : 'list of string'
        List contains all onject names in THINGS.
    FWHM : 'dictionary'
        Dictionary contains FWHM of each survey in arcsec.
    FWHM2sigma : 'float'
        Conversion factor for FWHM to sigma of Gaussian

EXAMPLEs:

    >>> cmaps.plot_RGD('NGC 3198', 'THINGS', 'SPIRE_500')

    # Adding fits files from a survey and calculate their properties.    
    >>> THINGS = Galaxies_M0('THINGS')
    >>> THINGS.add_galaxies(THINGS_objects, cal = True)
    >>> THINGS.plot_rp_all()
    >>> THINGS.plot_map_rp(THINGS_objects[0])
    >>> THINGS.mass_compare()

MODIFICATION HISTORY:
    06/30/2016, ver. 0.0
        -- File created.
    07/01/2016, ver. 1.0
        -- Radial profile and mass calculation finished.
    07/06/2016, ver. 1.0.1
        -- Fixing mass calculation: including cos(Dec) correction in area.
    07/20/2016, ver. 1.1
        -- Adding matching WCS regridding function and cutting images
    07/27/2016, ver. 1.2
        -- Adding convolution, importing/creating kernel
        -- Multi-galaxies comparison possible
        -- Restructuring the entire code: comments and debugging
        -- Unify method for calculating pixel scale
    08/03/2016, ver. 1.2.1
        -- Fixing FWHM --> sigma errors

CURRENT TO DO:
    1. Dust surface mass density fitting
    2. Method: generate kernel, convolution, regrid for all surveys
    3. Combine data files

FUTURE TO DO:
    1. Normalization of the regrid map if already in "density"
    2. add_kernel: checking centered
"""

# Future functions
from __future__ import division
# Python built in libraries
import time
# Astropy functions
import astropy.units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.constants import M_sun, m_p, c, h, k_B
from astropy.convolution import convolve_fft
from astropy.io import fits
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d, griddata
from scipy.optimize import minimize
import emcee
import h5py

    ######################################
    ######    Working here now!!    ######
    ######################################   

"""
with h5py.File('dust_data.h5', 'a') as hf:
    p = []
    for key in hf.keys():
        p.append([key, np.array(hf.get(key))])
"""


# Defined global constants
THINGS_objects = ['NGC 628', 'NGC 925', 'NGC 1569', 'NGC 2366', 'NGC 2403', 
                  'Ho II', 'M81 DwA', 'DDO53', 'NGC 2841', 'NGC 2903', 
                  'Ho I', 'NGC 2976', 'NGC 3031', 'NGC 3077', 'M81 DwB',
                  'NGC 3184', 'NGC 3198', 'IC 2574', 'NGC 3351', 'NGC 3521',
                  'NGC 3621', 'NGC 3627', 'NGC 4214', 'NGC 4449', 'NGC 4736',
                  'DDO154', 'NGC 4826', 'NGC 5055', 'NGC 5194', 'NGC 5236',
                  'NGC 5457', 'NGC 6946', 'NGC 7331', 'NGC 7793']
FWHM = {'SPIRE_500': 36.09, 'SPIRE_350': 24.88, 'SPIRE_250': 18.15, 'Gauss_25': 25,\
        'PACS_160': 11.18, 'PACS_100': 7.04, 'HERACLES': 13}
# ref Aniano 2011, Leroy 2009
FWHM2sigma = 0.5 / np.sqrt(2*np.log(2))

class Compare_images(object):
    """
    Class designed for comparing maps of the same object from different surveys.
    Should include methods like:
    1. Importing fits map files
    2. Import fits Kernels files
    3. Import fits PSF files
    4. Building PSF/Kernels
    5. Kernel convolution
    6. Matching WCS and regrid
    7. Cut regridded images
    
    Parameters
    ----------
    names : list of str
        Galaxy names that are included in this comparison. Must be galaxies in
        the surveys listed in *.
    surveys : list of str
        The surveys included in this comparison. Must be surveys listed in *.
    * 'THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100'
        
    Attributes
    ----------
    names : `list of str`
        Galaxy names that are included in this comparison.
    surveys : 'list of str'
        The surveys included in this comparison.
    num_surveys : 'int'
        Number of surveys included.
    df : 'pandas.DataFrame'
        DataFrame that stores every fits maps and WCS information in this object
            'index': Double index, object name and survey name.
            'BMAJ', 'BMIN': Major axis and minor axis of sythesized beam / PSF in arcsec
            'BPA': PA of Beam / PSF in degrees
            'L': shape of the image
            'MAP': 2-dimensional array of image data
            'WCS': WCS information
            'CVL_MAP': Convovled map
            'RGD_MAP': Regridded map
    kernels : 'pandas.DataFrame'
        DataFrame that stores every fits kernels in this object
        
    Methods
    ----------
    add_galaxy(self, name, survey)
        Add galaxy fits file to self.df
    add_kernel(self, name1, name2, FWHM1 = None, FWHM2 = None, filename = None)
        Add kernel fits file to self.kernels
    Kernel_regrid(self, name1, name2, target_name = None, method = 'cubic')
        Regrid kernel to match with the pixel scale of our image
    matching_PSF_2step(self, name1, survey1, k2_name1, k2_name2, plotting = True, \
                       saving = False)
        Using 2 steps to match PSFs of two images.
    WCS_congrid(self, name, fine_survey, course_survey, method = 'linear')
        Matching the WCS and pixel scale of two images.
    plot_RGD(self, name, fine_survey, course_survey)
        Plot original, convolved, regrid images of the fine survey

    Notes
    ----------
    """
    def __init__(self, names = [], surveys = []):
        self.names = names
        self.surveys = surveys
        self.num_surveys = len(surveys)
        self.df = pd.DataFrame()
        self.kernels = pd.DataFrame()

        print "Importing " + str(len(names)*len(surveys)) + " fits files..."        
        tic = time.clock()
        for name in names:
            for survey in surveys:
                self.add_galaxy(name, survey)
        # self.df.index.names = ['Name', 'Survey']
        toc = time.clock()
        print "Done. Elapsed time: " + str(round(toc-tic, 3)) + " s."
                
    def add_galaxy(self, name, survey):
        """
        Add galaxy fits file to self.df
    
        Parameters
        ----------
        name : 'str'
            Galaxy name.
        survey : 'str'
            Survey name.
        """
        galaxy = Galaxies_M0(survey)
        galaxy.add_galaxy(name, cal = False)
        self.df = self.df.append(galaxy.df[['BMAJ', 'BMIN', 'BPA', 'L', 'MAP',\
                  'WCS', 'PS', 'CVL_MAP', 'RGD_MAP']]\
                  .set_index([[name],[survey]]))        

    def add_kernel(self, name1s, name2, FWHM1s = [], FWHM2 = [], \
                   filenames = []):
        """
        Add kernel fits file to self.kernels
    
        Parameters
        ----------
        name1s : 'list of str' or 'str'
            Kernel name of the incoming PSF.        
        name2 : 'str'
            Kernel name of the outgoing PSF.
        FWHM1 / FWHM2 : 'list of float' or 'float'
            FHWM of the incoming / outgoing PSF (in arcsec). Will be acquired from database
            if not entered.
        filename : 'list of str' or 'str', optional
            File name of the Kernel. Will be generated to Anaiano format if not entered.
            
        Notes
        ---------
        Currently, it only recognize pixel scale in 'CD1_1', 'CD2_2' without rotation.
        """
        if type(name1s) == str:        
            name1s = [name1s]
        if not filenames:
            for name1 in name1s:
                filenames.append('Kernels/Kernel_LoRes_' + name1 + '_to_' + \
                                 name2 + '.fits.gz')
        if not FWHM1s:
            for name1 in name1s:
                FWHM1s.append(FWHM[name1])
        if not FWHM2:
            FWHM2 = FWHM[name2]
        assert len(filenames) == len(name1s)
        assert len(FWHM1s) == len(name1s)            

        print "Importing " + str(len(name1s)) + " kernel files..."
        tic = time.clock()        
        for i in xrange(len(name1s)):
            s = pd.Series()
            try:
                s['KERNEL'], hdr = fits.getdata(filenames[i], 0, header = True)
                s['KERNEL'] /= np.nansum(s['KERNEL'])
                assert hdr['CD1_1'] == hdr['CD2_2']
                assert hdr['CD1_2'] == hdr['CD2_1'] ==0
                assert hdr['CD1_1'] % 2
                s['PS'] = np.array([hdr['CD1_1'] * u.degree.to(u.arcsec),
                                    hdr['CD2_2'] * u.degree.to(u.arcsec)])
                s['FWHM1'], s['FWHM2'], s['NAME1'], s['NAME2'] = \
                            FWHM1s[i], FWHM2, name1s[i], name2
                s['REGRID'], s['PSR'] = np.zeros([1,1]), np.zeros([1,1])
                self.kernels = self.kernels.append(s.to_frame().T.\
                               set_index(['NAME1','NAME2']))
            except IOError:
                print "Warning: " + filenames[i] + " doesn't exist!!"
        toc = time.clock()
        print "Done. Elapsed time: " + str(round(toc-tic, 3)) + " s."

    def Kernel_regrid(self, name1, name2, target_name = None, \
                      method = 'cubic'):
        """
        Regrid kernel to match with the pixel scale of our image

        Parameters
        ----------
        name1, name2 : 'str'
            Survey names corresponds to the Kernel needs to be regridded.
        target_name : 'str'
            Name of target image. Will be set to name1 one if not entered.
        method : 'str'
            Interpolate method. 'linear', 'cubic', 'quintic'

        Notes
        ---------
        """
        try:
            # Grabbing data
            kernel = self.kernels.loc[name1, name2].KERNEL
            ps = self.kernels.loc[name1, name2].PS
            if not target_name:
                target_name = name1
            ps_new = self.df.xs(target_name, level = 1).PS.values[0]
            
            # Check if the kernel is squared / odd pixel
            assert kernel.shape[0] == kernel.shape[1]
            assert len(kernel) % 2
        
            # Generating grid points. Ref Anaino total dimension ~729", half 364.5"
            l = (len(kernel)-1)//2        
            x = np.arange(-l, l+1) * ps[0] / ps_new[0]
            y = np.arange(-l, l+1) * ps[1] / ps_new[1]
            lxn, lyn = int(l * ps[0] / ps_new[0]), int(l * ps[1] / ps_new[1])
            xn, yn = np.arange(-lxn, lxn+1), np.arange(-lyn, lyn+1)

            print "Start regridding \"" + name1 + " to " + name2 + \
                  "\" kernel to match " + target_name + " map..."
            tic = time.clock()
            k = interp2d(x, y, kernel, kind = method, fill_value=np.nan)
            n_kernel = k(xn, yn)
            n_kernel /= np.sum(n_kernel)
            self.kernels.set_value((name1, name2), 'REGRID', n_kernel)
            self.kernels.set_value((name1, name2), 'PSR', ps_new)
            toc = time.clock()
            print "Done. Elapsed time: " + str(round(toc-tic, 3)) + " s."
            print "Kernel sum: " + str(np.sum(n_kernel))  
            
        except KeyError:
            print "Warning: kernel or target does not exist."

    def matching_PSF_1step(self, names, survey1s, survey2, plotting = True):
        """
        Using 1 step to match PSFs of two images.
            1. Convolve the image with kernel1
    
        Parameters
        ----------
        names, survey1s : 'list of str'
            Object names and survey names to be convolved.
        survey2 : 'str'
            Name of target PSF.
        plotting : 'bool'
            Plots the images or not
        saving : 'bool'
            Saving the first step kernel or not
        
        Notes
        ---------
        Need to regrid kernel2 first
        """
        if type(names) == str:
            names = [names]
        if type(survey1s) == str:
            survey1s = [survey1s]
        
        for survey1 in survey1s:
            ps = self.df.xs(survey1, level = 1).PS.values[0]
            ps2 = self.kernels.loc[survey1, survey2].PS
            err = np.sum(np.abs((ps-ps2)/ps))
            if err > 0.02:
                try:
                    ps2 = self.kernels.loc[survey1, survey2].PSR
                    err = np.sum(np.abs((ps-ps2)/ps))
                    if err > 0.02:
                        self.Kernel_regrid(survey1, survey2, \
                                           target_name = survey1)
                        ps2 = self.kernels.loc[survey1, survey2].PSR
                    kernel = self.kernels.loc[survey1, survey2].REGRID                    
                except AttributeError:
                    self.Kernel_regrid(survey1, survey2, target_name = survey1)
                    ps2 = self.kernels.loc[survey1, survey2].PSR
                    kernel = self.kernels.loc[survey1, survey2].REGRID
            else:
                kernel = self.kernels.loc[survey1, survey2].KERNEL

            for name in names:       
                image1 = self.df.loc[name, survey1].MAP
        
                print "Convolving " + name + " " + survey1 + " map (1/1)..."
                tic = time.clock()

                image1_2 = convolve_fft(image1, kernel)
                image1_2[np.isnan(image1)] = np.nan
                self.df.set_value((name, survey1), 'CVL_MAP', image1_2)
                toc = time.clock()
                print "Done. Elapsed time: " + str(round(toc-tic, 3)) + " s."
                f1, f2 = np.nansum(image1), np.nansum(image1_2)
                print "Normalized flux variation:  " + str(np.abs(f1-f2)/f1)
        
                if plotting:
                    plt.figure()
                    plt.suptitle(survey1 + " " + name, fontsize = 20)
                    plt.subplot(121)
                    plt.imshow(image1, origin = 'lower')
                    plt.title('Original')
                    plt.subplot(122)
                    plt.imshow(image1_2, origin = 'lower')
                    plt.title('Convolved')

    def matching_PSF_2step(self, names, survey1s, k2_survey1, k2_survey2,\
                           plotting = True, saving = False):
        """
        Using 2 steps to match PSFs of two images.
            1. Using w, bpa, bmaj, bmin, FWHM2 to create the first kernel
            2. Convolve the convolved image with kernel2
    
        Parameters
        ----------
        names, survey1s : 'list of str'
            Objects names and surveys of image to be convolved.
        k2_survey1, k2_survey2 : 'str'
            Names of second kernel.
        plotting : 'bool'
            Plots the images or not
        saving : 'bool'
            Saving the first step kernel or not
        
        Notes
        ---------
        Need to regrid kernel2 first
        """
        if type(names) == str:
            names = [names]
        if type(survey1s) == str:
            survey1s = [survey1s]
        
        for survey1 in survey1s:
            ps = self.df.xs(survey1, level = 1).PS.values[0]
            ps2 = self.kernels.loc[k2_survey1, k2_survey2].PS
            err = np.sum(np.abs((ps-ps2)/ps))
            if err > 0.02:
                try:
                    ps2 = self.kernels.loc[k2_survey1, k2_survey2].PSR
                    err = np.sum(np.abs((ps-ps2)/ps))
                    if err > 0.02:
                        self.Kernel_regrid(k2_survey1, k2_survey2, \
                                           target_name = survey1)
                        ps2 = self.kernels.loc[k2_survey1, k2_survey2].PSR
                    kernel2 = self.kernels.loc[k2_survey1, k2_survey2].REGRID                    
                except AttributeError:
                    self.Kernel_regrid(k2_survey1, k2_survey2, \
                                       target_name = survey1)
                    ps2 = self.kernels.loc[k2_survey1, k2_survey2].PSR
                    kernel2 = self.kernels.loc[k2_survey1, k2_survey2].REGRID
            else:
                kernel2 = self.kernels.loc[k2_survey1, k2_survey2].KERNEL

            for name in names:
                bpa = self.df.loc[name, survey1].BPA
                bmaj = self.df.loc[name, survey1].BMAJ
                bmin = self.df.loc[name, survey1].BMIN
                image1 = self.df.loc[name, survey1].MAP
                FWHM2 = self.kernels.loc[k2_survey1, k2_survey2].FWHM1
        
                print "Convolving " + name + " " + survey1 + " map (1/2)..."
                tic = time.clock()
                kernel1 = Gaussian_Kernel_C1(ps, bpa, bmaj, bmin, FWHM2)
                image1_1 = convolve_fft(image1, kernel1)
                image1_1[np.isnan(image1)] = np.nan
                if saving: # Saving generated kernel
                    s = pd.Series()        
                    s['KERNEL'], s['PS'], s['FWHM1'], s['FWHM2'] = kernel1, \
                                                               ps, bmaj, FWHM2
                    s['NAME1'], s['NAME2'] = name + "_" + survey1, k2_survey1
                    s['REGRID'], s['PSR'] = None, None
                    self.kernels = self.kernels.append(s.to_frame().T\
                                   .set_index(['NAME1','NAME2']))
                toc = time.clock()
                print "Done. Elapsed time: " + str(round(toc-tic, 3)) + " s."
                print "Convolving " + name + " " + survey1 + " map (2/2)..."
                tic = time.clock()
                image1_2 = convolve_fft(image1_1, kernel2)
                image1_2[np.isnan(image1)] = np.nan
                self.df.set_value((name, survey1), 'CVL_MAP', image1_2)
                toc = time.clock()
                print "Done. Elapsed time: " + str(round(toc-tic, 3)) + " s."
                f1, f2, f3 = np.nansum(image1), np.nansum(image1_1), \
                             np.nansum(image1_2)
                print "Normalized flux variation. first step:  " + \
                      str(np.abs(f1-f2)/f1)
                print "                           second step: " + \
                      str(np.abs(f2-f3)/f2)
                print "                           overall:     " + \
                      str(np.abs(f1-f3)/f1)
        
                if plotting:
                    plt.figure()
                    plt.suptitle(survey1 + " " + name, fontsize = 20)
                    plt.subplot(131)
                    plt.imshow(image1, origin = 'lower')
                    plt.title('Original')
                    plt.subplot(132)
                    plt.imshow(image1_1, origin = 'lower')
                    plt.title('1st Convolved')
                    plt.subplot(133)
                    plt.imshow(image1_2, origin = 'lower')
                    plt.title('2nd Convolved')

    def WCS_congrid(self, names, fine_surveys, course_survey, \
                    method = 'linear'):
        """
        Matching the WCS and pixel scale of two images.
    
        Parameters
        ----------
        names : 'str'
            Name of objects to be congrid.
        fine_surveys
            Name of surveys to be congrid.
        course_survey : 'str'
            Names of the target survey.
        method : 'str'
            Interpolate method. 'linear', 'nearest', 'cubic'   
            
        Notes
        ---------
        Need to convolve (name, fine_survey) first.
        Might need to add normalization True/False option
        """
        if type(names) == str:
            names = [names]
        if type(fine_surveys) == str:
            fine_surveys = [fine_surveys]

        for fine_survey in fine_surveys:
            for name in names:
                value = self.df.loc[name, fine_survey].CVL_MAP
                if len(value) == 1:
                    print name + " " + fine_survey + " map has not been \
                          convolved. Please convolve first."
                    pass
                print "Start matching " + name + " " + fine_survey + \
                      " grid to match " + course_survey + "..."
                tic = time.clock()
                w1 = self.df.loc[name, fine_survey].WCS
                naxis12, naxis11 = self.df.loc[name, fine_survey].L   # RA, Dec
                w2 = self.df.loc[name, course_survey].WCS
                naxis22, naxis21 = self.df.loc[name, course_survey].L # RA, Dec

                xg, yg = np.meshgrid(np.arange(naxis11), np.arange(naxis12))
                xwg, ywg = w1.wcs_pix2world(xg, yg, 1)
                xg, yg = w2.wcs_world2pix(xwg, ywg, 1)
                xng, yng = np.meshgrid(np.arange(naxis21), np.arange(naxis22))
        
                assert np.size(value) == np.size(xg) == np.size(yg)
                s = np.size(value)
                value = value.reshape(s)
                points = np.concatenate((xg.reshape(s,1),yg.reshape(s,1)),axis=1)

                image1_1 = griddata(points, value, (xng, yng), method = method)
                self.df.set_value((name, fine_survey), 'RGD_MAP', image1_1)
                toc = time.clock()
                print "Done. Elapsed time: " + str(round(toc - tic, 3)) + " s."

    def plot_RGD(self, names, fine_surveys, course_survey):
        """
        Plot original, convolved, regrid images of the fine survey
        and original course survey.
    
        Parameters
        ----------
        name : 'str'
            Name of object.
        fine_survey, course_survey : 'str'
            Names of the two surveys.
            
        Notes
        ---------
        Need to convolve & regrid (name, fine_survey) first.
        """
        for name in names:
            for fine_survey in fine_surveys:
                plt.figure()
                plt.suptitle(name, fontsize = 20)
                plt.subplot(221)
                plt.imshow(self.df.loc[name, fine_survey].MAP, \
                           origin = 'lower')
                plt.title('Original ' + fine_survey)
                plt.subplot(222)
                plt.imshow(self.df.loc[name, fine_survey].CVL_MAP, \
                           origin = 'lower')
                plt.title('Convolved ' + fine_survey)
                plt.subplot(223)
                plt.imshow(self.df.loc[name, fine_survey].RGD_MAP, \
                           origin = 'lower')
                plt.title('Regrid ' + fine_survey)
                plt.subplot(224)
                plt.imshow(self.df.loc[name, course_survey].MAP, \
                           origin = 'lower')
                plt.title('Original ' + course_survey)

    def fit_dust_density(self, names, nwalkers = 10, nsteps = 200):
        """
        Fitting dust mass density from what you know.
    
        Parameters
        ----------
        names : 'str' or 'lis of str'
            Names of objects to be fitted.
            
        Notes
        ---------
        Ref. Gordon 2014
        """
        # Dust fitting constants
        #    Dust density in Solar Mass / pc^2
        #    kappa_lambda in cm^2 / g
        #    SED in MJy / sr
        wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
        const = 2.0891E-4
        kappa160 = 9.6
        # fitting uncertainty = 0.4
        # Calibration uncertainty = 2.5
        THINGS_Limit = 1.0E18
        # HERACLES_LIMIT: heracles*2 > things
        calerr_matrix2 = np.array([0.10,0.10,0.08,0.08,0.08]) ** 2
        # Calibration error of PACS_100, PACS_160, SPIRE_250, SPIRE_350, SPIRE_500
        # For extended source
        bkgerr = np.zeros(5)
        # sigma, T
        theta0 = [10.0, 50.0]
        ndim = len(theta0)
        
        # Probability functions for fitting
        def lnlike(theta, x, y, bkgerr):
            sigma, T = theta
            T = T * u.K
            nu = (c / x / u.um).to(u.Hz)
            
            B = 2*h*nu**3 / c**2 / (np.exp(h*nu/k_B/T) - 1)
            B = (B.to(u.Jy)).value * 1.0E-6   # to MJy    
        
            model = const * kappa160 * (160.0 / x)**2 * sigma * B
            calerr2 = calerr_matrix2 * y**2
            inv_sigma2 = 1.0/(bkgerr**2 + calerr2)
    
            if np.sum(np.isinf(inv_sigma2)):
                return -np.inf
            else:
                return -0.5*(np.sum((y-model)**2*inv_sigma2 \
                       - np.log(inv_sigma2)))
        
        def lnprior(theta):
            sigma, T = theta
            if np.log10(sigma) < 3.0 and 0 < T < 200:
                return 0.0
            return -np.inf
        
        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)
            
        nll = lambda *args: -lnlike(*args)
                
        if type(names) == str:
            names = [names]
        # axissum = [0]*2
        # lc = np.zeros([2,2], dtype=int)
        for name in names:
            things = self.df.loc[(name, 'THINGS')].RGD_MAP
            # Cutting off the nan region of THINGS map.
            """
            for i in xrange(2):
                axissum[i] = np.nansum(things, axis=i, dtype = bool)
                for j in xrange(len(axissum[i])):
                    if axissum[i][j]:
                        lc[i-1, 0] = j
                        break
                lc[i-1, 1] = j + np.sum(axissum[i], dtype=int)
            """
            # Defining image size
            l = self.df.loc[(name, 'SPIRE_500')].L
            
            sed = np.zeros([l[0], l[1], 5])
            # sigma, T, lnf
            popt = np.full([l[0], l[1], ndim], np.nan)
            perr = popt.copy()
            
            heracles = self.df.loc[(name, 'HERACLES')].RGD_MAP
            sed[:,:,0] = self.df.loc[(name, 'PACS_100')].RGD_MAP
            sed[:,:,1] = self.df.loc[(name, 'PACS_160')].RGD_MAP
            sed[:,:,2] = self.df.loc[(name, 'SPIRE_250')].RGD_MAP
            sed[:,:,3] = self.df.loc[(name, 'SPIRE_350')].RGD_MAP
            sed[:,:,4] = self.df.loc[(name, 'SPIRE_500')].MAP
            nanmask = ~np.sum(np.isnan(sed), axis=2, dtype=bool)
            glxmask = (things > THINGS_Limit)
            diskmask = glxmask * (~(heracles*2>things)) * nanmask
            
            # Using the variance of non-galaxy region as uncertainty
            """
            temp = []
            fig = plt.figure()
            ax = fig.add_subplot(2,3,1)
            ax.imshow(things, origin='lower')
            ax.set_title('THINGS map')
            title_temp = ['PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', \
                          'SPIRE_500']
            """
            for i in xrange(5):
                inv_glxmask2 = ~(np.isnan(sed[:,:,i]) + glxmask)
                bkgerr[i] = np.std(sed[inv_glxmask2,i])
                """
                temp.append(sed[:,:,i].copy())
                temp[i][~inv_glxmask2] = np.nan
                ax = fig.add_subplot(2,3,i+2)
                ax.imshow(temp[i], origin='lower')
                ax.set_title(title_temp[i])
                print title_temp[i]
                print 'max: ' + str(np.nanmax(temp[i]))
                print 'min: ' + str(np.nanmin(temp[i]))
                print 'mean: ' + str(np.nanmean(temp[i]))
                print 'bkgerr: ' + str(bkgerr[i])
                print 'total points: ' + str(np.sum(inv_glxmask2))
                """

            ## Debugging part for Herschel images information
            """
            temp2 = []
            plt.figure()
            plt.subplot(2,3,1)
            plt.imshow(things[40:155,35:150], origin='lower')
            plt.title('THINGS map')
            title_temp = ['PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', \
                          'SPIRE_500']
            for i in xrange(5):
                temp2.append(sed[:,:,i].copy())
                temp2[i][~diskmask] = np.nan
                mask3 = (temp2[i] > 3*bkgerr[i])
                mask2 = (temp2[i] > 2*bkgerr[i]) * (temp2[i] <= 3*bkgerr[i])
                mask1 = (temp2[i] > bkgerr[i]) * (temp2[i] <= 2*bkgerr[i])
                mask0 = (temp2[i] <= bkgerr[i])
                temp2[i][mask3] = 3
                temp2[i][mask2] = 2
                temp2[i][mask1] = 1
                temp2[i][mask0] = 0
                plt.subplot(2,3,i+2)
                plt.imshow(temp2[i][40:155,35:150], origin='lower')
                plt.title(title_temp[i])
                print title_temp[i]
                print '> 3 sigma: ' + str(np.sum(mask3))
                print '2~3 sigma: ' + str(np.sum(mask2))
                print '1~2 sigma: ' + str(np.sum(mask1))
                print '< 1 sigma: ' + str(np.sum(mask0))
            """
            
            """
            def reject_outliers(data, sig=2.):
                d = np.abs(data - np.median(data))    # get abs distances to the median
                mad = np.median(d)                   # the median of the devs from the median
                s = d/mad if mad else 0.            # % each deviation is of median dev
                return data[s < sig]                  # want data where this % < n "stdevs"
            """
            
            # Random sampling for [i,j] with high SNR
            # i, j = 95, 86  # NGC 3198
            """
            while(True):
                i = np.random.randint(0,things.shape[0])
                j = np.random.randint(0,things.shape[1])
                cnt = np.sum(sed[i,j,:]>3*(bkgerr)) + diskmask[i, j]
                if cnt == 6:
                    print '[i,j] = ['+str(i)+','+str(j)+']'
                    break
            """
            
            for i in xrange(l[0]):
                for j in xrange(l[1]):
                    print name + ': (' + str(i+1) + '/' + str(l[0]) +', ' +\
                          str(j+1) + '/' + str(l[1]) + ')'
                    if diskmask[i, j]:
                        result = minimize(nll, theta0, \
                                 args=(wl, sed[i,j], bkgerr))
                        pos = np.full([nwalkers, ndim], result['x'])
                        for k in xrange(ndim):
                            pos[:,k] += np.random.normal(0.0, \
                                        np.abs(pos[0,k]/10.0), nwalkers)
                        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob\
                                  , args=(wl, sed[i,j], bkgerr))
                        sampler.run_mcmc(pos, nsteps)
                        
                        temp = sampler.chain[:,nsteps//2:,:].reshape(-1,ndim)
                        del sampler
                        popt[i,j] = np.mean(temp, axis = 0)
                        perr[i,j] = np.std(temp, axis = 0)
                        """
                        temp = np.mean(sampler.chain, axis = 0)
                        plt.figure()
                        plt.suptitle('NGC 3198 [95,86] (10 walkers)')
                        plt.subplot(121)
                        plt.plot(temp[:,0])
                        plt.ylabel(r'Dust surface mass density (M$_\odot$/pc$^2$)')
                        plt.xlabel('Run')
                        plt.subplot(122)
                        plt.plot(temp[:,1])
                        plt.ylabel('Temperature (T)')
                        plt.xlabel('Run')
                        """
                        """
                        x = np.linspace(70,520)
                        sigma, T = popt[i,j]
                        T = T * u.K
                        nu = (c / x / u.um).to(u.Hz)
                        B = 2*h*nu**3 / c**2 / (np.exp(h*nu/k_B/T) - 1)
                        B = (B.to(u.Jy)).value * 1.0E-6   # to MJy    
                        model = const * kappa160 * (160.0 / x)**2 * sigma * B
                        plt.figure()
                        plt.plot(x, model, label='Model')
                        plt.errorbar(wl, sed[i,j], yerr = bkgerr, fmt='o', \
                                     label='Data')                        
                        plt.axis('tight')
                        plt.legend()
                        plt.title('NGC 3198 [95,86] (10 walkers)')
                        plt.xlabel(r'Wavelength ($\mu$m)')
                        plt.ylabel('SED')
                        """
            with h5py.File('dust_data_new.h5', 'a') as hf:
                hf.create_dataset(name+'_popt', data=popt)
                hf.create_dataset(name+'_perr', data=perr)

class Galaxies_M0(object):
    """
    Class designed for manipulating maps from the same survey
    
    Parameters
    ----------
    survey_name : str
        Name of the survey.
        
    Attributes
    ----------
    df : 'pandas.DataFrame'
        DataFrame that stores every thing 
    name : `str`
        Name of the survey.
    conversion : 'float'
        The density conversion factor for a survey.
    galaxy_data : 'pandas.DataFrame'
        Parameters of the survey.
        
    Methods
    ----------
    add_galaxies(self, names, filenames = None, cal = True)
        Add galaxies fits files to self.df.
    add_galaxy(self, name, filename = None, cal = True)
        Add a galaxy fits file to self.df.
    total_mass(self, s)
        Calculate the total mass of the map
    dp_radius(self, s)
        Calculate the deprojected radius from galaxy center for each point.
    Radial_prof(self, s, bins=100)
        Calculate the radial profile of selected galaxy.
    plot_rp_all(self)
        Plot radial profile of all objects
    plot_map_rp(self, name) 
        Plot map and radial profile of a single galaxy.
    plot_map_and_r(self, name)
        Plot map and deprojected radias of a single galaxy.
    mass_compare(self)
        Plot calculated mass versus mass value given in the original paper.

    Notes
    ----------

    """
    def __init__(self, survey_name = 'THINGS'):
        self.df = pd.DataFrame()
        self.name = survey_name
        
        self.galaxy_data = pd.read_csv("galaxy_data.csv")            
        self.galaxy_data.index = self.galaxy_data.OBJECT.values

        if survey_name == 'THINGS':
            # for converting flux to column density
            THINGS_data = pd.read_csv("THINGS_data.csv")
            THINGS_data.index = THINGS_data.OBJECT.values
            self.galaxy_data = pd.concat([self.galaxy_data, \
                               THINGS_data.drop('OBJECT', axis = 1)], axis = 1)
        elif survey_name == 'HERACLES':
            # for converting flux to column density
            self.galaxy_data['BMAJ'] = [13.0] * len(self.galaxy_data)            
            self.galaxy_data['BMIN'] = [13.0] * len(self.galaxy_data)            
            self.galaxy_data['BPA'] = [0.0] * len(self.galaxy_data)            
        else:
            # Not enough information for conversion here
            self.galaxy_data['BMAJ'] = [1.0] * len(self.galaxy_data)            
            self.galaxy_data['BMIN'] = [1.0] * len(self.galaxy_data)            
            self.galaxy_data['BPA'] = [0.0] * len(self.galaxy_data)            
            
    def add_galaxies(self, names, filenames = None, cal = True):
        """
        Add galaxies fits files to self.df.
    
        Parameters
        ----------
        names : 'list of str'
            Names of the objects.
        filenames : 'list of str', default = None.
            File names of the fits. Will be generated from known format if no input.
            Should have the same length as names
        cal : 'bool'
            Perform calculation right after importing file.
            
        Notes
        ---------
        """
        if not filenames:
            filenames = [None] * len(names)
        assert len(filenames) == len(names)
        tic = time.clock()
        print "Importing " + str(len(names)) + " fits files..."
        for i in xrange(len(names)):
            self.add_galaxy(names[i], filenames[i], cal = cal)
        toc = time.clock()
        print "Done. Elapsed time: " + str(round(toc-tic, 3)) + " s."

    def add_galaxy(self, name, filename = None, cal = True):
        """
        Add a galaxy fits file to self.df.
    
        Parameters
        ----------
        name : 'str'
            Name of the object.
        filename : 'str', default = None.
            File name of the fits. Will be generated from known format if no input.
        cal : 'bool'
            Perform calculation right after importing file.
            
        Notes
        ---------
        """
        continuing = True
        
        for i in xrange(len(name)):
            if name[i] in [' ', '_', '0']:
                name1 = name[:i]
                name2 = name[i+1:]
                break
            elif name[i] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                name1 = name[:i]
                name2 = name[i:]
                break
        name = name1.upper() + ' ' + name2

        if self.name in ['SPIRE_500', 'SPIRE_350', 'SPIRE_250', 'PACS_160',\
                         'PACS_100', 'HERACLES']:
            if name1.upper() == 'NGC' and len(name2) == 3:
                name2 = '0' + name2
        
        if self.name == 'THINGS':
            fn = 'THINGS/' + name1 + '_' + name2 + '_NA_MOM0_THINGS.FITS'
        elif self.name == 'SPIRE_500':
            fn = 'SPIRE/' + name1 + '_' + name2 + '_I_500um_hipe_k2011.fits.gz'
        elif self.name == 'SPIRE_350':
            fn = 'SPIRE/' + name1 + '_' + name2 + '_I_350um_hipe_k2011.fits.gz'
        elif self.name == 'SPIRE_250':
            fn = 'SPIRE/' + name1 + '_' + name2 + '_I_250um_hipe_k2011.fits.gz'
        elif self.name == 'PACS_160':
            fn = 'PACS/' + name1 + '_' + name2 + '_I_160um_k2011.fits.gz'
        elif self.name == 'PACS_100':
            fn = 'PACS/' + name1 + '_' + name2 + '_I_100um_k2011.fits.gz'
        elif self.name == 'HERACLES': 
            fn = 'HERACLES/' + name1.lower() + name2 + '_heracles_mom0.fits.gz'
        else:
            continuing = False
            print "Warning: Survey not supported yet!! Please check or enter file name \
                   directly."

        if continuing:
            try:
                s = self.galaxy_data.loc[name].copy()
                if not filename:
                    filename = fn
                data, hdr = fits.getdata(filename,0, header=True)

                if self.name == 'THINGS':
                    # THINGS: Raw data in JY/B*M. Change to
                    # column density 1/cm^2    
                    data = data[0,0]
                    data *= 1.823E18 * 6.07E5 / 1.0E3 / s.BMAJ / s.BMIN                
                elif self.name == 'HERACLES':
                    # HERACLES: Raw data in K*km/s. Change to
                    # column density 1/cm^2
                    # This is a calculated parameter by fitting HI to H2 mass
                    R21 = 0.8
                    XCO = 2.0E20
                    data *= XCO * (R21/0.8)
                else:
                    if self.name in ['PACS_160', 'PACS_100']:
                        data = data[0]
                    print self.name + " not supported for density calculation!!"

                w = wcs.WCS(hdr, naxis=2)
                # add the generated data to dataframe
                s['WCS'], s['MAP'], s['L'] = w, data, np.array(data.shape)
                ctr = s.L // 2
                ps = np.zeros(2)
                xs, ys = w.wcs_pix2world([ctr[0]-1, ctr[0]+1, ctr[0], ctr[0]],
                                         [ctr[1], ctr[1], ctr[1]-1, ctr[1]+1], 1)
                ps[0] = np.abs(xs[0]-xs[1])/2*np.cos((ys[0]+ys[1])*np.pi/2/180)
                ps[1] = np.abs(ys[3]-ys[2])/2
                ps *= u.degree.to(u.arcsec)
                if self.name in ['PACS_160', 'PACS_100']:
                    # Converting Jy/pixel to MJy/sr
                    data *= (np.pi/36/18)**(-2) / ps[0] / ps[1]
                s['PS'] = ps
                s['CVL_MAP'] = np.zeros([1,1])
                s['RGD_MAP'] = np.zeros([1,1])
                if cal:
                    print "Calculating " + name + "..."
                    s['CAL_MASS'] = self.total_mass(s)
                    s['DP_RADIUS'] = self.dp_radius(s)
                    s['RVR'] = self.Radial_prof(s)
                else:
                    s['CAL_MASS'] = np.zeros([1,1])
                    s['DP_RADIUS'] = np.zeros([1,1])
                    s['RVR'] = np.zeros([1,1])
                # Update DataFrame
                self.df = self.df.append(s)
            except KeyError:
                print "Warning: " + name + " not in " + self.name + " csv database!!"
            except IOError:
                print "Warning: " + filename + " doen't exist!!"

    def total_mass(self, s):
        """
        Calculate the total mass of the map.
    
        Parameters
        ----------
        s : 'pandas.Series'
            Series containing dimension, WCS, map and so on.

        Return
        ----------
        mass : 'float'
            Total mass in 1.0E8 Solar mass        

        Notes
        ---------
        """
        if self.name == 'THINGS':
            # THINGS: data in 1/cm^2        
            mean_den = np.nanmean(s.MAP) * m_p
            d = s.D * u.Mpc.to(u.cm)

        elif self.name == 'HERACLES':
            # HERACLES: data in 1/cm^2
            mean_den = np.nanmean(s.MAP) * 2*m_p
            d = s.D * u.Mpc.to(u.cm)

        wh = s.L * s.PS * d * u.arcsec.to(u.radian)        
        mass = wh[0] * wh[1] * mean_den / M_sun * 1.0E-8 * \
               np.sum(~np.isnan(s.MAP)) / float(s.MAP.size)
        return mass.value

    def dp_radius(self, s):
        """
        Calculate the deprojected radius from galaxy center for each point.
    
        Parameters
        ----------
        s : 'pandas.Series'
            Series containing dimension, WCS, map and so on.

        Return
        ----------
        radius : 'numpy.ndarray of float'
            Deprojected radius in kpc of each point. Shape (ly, lx)        
        
        Notes
        ---------
        """
        l = s.L
        radius = np.empty_like(s.MAP)
        cosPA = np.cos((s.PA)*np.pi/180)
        sinPA = np.sin((s.PA)*np.pi/180)
        cosINCL = np.cos(s.INCL*np.pi/180)
        w = s.WCS
        
        ctr = SkyCoord(s.CMC)
        xcm, ycm = ctr.ra.radian, ctr.dec.radian

        dp_coords = np.zeros([l[0],l[1],2])
        # Original coordinate is (y, x)
        # :1 --> x, RA --> the one needed to be divided by cos(incl)
        # :0 --> y, Dec
        dp_coords[:,:,0], dp_coords[:,:,1] = np.meshgrid(np.arange(l[1]), np.arange(l[0]))
        # Now, value inside dp_coords is (x, y)
        # :0 --> x, RA --> the one needed to be divided by cos(incl)
        # :1 --> y, Dec        
        for i in xrange(l[0]):
            dp_coords[i] = w.wcs_pix2world(dp_coords[i],1) * np.pi / 180
        dp_coords[:,:,0] = 0.5*(dp_coords[:,:,0]-xcm)*(np.cos(dp_coords[:,:,1]) + np.cos(ycm))
        dp_coords[:,:,1] -= ycm
        # Now, dp_coords is (dx, dy) in the original coordinate
        # cosPA*dy-sinPA*dx is new y
        # cosPA*dx+sinPA*dy is new x
        radius = np.sqrt((cosPA*dp_coords[:,:,1]+sinPA*dp_coords[:,:,0])**2 
                        +((cosPA*dp_coords[:,:,0]-sinPA*dp_coords[:,:,1])/cosINCL)**2)
        return radius*s.D*1.0E3 # Radius in kpc

    def Radial_prof(self, s, bins=100):
        """
        Calculate the radial profile of selected galaxy.
    
        Parameters
        ----------
        s : 'pandas.Series'
            Series containing dimension, WCS, map and so on.
        bins : 'int'
            bins number for histogram

        Return
        ----------
        RvR : 'numpy.ndarray of float'
            Density versus radius. Shape (bins, lx).
        
        Notes
        ---------
        """
        RvR = np.zeros([bins,2])
        mask = ~np.isnan(s.MAP)
        radius = s.DP_RADIUS[mask]
        data = s.MAP[mask]

        r_min, r_max = np.min(radius), np.max(radius)
        dr = (r_max - r_min)/(2*bins)
        RvR[:,0] = np.linspace(r_min,r_max,(2*bins+1))[1:-1:2]
        for i in xrange(bins):
            mask = np.abs(radius - RvR[i,0]) < dr
            RvR[i,1] = np.sum(data[mask]) / float(np.sum(mask))
        return RvR[RvR[:,1]> np.max(RvR[:,1])*1.0E-4]

    def plot_rp_all(self):
        """
        Plot radial profile of all objects
        
        Notes
        ---------
        Auto adjustment of nor, noc haven't tested yet
        """
        objects = self.df.OBJECT.values
        num_obj = len(objects)
        if num_obj < 4:
            nor, noc = len(objects), 1 # number of rows, number of columns
        elif num_obj == 4:
            nor, noc = 2, 2
        elif num_obj < 7:
            nor, noc = 2, 3
        else:
            nor, noc = 3, 3
        opg = nor * noc # objects per page
        for i in xrange(int(np.ceil(num_obj/opg))):
            plt.figure()
            for j in xrange(opg):
                k = opg*i + j
                if k < num_obj:
                    RvR = self.df.loc[objects[k]].RVR
                    plt.subplot(nor,noc,(j+1))
                    plt.semilogy(RvR[:,0],RvR[:,1])
                    plt.title(self.df.loc[objects[k]].OBJECT)
                    plt.xlabel('Radius (kpc)')
                    plt.ylabel(r'Column density (cm$^{-2}$)')
                    plt.xlim(0,np.max(RvR[:,0]))
                else:
                    pass

    def plot_map_rp(self, name):    
        """
        Plot map and radial profile of a single galaxy.
    
        Parameters
        ----------
        name : 'str'
            Name of the galaxy to be plotted

        Notes
        ---------
        """
        RvR = self.df.loc[name].RVR
        
        fig = plt.figure()
        fig.add_subplot(121, projection = self.df.loc[name].WCS)
        plt.imshow(self.df.loc[name].MAP, origin = 'lower', cmap = 'hot')
        #    , norm=LogNorm(vmin=np.min(data), vmax=np.max(data)))
        plt.colorbar()
        plt.title(name + " " + self.name)
        plt.xlabel('RA (degrees)')
        plt.ylabel('Dec (degrees)')

        fig.add_subplot(122)
        plt.semilogy(RvR[:,0],RvR[:,1])
        plt.title(name + " HI column density radial profile")
        plt.xlabel('Radius (kpc)')
        plt.ylabel(r'Column density (cm$^{-2}$)')
        plt.xlim(0,np.max(RvR[:,0]))

    def plot_map_and_r(self, name):
        """
        Plot map and deprojected radias of a single galaxy.
    
        Parameters
        ----------
        name : 'str'
            Name of the galaxy to be plotted

        Notes
        ---------
        """
        plt.figure()
        plt.suptitle(name, fontsize = 20)
        plt.subplot(121)
        plt.imshow(self.df.loc[name].MAP,origin='lower')
        plt.title(self.name + " map")
        plt.ylabel('pixel')
        plt.xlabel('pixel')
        plt.subplot(122)
        plt.imshow(self.df.loc[name].DP_RADIUS,origin='lower')
        plt.title('Corrected radius')
        plt.xlabel('Pixel')

    def mass_compare(self):
        """
        Plot calculated mass versus mass value given in the original paper.
        
        Notes
        ---------
        """
        cal_mass = self.df.CAL_MASS.values
        paper_mass = self.df.MASS.values
        xmax = max([max(cal_mass),max(paper_mass)]) * 2.0
        xmin = min([min(cal_mass),min(paper_mass)]) / 2.0

        x = np.power(10,np.linspace(np.log10(xmin),np.log10(xmax)))
        y = np.copy(x)
        
        plt.figure()        
        plt.loglog(paper_mass, cal_mass,'ro',x,y,'b-')
        plt.xlabel(r'Mass in original paper (10$^8$ M$_{Sun}$)')
        plt.ylabel(r'Calculated mass (10$^8$ M$_{Sun}$)')
        plt.legend(['Data points', 'y=x, not fitting'], loc=2)  

def Gaussian_rot(x_in, y_in, x_c, y_c, bpa, bmaj, bmin, ps):
    """
    Generate a elliptical, rotated Gaussian PSF.
 
    Parameters
    ----------
    x_in, y_in : 'numpy.ndarray of int'
        The grid points if output PSF. Shape (l, l).
    x_c, y_c : 'float'
        Central pixel of the image.
    bpa : 'float'
        Position angle of the Gaussian in degrees.
    bmaj, bmin : 'float'
        y and x axis FWHM of the Gaussian in arcsec.
    ps : 'list of float'
        Pixel scale of the PSF in arcsec. Shape (2).

    Return
    ----------
    Gaussian_PSF : 'numpy.ndarray of float'
        The generated PSF. Shape (l, l)
        
    Notes
    ---------
    """
    x = (x_in - x_c) * ps[0]
    y = (y_in - y_c) * ps[1]
    bpa = bpa * np.pi / 180.0
    cosbpa = np.cos(bpa)
    sinbpa = np.sin(bpa)
    bmaj *= FWHM2sigma
    bmin *= FWHM2sigma
    a = cosbpa**2 / 2.0 / bmin**2 + sinbpa**2 / 2.0 / bmaj**2
    b = (1/bmin**2 - 1/bmaj**2) * np.sin(2*bpa) / 4.0
    d = sinbpa**2 / 2.0 / bmin**2 + cosbpa**2 / 2.0 / bmaj**2
    return np.exp(-(a*x**2 + 2*b*x*y + d*y**2))

def Gaussian_Kernel_C1(ps, bpa, bmaj, bmin, FWHM=25.0):
    """
    Generate Kernel Case 1: from a elliptical, rotated Gaussian to a normal Gaussian.
 
    Parameters
    ----------
    ps : 'numpy.ndarray of float'
        Pixel scale of the Kernel in arcsec. Shape (2)
    bpa : 'float'
        Position angle of the first Gaussian in degrees.
    bmaj, bmin : 'float'
        y and x axis FWHM of the first Gaussian in arcsec.
    FWHM : 'float'
        FWHM of the second Gaussian in arcsec.

    Return
    ----------
    Gaussian_Kernel_C1 : 'numpy.ndarray of float'
        The generated Kernel with pixel scale ps.
        
    Notes
    ---------
    """
    # Converting scales
    bpa *= np.pi / 180.0
    sigma_x_sq = (FWHM**2 - bmin**2) * FWHM2sigma**2 / ps[0]**2
    sigma_y_sq = (FWHM**2 - bmaj**2) * FWHM2sigma**2 / ps[1]**2
    # Generating grid points. Ref Anaino total dimension ~729", half 364.5"
    lx, ly = int(364.5/ps[0]), int(364.5/ps[1])
    x, y = np.meshgrid(np.arange(-lx, lx+1), np.arange(-ly, ly+1))
    cosbpa, sinbpa = np.cos(bpa), np.sin(bpa)
    xp, yp = cosbpa*x + sinbpa*y, cosbpa*y - sinbpa*x
    result = np.exp(-0.5*( xp**2/sigma_x_sq + yp**2/sigma_y_sq ))
    return result / np.sum(result)
    
def reasonably_close(a, b, pct_err):
    return (np.abs(a-b)/float(a)) < (pct_err/100.0)