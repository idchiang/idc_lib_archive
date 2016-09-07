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

                if cal:
                    print "Calculating " + name + "..."
                    s['CAL_MASS'] = self.total_mass(s)
                    s['DP_RADIUS'] = self.dp_radius(s)
                    s['RVR'] = self.Radial_prof(s)

    def total_mass(self, s):
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