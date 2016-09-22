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