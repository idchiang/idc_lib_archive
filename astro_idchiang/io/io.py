import os
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from h5py import File
from time import clock

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
    def __init__(self, names, surveys, auto_import = True):
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
        DATA_PATH = os.path.join(this_dir, "galaxy_data.csv")
        self.galaxy_data = pd.read_csv(DATA_PATH)
        self.galaxy_data.index = self.galaxy_data.OBJECT.values
        
        if auto_import:
            print "Importing " + str(len(names)*len(surveys)) + " fits files..."        
            tic = clock()
            self.add_galaxies(names, surveys)
            # self.df.index.names = ['Name', 'Survey']
            toc = clock()
            print "Done. Elapsed time: " + str(round(toc-tic, 3)) + " s."
                
    def add_galaxies(self, names, surveys, filenames = None):
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
            assert len(names) == len(surveys) == len(filenames), "Input lengths are not equal!!"
            for i in xrange(len(filenames)):
                print "Warning: BMAJ, BMIN, BPA not supported now!!"
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
		
    def add_galaxy(self, name, survey, filename = None):
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

        if survey in ['SPIRE_500', 'SPIRE_350', 'SPIRE_250', 'PACS_160',\
                         'PACS_100', 'HERACLES']:
            if name1.upper() == 'NGC' and len(name2) == 3:
                name2 = '0' + name2
        
        if not filename:
            if survey == 'THINGS':
                filename = 'data/THINGS/' + name1 + '_' + name2 + '_NA_MOM0_THINGS.FITS'
            elif survey == 'SPIRE_500':
                filename = 'data/SPIRE/' + name1 + '_' + name2 + '_I_500um_hipe_k2011.fits.gz'
            elif survey == 'SPIRE_350':
                filename = 'data/SPIRE/' + name1 + '_' + name2 + '_I_350um_hipe_k2011.fits.gz'
            elif survey == 'SPIRE_250':
                filename = 'data/SPIRE/' + name1 + '_' + name2 + '_I_250um_hipe_k2011.fits.gz'
            elif survey == 'PACS_160':
                filename = 'data/PACS/' + name1 + '_' + name2 + '_I_160um_k2011.fits.gz'
            elif survey == 'PACS_100':
                filename = 'data/PACS/' + name1 + '_' + name2 + '_I_100um_k2011.fits.gz'
            elif survey == 'HERACLES': 
                filename = 'data/HERACLES/' + name1.lower() + name2 + '_heracles_mom0.fits.gz'
            else:
                continuing = False
                print "Warning: Survey not supported yet!! Please check or enter file name directly."

        if continuing:
            try:
                s = self.galaxy_data.loc[name].copy()
                del s['TBMIN'], s['TBMAJ'], s['TBPA']
                data, hdr = fits.getdata(filename,0, header=True)

                if survey == 'THINGS':
                    # THINGS: Raw data in JY/B*M. Change to
                    # column density 1/cm^2    
                    data = data[0,0]
                    data *= 1.823E18 * 6.07E5 / 1.0E3 / s.BMAJ / s.BMIN                
                elif survey == 'HERACLES':
                    # HERACLES: Raw data in K*km/s. Change to
                    # column density 1/cm^2
                    # This is a calculated parameter by fitting HI to H2 mass
                    R21 = 0.8
                    XCO = 2.0E20
                    data *= XCO * (R21/0.8)
                else:
                    if survey in ['PACS_160', 'PACS_100']:
                        data = data[0]
                    # print survey + " not supported for density calculation!!"

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
                if survey in ['PACS_160', 'PACS_100']:
                    # Converting Jy/pixel to MJy/sr
                    data *= (np.pi/36/18)**(-2) / ps[0] / ps[1]
                s['PS'] = ps
                s['CVL_MAP'] = np.zeros([1,1])
                s['RGD_MAP'] = np.zeros([1,1])
                s['CAL_MASS'] = np.zeros([1,1])
                s['DP_RADIUS'] = np.zeros([1,1])
                s['RVR'] = np.zeros([1,1])
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
                self.df = self.df.append(s.to_frame().T.set_index([[name],[survey]]))
            except KeyError:
                print "Warning: " + name + " not in " + self.name + " csv database!!"
            except IOError:
                print "Warning: " + filename + " doen't exist!!"