import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
from astropy.convolution import convolve_fft
from reproject import reproject_exact, reproject_interp
from scipy.interpolate import interp2d
from sklearn import linear_model
from .idc_functions import reasonably_close

plt.ioff()


def load_kernel(kernelpath, res_in, res_out):
    print('Start loading', res_in, 'to', res_out, 'kernel')
    print('Kernel path:', kernelpath)
    if kernelpath[-1] != '/':
        kernelpath += '/'
    filelist = os.listdir(kernelpath)
    for fn in filelist:
        temp = fn.split('_')
        if len(temp) == 7:
            if ('_'.join(temp[2:4]) == res_in) and \
                    ('_'.join(temp[5:])).split('.')[0] == res_out:
                kernel, khdr = fits.getdata(kernelpath + fn, header=True)
                print('Kernel found and imported\n')
                break
    return kernel, khdr


def pixel_scale(data, hdr):
    ps = np.zeros(2)
    w = WCS(hdr, naxis=2)
    ctr = np.array(data.shape) // 2
    xs, ys = \
        w.wcs_pix2world([ctr[0] - 1, ctr[0] + 1, ctr[0], ctr[0]],
                        [ctr[1], ctr[1], ctr[1] - 1, ctr[1] + 1], 1)
    ps[0] = np.abs(xs[0] - xs[1]) / 2 * \
        np.cos(Angle((ys[0] + ys[1]) * u.deg).rad / 2)
    ps[1] = np.abs(ys[3] - ys[2]) / 2
    ps *= u.degree.to(u.arcsec)
    return ps  # ps in arcsec


def regrid_kernel(kernel, ps_new, ps_old, method='linear'):
    # Check if the kernel is squared / odd pixel
    print(' --Assertion checks. Kernel shape:', kernel.shape)
    print(' Old ps:', ps_old)
    print(' New ps:', ps_new)
    assert kernel.shape[0] == kernel.shape[1]
    assert len(kernel) % 2
    # Generating grid points. Ref Anaino total dimension ~729", half 364.5"
    print(' --Defining parameters')
    s = (len(kernel) - 1) // 2
    x = np.arange(-s, s + 1) * ps_old[0] / ps_new[0]
    y = np.arange(-s, s + 1) * ps_old[1] / ps_new[1]
    lxn, lyn = (s * ps_old[0]) // ps_new[0], (s * ps_old[1]) // ps_new[1]
    xn, yn = np.arange(-lxn, lxn + 1), np.arange(-lyn, lyn + 1)
    # Start binning
    print(' --Interpoaltion starts')
    k = interp2d(x, y, kernel, kind=method, fill_value=np.nan)
    n_kernel = k(xn, yn)
    n_kernel /= np.sum(n_kernel)
    print(' --Kernel regrid done. Regrid sum:', np.sum(n_kernel))
    return n_kernel


def convolve_map(data, hdr, kernel, khdr, threshold=0.1, job_name=''):
    print('Convolution job', job_name, 'starts.')
    if khdr is not None:
        ps = pixel_scale(data, hdr)
        kps = np.array([khdr['CD1_1'], khdr['CD2_2']]) * 3600
        if not reasonably_close(ps, kps, 2.0):
            print('Pixel scale does not match. Start Regrid', job_name)
            kernel = regrid_kernel(kernel, ps, kps)
            print('Kernel regrid done', job_name)
    #
    bad_pts = np.full_like(data, 0)
    bad_pts[~np.isfinite(data)] = 1.0
    # Convolve map
    with np.errstate(invalid='ignore', divide='ignore'):
        cdata = convolve_fft(data, kernel, quiet=True, allow_huge=True)
        cbad_pts = convolve_fft(bad_pts, kernel, quiet=True, allow_huge=True)
        cdata[~np.isfinite(data)] = np.nan
        cdata[cbad_pts > threshold] = np.nan
    f1, f2 = np.nansum(data), np.nansum(cdata)
    print("Convolution done", job_name)
    print("--Flux variation (%):", round(100 * (f2 - f1) / f1, 2))
    return cdata


def regrid(data, hdr, tdata, thdr, exact=True, job_name=''):
    print('Regrid job', job_name, 'starts.')
    print('Shape in:', data.shape)
    w_in = WCS(hdr, naxis=2)
    print('Shape out:', tdata.shape)
    w_out = WCS(thdr, naxis=2)
    s_out = tdata.shape
    if exact:
        rdata, _ = reproject_exact((data, w_in),
                                   w_out, s_out)
    else:
        rdata, _ = reproject_interp((data, w_in),
                                    w_out, s_out)
    bitpix = abs(int(hdr['BITPIX']))
    if bitpix == 32:
        rdata = rdata.astype(np.float32)
    elif bitpix == 16:
        rdata = rdata.astype(np.float16)
    print("Regrid done", job_name)
    return rdata


def bkg_tilted_plane(image, bkgmask):
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


def bkg_removal(data, diskmask, job_name=''):
    print('Background removal job', job_name, 'starts.')
    bkgmask = (~diskmask.astype(bool)) * np.isfinite(data)
    print('Num of raw background pixels:', np.sum((~diskmask.astype(bool))))
    print('Num of effective background pixels:', np.sum(bkgmask))
    # tilted plane fitting and plotting
    # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # ax[0, 0].hist(data[bkgmask])
    bkg_plane, coef_ = bkg_tilted_plane(data, bkgmask)
    # ax[0, 1].hist((data - bkg_plane)[bkgmask])
    for i in range(5):
        full_diff = data - bkg_plane
        full_AD = np.abs(full_diff -
                         np.nanmedian(full_diff[bkgmask]))
        MAD = np.nanmedian(full_AD[bkgmask])
        with np.errstate(invalid='ignore'):
            MAD_mask = (full_AD <= 3 * MAD) * bkgmask
        bkg_plane, coef_ = \
            bkg_tilted_plane(data, MAD_mask)
        # ax[1, i].hist((data - bkg_plane)[bkgmask])
    # fig.savefig('output/' + job_name + '.png')
    bdata = data - bkg_plane
    print("Background removal done", job_name)
    return bdata


def radius_arcsec(shape, w, ra, dec, pa, incl,
                  incl_correction=False, cosINCL_limit=0.5):
    # All inputs assumed as Angle
    if incl_correction and (np.isnan(pa.rad + incl.rad)):
        pa = Angle(0 * u.rad)
        incl = Angle(0 * u.rad)
        # Not written to the header
        msg = '\n::z0mgs:: PA or INCL is NaN in ' + \
            'radius calculation \n' + \
            '::z0mgs:: Setting both to zero.'
        # Warning message ends
        warnings.warn(msg, UserWarning)
        # Warning ends
    cosPA, sinPA = np.cos(pa.rad), np.sin(pa.rad)
    cosINCL = np.cos(incl.rad)
    if incl_correction and (cosINCL < cosINCL_limit):
        cosINCL = cosINCL_limit
        # Not written to the header
        msg = '\n::z0mgs:: Large inclination encountered in ' + \
            'radius calculation \n' + \
            '::z0mgs:: Input inclination: ' + str(incl.deg) + \
            ' degrees. \n' + \
            '::z0mgs:: cos(incl) is set to ' + str(cosINCL_limit)
        # Warning message ends
        warnings.warn(msg, UserWarning)
        # Warning ends
    xcm, ycm = ra.rad, dec.rad

    dp_coords = np.zeros(list(shape) + [2])
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
    radius = np.sqrt((cosPA * dp_coords[:, :, 1] +
                      sinPA * dp_coords[:, :, 0])**2 +
                     ((cosPA * dp_coords[:, :, 0] -
                       sinPA * dp_coords[:, :, 1]) / cosINCL)**2)
    radius = Angle(radius * u.rad).arcsec
    return radius


"""
def SFR(self, names, mode=1):
    mode 1: GALEX FUV + MIPS 24 --> SFR
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
