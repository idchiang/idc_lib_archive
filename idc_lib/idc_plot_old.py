# from time import ctime
from h5py import File
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import ticker
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from corner import corner
from .idc_corner import corner_m
from scipy import stats
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.constants import c
from astropy.io import fits
from astropy.wcs import WCS
from .gal_data import gal_data
from .z0mg_RSRF import z0mg_RSRF
from .idc_functions import map2bin, list2bin, bin2list
from .idc_functions import SEMBB, BEMBB, WD, PowerLaw
from .idc_fitting_old import fwhm_sp500, fit_DataY
from .idc_voronoi import voronoi_m
from .idc_io_old import col2sur, H2HaHe
from decimal import Decimal
import time

plt.ioff()
solar_oxygen_bundance = 8.69  # (O/H)_\odot, ZB12
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
bands = ['PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', 'SPIRE_500']


class Dust_Plots(object):
    def __init__(self, fake=False, dbin=1000):
        self.d = {}
        self.cmap0 = 'gist_heat'
        self.cmap1 = 'Greys'
        self.cmap2 = 'seismic'
        self.rbin = 21
        self.dbin = dbin
        self.tbin = 90
        self.x, self.y = 55, 55
        self.fake = fake

    def Load_Data(self, name, method_abbr):
        # Maybe change this to "load a method"
        try:
            self.d['kappa160']
        except KeyError:
            self.d['kappa160'] = {}
        with File('hdf5_MBBDust/Calibration.h5', 'r') as hf:
            grp = hf[method_abbr]
            self.d['kappa160'][method_abbr] = grp['kappa160'].value
        try:
            self.d[name]
            if self.fake:
                with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
                    grp1 = hf['Fake']
                    grp2 = grp1['Bin']
                    grp = grp2[method_abbr]
                    self.d[name]['binlist'] = grp['BINLIST'].value
                    self.d[name]['binmap'] = grp['BINMAP'].value
                    self.d[name]['aGas'] = grp['GAS_AVG'].value
                    self.d[name]['SigmaGas'] = \
                        list2bin(self.d[name]['aGas'], self.d[name]['binlist'],
                                 self.d[name]['binmap'])
                    self.d[name]['aSED'] = grp['Herschel_SED'].value
                    self.d[name]['aRadius'] = grp['Radius_avg'].value
                    self.d[name]['acov_n1'] = \
                        grp['Herschel_covariance_matrix'].value
        except KeyError:
            cd = {}
            with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
                grp = hf['Bin']
                if self.fake:
                    grp1 = hf['Fake']
                    grp2 = grp1['Bin']
                    grp = grp2[method_abbr]
                cd['binlist'] = grp['BINLIST'].value
                cd['binmap'] = grp['BINMAP'].value
                cd['aGas'] = grp['GAS_AVG'].value
                cd['SigmaGas'] = \
                    list2bin(cd['aGas'], cd['binlist'], cd['binmap'])
                cd['aSED'] = grp['Herschel_SED'].value
                cd['aRadius'] = grp['Radius_avg'].value
                cd['acov_n1'] = grp['Herschel_covariance_matrix'].value
                #
                grp = hf['Regrid']
                cd['R25'] = grp['R25_KPC'].value
                cd['aRadius'] /= cd['R25']
                cd['Radius'] = \
                    list2bin(cd['aRadius'], cd['binlist'], cd['binmap'])
                cd['SFR'] = \
                    map2bin(grp['SFR'].value, cd['binlist'], cd['binmap'])
                cd['SMSD'] = \
                    map2bin(grp['SMSD'].value, cd['binlist'], cd['binmap'])
            filename = 'data/PROCESSED/' + name + '/SPIRE_500_CRP.fits'
            data, hdr = fits.getdata(filename, 0, header=True)
            cd['WCS'] = WCS(hdr, naxis=2)
            self.d[name] = cd
        cd = {}
        num_para = {'SE': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4,
                    'WD': 3, 'BE': 3, 'PL': 4}
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Fitting_results']
            if self.fake:
                grp1 = hf['Fake']
                grp = grp1.require_group('Fitting_results')
                cd['aSED'] = self.d[name]['aSED']
            subgrp = grp[method_abbr]
            cd['alogSigmaD'] = subgrp['Dust_surface_density_log'].value
            cd['aSigmaD_err'] = subgrp['Dust_surface_density_err_dex'].value
            cd['diskmask'] = list2bin(~np.isnan(cd['aSigmaD_err']),
                                      self.d[name]['binlist'],
                                      self.d[name]['binmap'])
            cd['logSigmaD'] = \
                list2bin(cd['alogSigmaD'], self.d[name]['binlist'],
                         self.d[name]['binmap'])
            cd['SigmaD_err'] = \
                list2bin(subgrp['Dust_surface_density_err_dex'].value,
                         self.d[name]['binlist'], self.d[name]['binmap'])
            if method_abbr not in ['PL']:
                cd['aT'] = subgrp['Dust_temperature'].value
                cd['T'] = list2bin(cd['aT'], self.d[name]['binlist'],
                                   self.d[name]['binmap'])
                cd['aT_err'] = subgrp['Dust_temperature'].value
                cd['T_err'] = \
                    list2bin(cd['aT_err'], self.d[name]['binlist'],
                             self.d[name]['binmap'])
                cd['aBeta'] = subgrp['beta'].value
                cd['Beta'] = \
                    list2bin(cd['aBeta'], self.d[name]['binlist'],
                             self.d[name]['binmap'])
                cd['aBeta_err'] = subgrp['beta_err'].value
                cd['Beta_err'] = \
                    list2bin(subgrp['beta_err'].value, self.d[name]['binlist'],
                             self.d[name]['binmap'])
            cd['archi2'] = subgrp['Chi2'].value / (5.0 - num_para[method_abbr])
            cd['rchi2'] = \
                list2bin(cd['archi2'], self.d[name]['binlist'],
                         self.d[name]['binmap'])
            cd['aSED'] = subgrp['Best_fit_sed'].value
            cd['aPDFs'] = subgrp['PDF'].value
            self.SigmaDs = 10**subgrp['logsigmas'].value
            """
            logsigma_step = 0.025
            min_logsigma = -4.
            max_logsigma = 1.
            self.SigmaDs = \
                10**np.arange(min_logsigma, max_logsigma, logsigma_step)
            """
            if method_abbr in ['SE', 'FB', 'BEMFB', 'PB', 'WD', 'BE']:
                cd['aPDFs_T'] = subgrp['PDF_T'].value
                self.Ts = subgrp['Ts'].value
            elif method_abbr in ['FBPT']:
                self.FBPT_Ts = subgrp['Ts'].value
            if method_abbr == 'SE':
                self.betas = subgrp['betas'].value
            if method_abbr in ['WD', 'PL']:
                cd['aPDFs_T'] = subgrp['PDF_Teff'].value
                cd['Teff_bins'] = subgrp['Teff_bins'].value
            if method_abbr in ['BEMFB', 'BE']:
                cd['alambda_c'] = subgrp['Critical_wavelength'].value
                if len(cd['alambda_c'].shape) > 1:
                    cd['alambda_c'] = np.full(2176, 300.0)
                cd['lambda_c'] = \
                    list2bin(cd['alambda_c'], self.d[name]['binlist'],
                             self.d[name]['binmap'])
                cd['lambda_c_err'] = \
                    list2bin(subgrp['Critical_wavelength_err'].value,
                             self.d[name]['binlist'], self.d[name]['binmap'])
                if method_abbr == 'BEMFB':
                    self.lambda_cs = subgrp['lambda_cs'].value
                    cd['aPDFs_lc'] = subgrp['PDF_lc'].value
                cd['abeta2'] = subgrp['beta2'].value
                cd['beta2'] = \
                    list2bin(cd['abeta2'], self.d[name]['binlist'],
                             self.d[name]['binmap'])
                cd['beta2_err'] = \
                    list2bin(subgrp['beta2_err'].value,
                             self.d[name]['binlist'], self.d[name]['binmap'])
                self.beta2s = subgrp['beta2s'].value
                cd['aPDFs_b2'] = subgrp['PDF_b2'].value
            if method_abbr in ['WD']:
                cd['aWDfrac'] = subgrp['WDfrac'].value
                cd['WDfrac'] = list2bin(cd['aWDfrac'], self.d[name]['binlist'],
                                        self.d[name]['binmap'])
                cd['aWDfrac_err'] = subgrp['WDfrac_err'].value
                cd['WDfrac_err'] = \
                    list2bin(cd['aWDfrac_err'], self.d[name]['binlist'],
                             self.d[name]['binmap'])
                self.WDfracs = subgrp['WDfracs'].value
                cd['aPDFs_Wf'] = subgrp['PDF_Wf'].value
            if method_abbr in ['PL']:
                cd['aalpha'] = subgrp['alpha'].value
                cd['alpha'] = list2bin(cd['aalpha'], self.d[name]['binlist'],
                                       self.d[name]['binmap'])
                cd['alpha_err'] = list2bin(subgrp['alpha_err'].value,
                                           self.d[name]['binlist'],
                                           self.d[name]['binmap'])
                self.alphas = subgrp['alphas'].value
                cd['aloggamma'] = subgrp['loggamma'].value
                cd['loggamma'] = list2bin(cd['aloggamma'],
                                          self.d[name]['binlist'],
                                          self.d[name]['binmap'])
                cd['loggamma_err'] = list2bin(subgrp['loggamma_err'].value,
                                              self.d[name]['binlist'],
                                              self.d[name]['binmap'])
                self.gammas = 10**subgrp['loggammas'].value
                cd['alogUmin'] = subgrp['logUmin'].value
                cd['logUmin'] = list2bin(cd['alogUmin'],
                                         self.d[name]['binlist'],
                                         self.d[name]['binmap'])
                cd['logUmin_err'] = list2bin(subgrp['logUmin_err'].value,
                                             self.d[name]['binlist'],
                                             self.d[name]['binmap'])
                self.logUmins = subgrp['logUmins'].value
            if method_abbr in ['FBPT', 'PB']:
                cd['coef_'] = subgrp['coef_'].value
        self.d[name][method_abbr] = cd

    def STBC(self, name, method_abbr, err_selc=0.3):
        """
        Plotting all fitting results.
        Named from Sigma_d, T, beta, and chi^2
        """
        plt.close('all')
        print(' --Plotting fitting results from', method_abbr, '...')
        #
        """
        if method_abbr == 'SE':
            params = ['logSigmaD', 'SigmaD_err', 'T', 'T_err',
                      'Beta', 'Beta_err']
        elif method_abbr == 'FB':
            params = ['logSigmaD', 'SigmaD_err', 'T', 'T_err']
        elif method_abbr == 'BE':
            params = ['logSigmaD', 'SigmaD_err', 'T', 'T_err',
                      'beta2', 'beta2_err']
        elif method_abbr == 'WD':
            params = ['logSigmaD', 'SigmaD_err', 'T', 'T_err',
                      'WDfrac', 'WDfrac_err']
        elif method_abbr == 'PL':
            params = ['logSigmaD', 'SigmaD_err', 'alpha', 'alpha_err',
                      'loggamma', 'loggamma_err', 'logUmin', 'logUmin_err']
        params.append('rchi2')
        """
        if method_abbr == 'SE':
            params = ['logSigmaD', 'T', 'Beta']
        elif method_abbr == 'FB':
            params = ['logSigmaD', 'T']
        elif method_abbr == 'BE':
            params = ['logSigmaD', 'T', 'beta2']
        elif method_abbr == 'WD':
            params = ['logSigmaD', 'T', 'WDfrac']
        elif method_abbr == 'PL':
            params = ['logSigmaD', 'alpha', 'loggamma', 'logUmin']
        #
        SigmaD_err = self.d[name][method_abbr]['SigmaD_err']
        with np.errstate(invalid='ignore'):
            nanmask = SigmaD_err > err_selc
        #
        titles = {'logSigmaD': r'$\log_{10}\Sigma_d$ $(M_\odot/pc^2)$',
                  'SigmaD_err': r'$\log_{10}\Sigma_d$ error',
                  'T': r'$T_d$ (K)',
                  'T_err': r'$T_d$ error',
                  'Beta': r'$\beta$',
                  'Beta_err': r'$\beta$ error',
                  'rchi2': r'$\tilde{\chi}^2$',
                  'beta2': r'$\beta_2$',
                  'beta2_err': r'$\beta_2$ error',
                  'WDfrac': r'$f_W$',
                  'WDfrac_err': r'$f_W$ error',
                  'alpha': r'$\alpha$',
                  'alpha_err': r'$\alpha$ error',
                  'loggamma': r'$\log_{10}\gamma$',
                  'loggamma_err': r'$\log_{10}\gamma$ error',
                  'logUmin': r'$\log_{10}U_{min}$',
                  'logUmin_err': r'$\log_{10}U_{min}$ error'}
        rows = len(params)
        s = 2
        fig = plt.figure(figsize=(2 * s, rows * s))
        cmap = 'viridis'
        cm = plt.cm.get_cmap(cmap)
        #
        gs1 = GridSpec(rows, 1)
        gs2 = GridSpec(rows, 1)
        for i in range(rows):
            image = self.d[name][method_abbr][params[i]]
            image[nanmask] = np.nan
            max_, min_ = np.nanmax(image), np.nanmin(image)
            ax = fig.add_subplot(gs1[i])
            ax.imshow(self.d[name][method_abbr]['diskmask'],
                      origin='lower', cmap='Greys_r', alpha=0.5)
            ax.imshow(image, origin='lower', cmap=cmap, vmax=max_, vmin=min_)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[params[i]], x=0.05, y=0.85, size=10, ha='left')
            #
            ax = fig.add_subplot(gs2[i])
            bins = np.linspace(min_, max_, 12)
            mask2 = ~np.isnan(image)
            image = image[mask2]
            n, bins, patches = \
                ax.hist(image, bins=bins,
                        weights=self.d[name]['SigmaGas'][mask2],
                        orientation='horizontal')
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)
            for j in range(len(col)):
                plt.setp(patches[j], 'facecolor', cm(col[j]))
            ax.set_xticks([])
            ax.minorticks_on()
        gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])
        gs2.tight_layout(fig, rect=[0.5, 0, 1, 1])
        #
        fn = 'output/_STBC_' + name + '_' + method_abbr + '.pdf'
        with np.errstate(invalid='ignore'):
            with PdfPages(fn) as pp:
                pp.savefig(fig, bbox_inches='tight')

    def STBC_uncver(self, name='NGC5457', method_abbr='BE', err_selc=1.0):
        """
        Plotting all fitting results.
        Named from Sigma_d, T, beta, and chi^2
        """
        plt.close('all')
        print(' --Plotting fitting results from', method_abbr, '...')
        #
        if method_abbr == 'SE':
            params = ['logSigmaD', 'T', 'Beta']
        elif method_abbr == 'FB':
            params = ['logSigmaD', 'T']
        elif method_abbr == 'BE':
            params = ['logSigmaD', 'T', 'beta2']
        elif method_abbr == 'WD':
            params = ['logSigmaD', 'T', 'WDfrac']
        elif method_abbr == 'PL':
            params = ['logSigmaD', 'alpha', 'loggamma', 'logUmin']
        errs = {'logSigmaD': 'SigmaD_err', 'T': 'T_err', 'Beta': 'Beta_err',
                'beta2': 'beta2_err', 'WDfrac': 'WDfrac_err',
                'alpha': 'alpha_err', 'loggamma': 'loggamma_err',
                'logUmin': 'logUmin_err'}
        #
        SigmaD_err = self.d[name][method_abbr]['SigmaD_err']
        with np.errstate(invalid='ignore'):
            nanmask = SigmaD_err > err_selc
        #
        titles = {'logSigmaD': r'$\log_{10}\Sigma_d$ $(M_\odot/pc^2)$',
                  'SigmaD_err': r'$\log_{10}\Sigma_d$ unc.',
                  'T': r'$T_d$ (K)',
                  'T_err': r'$T_d$ unc.',
                  'Beta': r'$\beta$',
                  'Beta_err': r'$\beta$ unc.',
                  'rchi2': r'$\tilde{\chi}^2$',
                  'beta2': r'$\beta_2$',
                  'beta2_err': r'$\beta_2$ unc.',
                  'WDfrac': r'$f_W$',
                  'WDfrac_err': r'$f_W$ unc.',
                  'alpha': r'$\alpha$',
                  'alpha_err': r'$\alpha$ unc.',
                  'loggamma': r'$\log_{10}\gamma$',
                  'loggamma_err': r'$\log_{10}\gamma$ unc.',
                  'logUmin': r'$\log_{10}U_{min}$',
                  'logUmin_err': r'$\log_{10}U_{min}$ unc.'}
        rows = len(params)
        s = 2
        fig = plt.figure(figsize=(2.3 * s, rows * s))
        cmap = 'viridis'
        #
        gs1 = GridSpec(rows, 1)
        gs2 = GridSpec(rows, 1)
        for i in range(rows):
            image = self.d[name][method_abbr][params[i]]
            image[nanmask] = np.nan
            max_, min_ = np.nanmax(image), np.nanmin(image)
            ax = fig.add_subplot(gs1[i])
            ax.imshow(self.d[name][method_abbr]['diskmask'],
                      origin='lower', cmap='Greys_r', alpha=0.5)
            im = ax.imshow(image, origin='lower', cmap=cmap, vmax=max_,
                           vmin=min_)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            ax.set_title(titles[params[i]], x=0.05, y=0.85, size=10, ha='left')
            #
            image_name = errs[params[i]]
            image = self.d[name][method_abbr][image_name]
            image[nanmask] = np.nan
            max_, min_ = np.nanmax(image), np.nanmin(image)
            ax = fig.add_subplot(gs2[i])
            ax.imshow(self.d[name][method_abbr]['diskmask'],
                      origin='lower', cmap='Greys_r', alpha=0.5)
            im = ax.imshow(image, origin='lower', cmap=cmap, vmax=max_,
                           vmin=min_)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            ax.set_title(titles[image_name], x=0.05, y=0.85, size=10,
                         ha='left')
        gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])
        gs2.tight_layout(fig, rect=[0.5, 0, 1, 1])
        #
        fn = 'output/_STBC_' + name + '_' + method_abbr + '.pdf'
        with np.errstate(invalid='ignore'):
            with PdfPages(fn) as pp:
                pp.savefig(fig, bbox_inches='tight')

    def pdf_profiles(self, name, method_abbr):
        """
        DGR profile w/ linear regression
        DGR data point spread
        DTM vs metallicity
        DGR vs. D14
        DGR vs. R14
        DGR vs. J09
        """
        plt.close('all')
        # print(' --Plotting DGR/(O/H) profile...')
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        r = d = w = dm = np.array([])  # radius, dgr, weight, DTM, gas
        for i in range(len(self.d[name]['binlist'])):
            temp_G = self.d[name]['aGas'][i]
            temp_R = self.d[name]['aRadius'][i]
            mask = self.d[name][method_abbr]['aPDFs'][i] > \
                self.d[name][method_abbr]['aPDFs'][i].max() / 1000
            temp_DGR = self.SigmaDs[mask] / temp_G
            temp_P = self.d[name][method_abbr]['aPDFs'][i][mask]
            temp_P = temp_P / np.sum(temp_P) * temp_G * \
                (self.d[name]['binmap'] == self.d[name]['binlist'][i]).sum()
            r = np.append(r, [temp_R] * len(temp_P))
            m = 10**((8.715 - 0.027 * temp_R * self.d[name]['R25'] * 7.4 /
                      GD_dist) - 12.0) * 16.0 / 1.008 / 0.51 / 1.36
            d = np.append(d, temp_DGR)
            dm = np.append(dm, temp_DGR / m)
            w = np.append(w, temp_P)
        nanmask = np.isnan(r + d + w)
        r, d, w, dm = r[~nanmask], d[~nanmask], w[~nanmask], dm[~nanmask]
        rbins = np.linspace(np.min(r), np.max(r), self.rbin)
        dbins = \
            np.logspace(np.log10(3E-5), np.log10(6E-2), self.dbin)
        dmbins = \
            np.logspace(np.min(np.log10(dm)), np.max(np.log10(dm)), self.dbin)
        # Counting hist2d...
        counts, _, _ = np.histogram2d(r, d, bins=(rbins, dbins), weights=w)
        counts3, _, _ = np.histogram2d(r, dm, bins=(rbins, dmbins), weights=w)
        del r, d, w, dm
        counts, counts3 = counts.T, counts3.T
        dbins2 = np.sqrt(dbins[:-1] * dbins[1:])
        dmbins2 = (dmbins[:-1] + dmbins[1:]) / 2
        rbins2 = (rbins[:-1] + rbins[1:]) / 2
        DGR_Median = DGR_LExp = DGR_Max = DGR_16 = DGR_84 = np.array([])
        DTM_LExp = DTM_16 = DTM_84 = np.array([])
        n_zeromask = np.full(counts.shape[1], True, dtype=bool)
        # need them now
        df = pd.read_csv("data/Tables/Remy-Ruyer_2014.csv")
        xbins2 = (8.715 - 0.027 * rbins2 * self.d[name]['R25'] * 7.4 / GD_dist)
        #
        for i in range(counts.shape[1]):
            if np.sum(counts[:, i]) > 0:
                counts[:, i] /= np.sum(counts[:, i])
                counts3[:, i] /= np.sum(counts3[:, i])
                csp = np.cumsum(counts[:, i])
                csp = csp / csp[-1]
                ssd = np.interp([0.16, 0.5, 0.84], csp, np.log10(dbins2))
                DGR_Median = np.append(DGR_Median, 10**ssd[1])
                DGR_LExp = np.append(DGR_LExp, 10**np.sum(np.log10(dbins2) *
                                                          counts[:, i]))
                DGR_Max = np.append(DGR_Max, dbins2[np.argmax(counts[:, i])])
                DGR_16 = np.append(DGR_16, 10**ssd[0])
                DGR_84 = np.append(DGR_84, 10**ssd[2])
                #
                csp = np.cumsum(counts3[:, i])
                csp = csp / csp[-1]
                ssd = np.interp([0.16, 0.5, 0.84], csp, np.log10(dmbins2))
                DTM_LExp = np.append(DTM_LExp, 10**np.sum(np.log10(dmbins2) *
                                                          counts3[:, i]))
                DTM_16 = np.append(DTM_16, 10**ssd[0])
                DTM_84 = np.append(DTM_84, 10**ssd[2])
            else:
                n_zeromask[i] = False
        print(DGR_LExp) 
        R25 = self.d[name]['R25']
        # first line: MO/MZ
        # second line: metallicity
        DTM_unc_dex = 0.06 + \
            (0.023 + (0.001 * rbins2 * R25 * 7.4 / GD_dist))
        DGR_unc_dex = np.zeros_like(rbins2)
        # third line: gas zero point uncertainty
        radius_map = np.zeros_like(self.d[name]['binmap']) * np.nan
        gas_map = np.zeros_like(self.d[name]['binmap']) * np.nan
        for i in range(len(self.d[name]['binlist'])):
            b_mask = self.d[name]['binmap'] == self.d[name]['binlist'][i]
            radius_map[b_mask] = self.d[name]['aRadius'][i]
            gas_map[b_mask] = self.d[name]['aGas'][i]
        for i in range(len(DTM_unc_dex)):
            u_mask = (rbins[i] <= radius_map) * (rbins[i + 1] >= radius_map)
            temp_gas = np.nanmean(gas_map[u_mask])
            DTM_unc_dex[i] += np.log10((temp_gas + 1) / temp_gas)
            DGR_unc_dex[i] += np.log10((temp_gas + 1) / temp_gas)
        DTM_unc_mtp = 10**DTM_unc_dex
        DGR_unc_mtp = 10**DGR_unc_dex
        #
        DTM_16 /= DTM_unc_mtp
        DTM_84 *= DTM_unc_mtp
        DGR_16 /= DGR_unc_mtp
        DGR_84 *= DGR_unc_mtp
        #
        # My linear fitting
        #
        print(method_abbr)
        xbins2 = (8.715 - 0.027 * rbins2 * self.d[name]['R25'] * 7.4 / GD_dist)
        DataX = xbins2[n_zeromask]
        DataY = np.log10(DGR_LExp)
        yerr = np.zeros_like(DGR_LExp)
        for yi in range(len(yerr)):
            yerr[yi] = max(np.log10(DGR_84[yi]) - np.log10(DGR_LExp[yi]),
                           np.log10(DGR_LExp[yi]) - np.log10(DGR_16[yi]))
        DGR_full, coef_ = fit_DataY(DataX, DataY, yerr)
        print(method_abbr)
        DGR_full = 10**DGR_full
        nmask = xbins2[n_zeromask] < 8.2
        yerr[nmask] = np.log10(DGR_LExp)[nmask]
        DGR_part, coef2_ = fit_DataY(DataX, DataY, yerr)
        DGR_part = 10**DGR_part
        #
        # Fitting end
        #
        df = pd.read_csv("data/Tables/Remy-Ruyer_2014.csv")
        r_ = (8.715 - df['12+log(O/H)'].values) / 0.027 * GD_dist / 7.4 / \
            self.d[name]['R25']
        r__ = np.linspace(np.nanmin(r_), np.nanmax(r_), 50)
        # log(Z/Z_solar)
        x__ = (8.715 - 0.027 * r__ * self.d[name]['R25'] * 7.4 / GD_dist -
               solar_oxygen_bundance)
        # Oxygen abundance (log)
        o__ = x__ + solar_oxygen_bundance
        zl = np.log10(1.81 * np.exp(-18 / 19))
        zu = np.log10(1.81 * np.exp(-8 / 19))
        # H2 fraction
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Regrid']
            heracles = grp['HERACLES'].value
        aH2 = np.array([np.nanmean(heracles[self.d[name]['binmap'] == bin_])
                        for bin_ in self.d[name]['binlist']])
        rH2, H2frac = self.simple_profile(aH2 / self.d[name]['aGas'],
                                          self.d[name]['aRadius'],
                                          self.rbin,
                                          self.d[name]['aGas'])
        funcH2 = interp1d(H2frac[:10], rH2[:10])
        oH204 = 8.715 - 0.027 * funcH2(0.4) * self.d[name]['R25'] * 7.4 / \
            GD_dist
        oH202 = 8.715 - 0.027 * funcH2(0.2) * self.d[name]['R25'] * 7.4 / \
            GD_dist
        oH2005 = 8.715 - 0.027 * funcH2(0.05) * self.d[name]['R25'] * 7.4 / \
            GD_dist
        #
        # My own fitting and data points
        #
        print(' --Plotting DGR vs. Metallicity...')
        fig, ax = plt.subplots(nrows=3, figsize=(5, 12))
        with np.errstate(invalid='ignore'):
            tempo = (8.715 - 0.027 * self.d[name]['aRadius'] *
                     self.d[name]['R25'] * 7.4 / GD_dist)
            tempd = 10**self.d[name][method_abbr]['alogSigmaD'] / \
                self.d[name]['aGas']
            tempe_dex = self.d[name][method_abbr]['aSigmaD_err']
            tempg = self.d[name]['aGas']
            tempr = self.d[name]['aRadius']
        nonnanmask = ~np.isnan(tempo + tempd + tempe_dex + tempg + tempr)
        tempo, tempd, tempe_dex = \
            tempo[nonnanmask], tempd[nonnanmask], tempe_dex[nonnanmask]
        tempg, tempr = \
            tempg[nonnanmask], tempr[nonnanmask]
        tempe_dex += (0.023 + (0.001 * tempr * self.d[name]['R25'] * 7.4 /
                               GD_dist)) + np.log10((tempg + 1) / tempg)
        tempe = 10**tempe_dex
        yerr = np.array([tempd * (1 - 1 / tempe), tempd * (tempe - 1)])
        ax[0].errorbar(tempo, tempd, yerr=yerr, alpha=0.3, ms=1, fmt='o',
                       elinewidth=0.5, label='DGR Results')
        ax[0].fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                           color='lightgray', label=r'DGR scatter')
        #
        ax[1].fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                           color='lightgray', label=r'DGR scatter')
        ax[1].plot(xbins2[n_zeromask], DGR_LExp, label='Exp. DGR',
                   linewidth=3.0)
        """
        # 06/22: send data to Kuan-Chou
        df2 = pd.DataFrame()
        df2['12+log(O/H)'] = xbins2[n_zeromask]
        df2['DGR (Expectated value)'] = DGR_LExp
        df2['DGR-1_sigma'] = DGR_16
        df2['DGR+1_sigma'] = DGR_84
        df2.to_csv('output/Chiang18_M101_DGR.csv', index=False)
        """
        ax[1].plot(xbins2[n_zeromask], DGR_full, '-.',
                   label='Fit', linewidth=2.0)
        ax[1].plot(xbins2[n_zeromask], DGR_part, '--',
                   label='Fit (High Z)', linewidth=2.0)
        #
        ax[2].fill_between(xbins2[n_zeromask], DTM_16, DTM_84,
                           color='lightgray', label=r'DTM scatter')
        ax[2].plot(xbins2[n_zeromask], DTM_LExp, label='Exp. DTM',
                   linewidth=3.0, color='c')
        for y in np.arange(0.1, 1.1, 0.1):
            ax[2].plot([np.nanmin(xbins2), np.nanmax(xbins2)], [y] * 2, 'gray',
                       lw=1, alpha=0.5)
        #
        xlim = ax[1].get_xlim()
        xlim_tight = [np.nanmin(xbins2), np.nanmax(xbins2)]
        ylim = [3E-5, 6E-2]
        ax[2].set_xlabel('12 + log(O/H)', size=12)
        titles = ['(a)', '(b)', '(c)']
        for i in range(2):
            ax[i].set_yscale('log')
            ax[i].set_ylabel('DGR', size=12)
            ax[i].set_xlim(xlim)
            ax[i].set_ylim(ylim)
            ax[i].legend(fontsize=12, framealpha=1.0, loc=4)
            ax[i].set_title(titles[i], size=16, x=0.1, y=0.85)
        i = 2
        ax[i].set_yscale('log')
        ax[i].set_ylabel('DTM', size=12)
        ax[i].set_xlim(xlim)
        # ax[i].set_ylim(ylim)
        ax[i].legend(fontsize=12, framealpha=1.0, loc=4)
        ax[i].set_title(titles[i], size=16, x=0.1, y=0.85)
        del i
        #
        fn = 'output/_DTM_C18_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        return
        #
        # Independent R14 plot
        #
        print(' --Plotting DTM with R14...')
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                        color='lightgray')
        ax.plot(xbins2[n_zeromask], DGR_LExp, linewidth=3.0, label='This work')
        ax.plot(o__, 10**(2.02 * x__ - 2.21), '--', linewidth=3.0,
                alpha=0.8, label='R14 power')
        ax.plot(o__, self.BPL_DGR(x__, 'Z'), ':', linewidth=3.0,
                alpha=0.8, label='R14 broken')
        ax.scatter(df['12+log(O/H)'], df['DGR_Z'], c='b', s=15,
                   label='R14 data')
        ax.set_yscale('log')
        ax.set_ylabel('DGR', size=12)
        ax.set_xlim([np.nanmin(df['12+log(O/H)']),
                     np.nanmax(df['12+log(O/H)'])])
        ax.set_ylim(ylim)
        ax.set_xlim([7.25, np.nanmax(o__)])
        ax.legend(fontsize=10, framealpha=0.5)
        ax.set_xlabel('12 + log(O/H)', size=12)
        fn = 'output/_DTM_R14_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        #
        # Independent D14 plot
        #
        print(' --Plotting DTM with D14...')
        """
        #
        # Single D14 plot for poster
        #
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                        color='lightgray')
        ax.fill_between([zl + solar_oxygen_bundance,
                         zu + solar_oxygen_bundance], 3E-5, 6E-2,
                        color='green', alpha=0.2,
                        label='D14: Selected by Z')
        ax.fill_between([oH2005, oH202], 3E-5, 6E-2, color='red', alpha=0.2,
                        label=r'D14: Selected by f$_{H_2}$')
        ax.plot(xbins2[n_zeromask], DGR_LExp, linewidth=3.0)
        ax.plot(o__[o__ > zl + solar_oxygen_bundance],
                10**(x__[o__ > zl + solar_oxygen_bundance]) / 150, 'k',
                linewidth=3.0, alpha=1, label='D14')
        ax.plot(o__, 10**(x__) / 150, 'k--', linewidth=3.0, alpha=1)
        ax.set_yscale('log')
        ax.set_ylabel('DGR', size=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.minorticks_on()
        ax.legend(fontsize=12, framealpha=1.0, loc=4)
        ax.set_xlabel('12 + log(O/H)', size=12)
        fn = 'output/_DTM_D14_Poster.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        """
        fig, ax = plt.subplots(2, 1, figsize=(5, 8))
        ax[0].fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                           color='lightgray')
        ax[0].fill_between([zl + solar_oxygen_bundance,
                            zu + solar_oxygen_bundance], 3E-5, 6E-2,
                           color='green', alpha=0.2)
        ax[0].fill_between([oH2005, oH202], 3E-5, 6E-2, color='red', alpha=0.2)
        ax[0].plot(xbins2[n_zeromask], DGR_LExp, linewidth=3.0,
                   label='This work')
        ax[0].plot(o__[o__ > zl + solar_oxygen_bundance],
                   10**(x__[o__ > zl + solar_oxygen_bundance]) / 150, 'k',
                   linewidth=3.0, alpha=1, label='D14')
        ax[0].plot(o__, 10**(x__) / 150, 'k--', linewidth=3.0, alpha=1)
        ax[0].set_yscale('log')
        ax[0].set_ylabel('DGR', size=12)
        ax[0].set_xlim(xlim_tight)
        ax[0].set_ylim(ylim)
        ax[0].minorticks_on()
        ax[0].legend(fontsize=12, framealpha=1.0, loc=4)
        # ax[0].set_xlabel('12 + log(O/H)', size=12)
        #
        fH2s = np.linspace(0.03, 0.9)
        r25_corr = self.d[name]['R25'] * 7.4 / GD_dist
        ylim_fH2 = (0.05, 0.7)
        ax[1].fill_between([zl + solar_oxygen_bundance,
                            zu + solar_oxygen_bundance],
                           0.03, 0.2,
                           color='Grey', alpha=0.8, zorder=4)
        ax[1].plot(8.715 - 0.027 * funcH2(fH2s) * r25_corr, fH2s,
                   label='This work: M101', linewidth=2)
        ax[1].fill_between([zl + solar_oxygen_bundance,
                            zu + solar_oxygen_bundance],
                           ylim_fH2[0], ylim_fH2[1],
                           color='green', alpha=0.2,
                           label='D14: Selected by Z')
        ax[1].fill_between([oH2005, oH202], ylim_fH2[0], ylim_fH2[1],
                           color='red', alpha=0.2,
                           label=r'D14: Selected by f$_{H_2}$')
        ax[1].plot(xlim_tight, [0.2] * 2, 'k', linestyle='dotted', alpha=0.3)
        ax[1].plot([zl + solar_oxygen_bundance] * 2, ylim_fH2, 'k',
                   linestyle='dotted', alpha=0.3)
        ax[1].text((zl + solar_oxygen_bundance + xlim_tight[1]) / 2,
                   (ylim[0] + 0.2) / 2, 'D14 Data',
                   horizontalalignment='center',
                   verticalalignment='bottom', zorder=5)
        ax[1].set_xlim(xlim_tight)
        ax[1].set_ylim(ylim_fH2)
        ax[1].minorticks_on()
        ax[1].set_ylabel(r'f$_{H_2}$', size=12)
        ax[1].set_xlabel('12 + log(O/H)', size=12)
        ax[1].legend()
        fn = 'output/_DTM_D14_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        #
        # Independent Jenkins plot
        #
        print(' --Plotting DTM with J09...')
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                        color='lightgray')
        ax.plot(xbins2[n_zeromask], DGR_LExp, linewidth=2.0, label='This work')
        fs = [0, 0.36, 1, 1E9]
        fs_str = ['0.00', '0.36', '1.00', 'inf']
        for f in range(4):
            y = self.Jenkins_F_DGR(o__, fs[f])
            text = r'F$_*$=' + fs_str[f]
            ax.plot(o__, y, alpha=0.7, linewidth=1.0, color='grey', ls='--')
            n = 14
            if f == 2:
                plt.text(o__[n], y[n], text, color='black', alpha=0.7,
                         rotation=12.5, size=11)
            else:
                plt.text(o__[n], 0.9*y[n], text, color='black', alpha=0.7,
                         rotation=12.5, size=11)
        ax.plot([oH204] * 2, [3E-5, 6E-2], label=r'40% H$_2$')
        ax.annotate(r'f(H$_2$)=0.4', xy=(oH204, 3E-4), xytext=(8.25, 9E-5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1,
                                    headwidth=5))
        plt.text(8.52, 0.03, r'f(H$_2$)>0.4', color='black', size=11)
        plt.text(8.32, 0.03, r'f(H$_2$)<0.4', color='black', size=11)
        ax.set_yscale('log')
        ax.set_ylabel('DGR', size=12)
        ax.set_xlim(xlim_tight)
        ax.set_ylim(ylim)
        ax.set_xlabel('12 + log(O/H)', size=12)
        fig.tight_layout()
        fn = 'output/_DTM_J09_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def max_possible_DGR(self, O_p_H):
        return 10**(O_p_H - 12) * 16.0 / 1.008 / 0.4953 / 1.36

    def Jenkins_F_DGR(self, O_p_H, F):
        if F == 0:
            ratio = 0.3215690052
        elif F == 0.36:
            ratio = 0.5331653995
        elif F == 1:
            ratio = 0.7670911572
        elif F > 1:
            ratio = 1.373804396
        return ratio * 10**(O_p_H - 12) * 16.0 / 1.008

    def BPL_DGR(self, x, XCO):
        """
        R14 broken power law fitting results
        """
        a, alphaH = 2.21, 1.00
        if XCO == 'MW':
            b, alphaL, xt = 0.68, 3.08, 7.96 - solar_oxygen_bundance
        elif XCO == 'Z':
            b, alphaL, xt = 0.96, 3.10, 8.10 - solar_oxygen_bundance
        DGR = np.empty_like(x)
        for i in range(len(x)):
            if x[i] > xt:
                DGR[i] = 10**(alphaH * x[i] - a)
            else:
                DGR[i] = 10**(alphaL * x[i] - b)
        return DGR

    def temperature_profiles(self, name, method_abbr):
        plt.close('all')
        """
        1X1: Temperature profile
        """
        print(' --Plotting Temperature profile...')
        tbins = self.d[name]['WD']['Teff_bins']
        tbins2 = (tbins[:-1] + tbins[1:]) / 2
        if method_abbr in ['FBPT']:
            r, t = self.simple_profile(self.d[name]['FBPT']['aT'],
                                       self.d[name]['aRadius'],
                                       self.rbin,
                                       self.d[name]['aGas'])
            fig = plt.figure(figsize=(10, 7.5))
            plt.plot(r, t, 'r', label='Best fit')
        else:
            r = t = w = np.array([])  # radius, temperature, weight
            for i in range(len(self.d[name]['binlist'])):
                temp_G = self.d[name]['aGas'][i]
                temp_R = self.d[name]['aRadius'][i]
                mask = self.d[name][method_abbr]['aPDFs_T'][i] > \
                    self.d[name][method_abbr]['aPDFs_T'][i].max() / 1000
                temp_T = self.Ts[mask]
                temp_P = self.d[name][method_abbr]['aPDFs_T'][i][mask]
                temp_P = temp_P / np.sum(temp_P) * temp_G * \
                    (self.d[name]['binmap'] ==
                     self.d[name]['binlist'][i]).sum()
                r = np.append(r, [temp_R] * len(temp_P))
                t = np.append(t, temp_T)
                w = np.append(w, temp_P)
            nanmask = np.isnan(r + t + w)
            r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
            rbins = np.linspace(np.min(r), np.max(r), self.rbin)
            rbins2 = (rbins[:-1] + rbins[1:]) / 2
            # Counting hist2d...
            counts, _, _ = np.histogram2d(r, t, bins=(rbins, tbins), weights=w)
            del r, t, w
            counts = counts.T
            T_Exp = np.array([])
            T_Max, T_Median = np.array([]), np.array([])
            n_zeromask = np.full(counts.shape[1], True, dtype=bool)
            for i in range(counts.shape[1]):
                if np.sum(counts[:, i]) > 0:
                    counts[:, i] /= np.sum(counts[:, i])
                    csp = np.cumsum(counts[:, i])
                    csp = csp / csp[-1]
                    sst = np.interp([0.16, 0.5, 0.84], csp, tbins2)
                    T_Exp = np.append(T_Exp, np.sum(tbins2 * counts[:, i]))
                    T_Median = np.append(T_Median, sst[1])
                    T_Max = np.append(T_Max, tbins2[np.argmax(counts[:, i])])
                else:
                    n_zeromask[i] = False
            #
            fig = plt.figure(figsize=(10, 7.5))
            plt.pcolormesh(rbins, tbins, counts, norm=LogNorm(),
                           cmap=self.cmap1, vmin=1E-3)
            plt.colorbar()
            plt.plot(rbins2[n_zeromask], T_Median, 'r', label='Median')
            plt.plot(rbins2[n_zeromask], T_Exp, 'g', label='Exp')
            plt.plot(rbins2[n_zeromask], T_Max, 'b', label='Max')
        plt.ylim([5, 50])
        plt.xlabel(r'Radius ($R_{25}$)', size=16)
        plt.ylabel(r'T (K)', size=16)
        plt.legend(fontsize=16)
        plt.title(r'Gas mass weighted $T_d$', size=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if self.fake:
            fn = 'output/_FAKE_T-profile_' + name + '_' + method_abbr + '.pdf'
        else:
            fn = 'output/_T-profile_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def residual_maps(self, name, method_abbr):
        plt.close('all')
        print(' --Plotting residual maps of', method_abbr + '...')
        titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']
        rows, columns = len(titles), 2
        fig, ax = plt.subplots(rows, columns, figsize=(6, 12))
        yranges = [-0.3, 0.5]
        xranges = [[-0.5, 2.5], [0.0, 2.5], [0.0, 2.5], [-0.5, 2.0],
                   [-0.5, 1.5]]
        for i in range(5):
            data = self.d[name]['aSED'][:, i]
            Res_d_data = \
                (data - self.d[name][method_abbr]['aSED'][:, i]) / data
            with np.errstate(invalid='ignore'):
                logdata = np.log10(data)
            nonnanmask = ~np.isnan(logdata)
            logdata, Res_d_data = logdata[nonnanmask], Res_d_data[nonnanmask]
            ax[i, 0].hist2d(logdata, Res_d_data, cmap='Blues',
                            range=[xranges[i], yranges], bins=[50, 50])
            ax[i, 0].set_ylim(yranges)
            ax[i, 0].set_ylabel('(Obs. - Fit) / Obs.')
            ax[i, 0].set_title(titles[i], x=0.3, y=0.8, size=12)
            ax[i, 0].minorticks_on()
            ax[i, 1].hist(Res_d_data, orientation='horizontal',
                          range=yranges, bins=50)
            ax[i, 1].set_ylim(yranges)
            ax[i, 1].set_title(titles[i], x=0.3, y=0.8, size=12)
            ax[i, 1].minorticks_on()
        ax[4, 1].set_xlabel('Count')
        ax[4, 0].set_xlabel(r'$\log$(Obs.)')
        fig.tight_layout()
        if self.fake:
            fn = 'output/_FAKE_Residual_' + name + '_' + method_abbr + '.pdf'
        else:
            fn = 'output/_Residual_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def residual_map_merged(self, name='NGC5457'):
        plt.close('all')
        titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']
        ms = ['SE', 'FB', 'BE', 'WD', 'PL']
        rows, columns = len(titles), 5
        fig, ax = plt.subplots(rows, columns, figsize=(12, 12))
        yrange = [-0.3, 0.5]
        xranges = [[0.0, 2.5], [0.5, 2.5], [0.5, 2.0], [0.0, 1.5], [-0.5, 1.0]]
        for m in range(5):
            method_abbr = ms[m]
            for i in range(5):
                data = self.d[name]['aSED'][:, i]
                Res_d_data = \
                    (data - self.d[name][method_abbr]['aSED'][:, i]) / data
                with np.errstate(invalid='ignore'):
                    logdata = np.log10(data)
                nonnanmask = ~np.isnan(logdata)
                logdata = logdata[nonnanmask]
                Res_d_data = Res_d_data[nonnanmask]
                ax[m, i].hist2d(logdata, Res_d_data, cmap='Blues',
                                range=[xranges[i], yrange], bins=[50, 50])
                ax[m, i].plot(xranges[i], [0.0] * 2, 'k', alpha=0.3)
                ax[m, i].set_ylim(yrange)
                ax[m, i].set_xlim(xranges[i])
                if i == 0:
                    ax[m, i].set_ylabel('(Obs. - Fit) / Obs.', size=14)
                else:
                    ax[m, i].set_yticklabels([])
                if m < 4:
                    ax[m, i].set_xticklabels([])
                ax[m, i].text(0.1, 0.85, titles[i], size=14,
                              horizontalalignment='left',
                              transform=ax[m, i].transAxes)
                ax[m, i].text(0.9, 0.85, '(' + ms[m] + ')', size=14,
                              horizontalalignment='right',
                              transform=ax[m, i].transAxes)
                ax[m, i].minorticks_on()
                ax[4, i].set_xlabel(r'$\log$(Obs.)', size=14)
        fig.tight_layout()
        fn = 'output/_Residual_' + name + '_merged.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def example_model(self, name, method_abbr):
        i = np.argmax(self.d[name]['binlist'] ==
                      self.d[name]['binmap'][self.y, self.x])
        print(' --Plotting example model...')
        wl_complete = np.linspace(1, 800, 1000)
        wl_plot = np.linspace(51, 549, 100)
        with File('hdf5_MBBDust/Models.h5', 'r') as hf:
            models = hf[method_abbr].value
        if method_abbr in ['FBPT']:
            models = \
                models[self.FBPT_Ts == self.d[name][method_abbr]['aT'][i]][0]
        elif method_abbr in ['PB']:
            models = \
                models[self.PB_betas == self.d[name][method_abbr]['aBeta'][i]
                       ][0]
        sed = self.d[name]['aSED'][i]
        cov_n1 = self.d[name]['acov_n1'][i]
        unc = np.sqrt(np.linalg.inv(cov_n1).diagonal())
        kappa160 = self.d['kappa160'][method_abbr]
        SigmaD = 10**self.d[name][method_abbr]['alogSigmaD'][i]
        if method_abbr not in ['PL']:
            T = self.d[name][method_abbr]['aT'][i]
            Beta = self.d[name][method_abbr]['aBeta'][i]
        #
        # Colour correction factors
        #
        if method_abbr in ['SE', 'FB', 'FBPT', 'PB']:
            model_complete = SEMBB(wl_complete, SigmaD, T, Beta,
                                   kappa160=kappa160)
            ccf = SEMBB(wl, SigmaD, T, Beta, kappa160=kappa160) / \
                z0mg_RSRF(wl_complete, model_complete, bands)
            sed_best_plot = SEMBB(wl_plot, SigmaD, T, Beta, kappa160=kappa160)
        elif method_abbr in ['BEMFB', 'BE']:
            lambda_c = self.d[name][method_abbr]['alambda_c'][i]
            beta2 = self.d[name][method_abbr]['abeta2'][i]
            model_complete = \
                BEMBB(wl_complete, SigmaD, T, Beta, lambda_c, beta2,
                      kappa160=kappa160)
            ccf = BEMBB(wl, SigmaD, T, Beta, lambda_c, beta2,
                        kappa160=kappa160) / \
                z0mg_RSRF(wl_complete, model_complete, bands)
            sed_best_plot = BEMBB(wl_plot, SigmaD, T, Beta, lambda_c, beta2,
                                  kappa160=kappa160)
        elif method_abbr in ['WD']:
            WDfrac = self.d[name][method_abbr]['aWDfrac'][i]
            model_complete = \
                WD(wl_complete, SigmaD, T, Beta, WDfrac, kappa160=kappa160)
            ccf = WD(wl, SigmaD, T, Beta, WDfrac, kappa160=kappa160) / \
                z0mg_RSRF(wl_complete, model_complete, bands)
            sed_best_plot = WD(wl_plot, SigmaD, T, Beta, WDfrac,
                               kappa160=kappa160)
        elif method_abbr in ['PL']:
            alpha = self.d[name][method_abbr]['aalpha'][i]
            gamma = 10**self.d[name][method_abbr]['aloggamma'][i]
            logUmin = self.d[name][method_abbr]['alogUmin'][i]
            model_complete = \
                PowerLaw(wl_complete, SigmaD, alpha, gamma, logUmin,
                         kappa160=kappa160)
            ccf = PowerLaw(wl, SigmaD, alpha, gamma, logUmin,
                           kappa160=kappa160) / \
                z0mg_RSRF(wl_complete, model_complete, bands)
            sed_best_plot = PowerLaw(wl_plot, SigmaD, alpha, gamma, logUmin,
                                     kappa160=kappa160)
        sed_obs_plot = sed * ccf
        unc_obs_plot = unc * ccf
        #
        # Begin fitting
        #
        if method_abbr == 'SE':
            Sigmas, Ts, Betas = np.meshgrid(self.SigmaDs, self.Ts, self.betas)
        elif method_abbr == 'FB':
            Sigmas, Ts = np.meshgrid(self.SigmaDs, self.Ts)
            Betas = np.full(Ts.shape, Beta)
        elif method_abbr == 'FBPT':
            Sigmas = self.SigmaDs
            Ts, Betas = np.full(Sigmas.shape, T), np.full(Sigmas.shape, Beta)
        elif method_abbr == 'PB':
            Ts, Sigmas = np.meshgrid(self.Ts, self.SigmaDs)
            Betas = np.full(Ts.shape, Beta)
        elif method_abbr == 'BEMFB':
            Sigmas, Ts, lambda_cs, beta2s = \
                np.meshgrid(self.SigmaDs, self.Ts, self.lambda_cs,
                            self.beta2s)
        elif method_abbr == 'BE':
            Sigmas, Ts, beta2s = \
                np.meshgrid(self.SigmaDs, self.Ts, self.beta2s)
            lambda_cs = np.full(Ts.shape, lambda_c)
            Betas = np.full(Ts.shape, Beta)
        elif method_abbr == 'WD':
            Sigmas, Ts, WDfracs = \
                np.meshgrid(self.SigmaDs, self.Ts, self.WDfracs)
            Betas = np.full(Ts.shape, Beta)
        elif method_abbr == 'PL':
            Sigmas, alphas, gammas, logUmins = \
                np.meshgrid(self.SigmaDs, self.alphas, self.gammas,
                            self.logUmins)
        diff = (models - sed)
        temp_matrix = np.empty_like(diff)
        for j in range(5):
            temp_matrix[..., j] = np.sum(diff * cov_n1[:, j], axis=-1)
        chi2 = np.sum(temp_matrix * diff, axis=-1)
        del temp_matrix, diff
        #
        # Selecting samples for plotting
        #
        chi2_threshold = 3.0
        chi2 -= np.nanmin(chi2)
        mask = chi2 < chi2_threshold
        if method_abbr not in ['PL']:
            chi2, Sigmas, Ts, Betas = \
                chi2[mask], Sigmas[mask], Ts[mask], Betas[mask]
        else:
            chi2, Sigmas, alphas, gammas, logUmins = \
                chi2[mask], Sigmas[mask], alphas[mask], gammas[mask], \
                logUmins[mask]
        if method_abbr in ['BEMFB', 'BE']:
            lambda_cs, beta2s = lambda_cs[mask], beta2s[mask]
        elif method_abbr in ['WD']:
            WDfracs = WDfracs[mask]
        num = 100
        if len(Sigmas) > num:
            mask = np.array([True] * num + [False] * (len(Sigmas) - num))
            np.random.shuffle(mask)
            if method_abbr not in ['PL']:
                chi2, Sigmas, Ts, Betas = \
                    chi2[mask], Sigmas[mask], Ts[mask], Betas[mask]
            else:
                chi2, Sigmas, alphas, gammas, logUmins = \
                    chi2[mask], Sigmas[mask], alphas[mask], gammas[mask], \
                    logUmins[mask]
            if method_abbr in ['BEMFB', 'BE']:
                lambda_cs, beta2s = lambda_cs[mask], beta2s[mask]
            elif method_abbr in ['WD']:
                WDfracs = WDfracs[mask]
        transp = np.exp(-0.5 * chi2) * 0.2
        #
        # Begin plotting
        #
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.set_ylim([0.0, np.nanmax(sed_best_plot) * 1.2])
        Sigmas0122, alphas0122, gammas0122, logUmins0122 = [], [], [], []
        for j in range(len(Sigmas)):
            if method_abbr in ['SE', 'FB', 'FBPT', 'PB']:
                model_plot = SEMBB(wl_plot, Sigmas[j], Ts[j], Betas[j],
                                   kappa160=kappa160)
            elif method_abbr in ['BEMFB', 'BE']:
                model_plot = BEMBB(wl_plot, Sigmas[j], Ts[j], Betas[j],
                                   lambda_cs[j], beta2s[j], kappa160=kappa160)
            elif method_abbr in ['WD']:
                model_plot = WD(wl_plot, Sigmas[j], Ts[j], Betas[j],
                                WDfracs[j], kappa160=kappa160)
            elif method_abbr in ['PL']:
                model_plot = PowerLaw(wl_plot, Sigmas[j], alphas[j], gammas[j],
                                      logUmins[j], kappa160=kappa160)
                if model_plot[0] > 13:
                    Sigmas0122.append(Sigmas[j])
                    alphas0122.append(alphas[j])
                    gammas0122.append(gammas[j])
                    logUmins0122.append(logUmins[j])
            ax.plot(wl_plot, model_plot, alpha=transp[j], color='k')
        ax.plot(wl_plot, sed_best_plot, linewidth=3,
                label=method_abbr + ' best fit')
        ax.errorbar(wl, sed_obs_plot, yerr=unc_obs_plot, fmt='o',
                    color='red', capsize=10, label='Herschel data')
        ax.legend(fontsize=12)
        ax.set_xlabel(r'Wavelength ($\mu m$)', size=12)
        ax.set_ylabel(r'SED ($MJy$ $sr^{-1}$)', size=12)
        fig.tight_layout()
        if self.fake:
            fn = 'output/_FAKE_Model_' + str(self.x) + str(self.y) + '_' + \
                name + '_' + method_abbr + '.pdf'
        else:
            fn = 'output/_Model_' + str(self.x) + str(self.y) + '_' + \
                name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def corner_plots(self, name, method_abbr):
        plt.close('all')
        with np.errstate(invalid='ignore'):
            pacs100 = np.log10(self.d[name]['aSED'][:, 0])
            mask = ~np.isnan(pacs100)
        print(' --Plotting corner plot...')
        if method_abbr == 'SE':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['aBeta'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$T_d$', r'$\beta$']
        elif method_abbr == 'FB':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$T_d$']
        elif method_abbr == 'BE':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['abeta2'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$T_d$', r'$\beta_2$']
        elif method_abbr == 'WD':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['aWDfrac'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$T_d$', r'$f_w$']
        elif method_abbr == 'PL':
            temp = self.d[name][method_abbr]['aalpha'][mask]
            temp[0] += 0.01
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                temp,
                                self.d[name][method_abbr]['aloggamma'][mask],
                                self.d[name][method_abbr]['alogUmin'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$\alpha$', r'$\log(\gamma)$',
                      r'$\log(U)_{min}$']
        mask2 = np.sum(~np.isnan(samples), axis=0).astype(bool)
        fig = corner(samples.T[mask2], labels=labels, quantities=(0.16, 0.84),
                     show_titles=True, title_kwargs={"fontsize": 16},
                     label_kwargs={"fontsize": 16})
        if self.fake:
            fn = 'output/_FAKE_Corner_' + name + '_' + method_abbr + '.pdf'
        else:
            fn = 'output/_Corner_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def simple_profile(self, data, radius, bins, gas=None, gas_weighted=True):
        gas = np.ones(data.shape) if not (gas_weighted or gas) else gas
        n_nanmask = ~np.isnan(gas + radius + data)
        gas, radius, data = gas[n_nanmask], radius[n_nanmask], data[n_nanmask]
        r, profile = [], []
        rbins = np.linspace(radius.min(), radius.max(), bins)
        for i in range(bins - 1):
            mask = (rbins[i] <= radius) * (radius < rbins[i + 1])
            r.append((rbins[i] + rbins[i + 1]) / 2)
            with np.errstate(invalid='ignore'):
                profile.append(np.sum(data[mask] * gas[mask]) /
                               np.sum(gas[mask]))
        return np.array(r), np.array(profile)

    def SFR_and_starlight(self, name):
        plt.close('all')
        """
        1X1: SFSR & SMSD profile
        """
        print(' --Plotting heating source profile...')
        R_SFR, SFR_profile = self.simple_profile(self.d[name]['SFR'],
                                                 self.d[name]['Radius'],
                                                 self.rbin,
                                                 self.d[name]['SigmaGas'])
        R_SMSD, SMSD_profile = self.simple_profile(self.d[name]['SMSD'],
                                                   self.d[name]['Radius'],
                                                   self.rbin,
                                                   self.d[name]['SigmaGas'])
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.semilogy(R_SFR, SFR_profile, 'k')
        ax.set_xlabel(r'Radius ($r_{25}$)', size=16)
        ax.set_ylabel(r'$\Sigma_{SFR}$ ($M_\odot kpc^{-2} yr^{-1}$)',
                      size=16, color='k')
        ax.tick_params('y', colors='k')
        ax2 = ax.twinx()
        ax2.semilogy(R_SMSD, SMSD_profile, c='b')
        ax2.set_ylabel(r'$\Sigma_*$ ($M_\odot pc^{-2}$)', size=16, color='b')
        ax2.tick_params('y', colors='b')
        fig.tight_layout()
        if self.fake:
            fn = 'output/_FAKE_T-profile_SFRSMSD_' + name + '.pdf'
        else:
            fn = 'output/_T-profile_SFRSMSD_' + name + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def corr(self, name, method_abbr):
        plt.close('all')
        print(' --Plotting WDfrac vs. parameters')
        df = pd.DataFrame()
        df['WDfrac'] = self.d[name][method_abbr]['WDfrac'].flatten()
        df[r'$\log(\Sigma_d)$'] = \
            self.d[name][method_abbr]['logSigmaD'].flatten()
        with np.errstate(invalid='ignore'):
            df[r'$\log(\Sigma_{SFR})$'] = \
                np.log10(self.d[name]['SFR'].flatten())
            df[r'$\log(\Sigma_*)$'] = np.log10(self.d[name]['SMSD'].flatten())
            df[r'$\log(\Sigma_g)$'] = \
                np.log10(self.d[name]['SigmaGas'].flatten())
        fig = plt.figure()
        sns.heatmap(df.corr(), annot=True, cmap='Reds')
        fn = 'output/_corr_WDfrac_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        #
        print(' --Plotting corr of Herschel bands')
        df = pd.DataFrame()
        with np.errstate(invalid='ignore'):
            df['PACS100'] = np.log10(self.d[name]['aSED'][:, 0])
            df['PACS160'] = np.log10(self.d[name]['aSED'][:, 1])
            df['SPIRE250'] = np.log10(self.d[name]['aSED'][:, 2])
            df['SPIRE350'] = np.log10(self.d[name]['aSED'][:, 3])
            df['SPIRE500'] = np.log10(self.d[name]['aSED'][:, 4])
        fig = plt.figure()
        sns.heatmap(df.corr(), annot=True, cmap='Reds')
        fn = 'output/_corr_Herschel_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def residual_chi2(self, name):
        plt.close('all')
        print(' --Plotting chi2 vs observations...')
        DoFs = [2, 3, 2, 2, 1]
        method_abbrs = ['SE',
                        'FB',
                        'BE',
                        'WD',
                        'PL']
        rows, columns = len(method_abbrs), 2
        fig, ax = plt.subplots(rows, columns, figsize=(6, 12))
        yrange = [0, 6.0]
        y = np.linspace(0, 6.0)
        xranges = [[-0.5, 2.5], [0.0, 1.0]]
        with np.errstate(invalid='ignore'):
            logPACS100 = np.log10(self.d[name]['aSED'][:, 0])
        for i in range(len(method_abbrs)):
            c2 = self.d[name][method_abbrs[i]]['archi2']
            avg_c2 = round(np.nanmean(c2), 2)
            if self.fake:
                with np.errstate(invalid='ignore'):
                    logPACS100 = \
                        np.log10(self.d[name][method_abbrs[i]]['aSED'][:, 0])
            nonnanmask = ~np.isnan(logPACS100 + c2)
            ax[i, 0].hist2d(logPACS100[nonnanmask],
                            c2[nonnanmask], cmap='Blues',
                            range=[xranges[0], yrange], bins=[50, 50])
            ax[i, 0].set_ylabel(r'$\tilde{\chi}^2$', size=12)
            ax[i, 0].set_title(method_abbrs[i] + ' (DoF=' + str(DoFs[i]) + ')',
                               x=0.3, y=0.8, size=12)
            ax[i, 0].minorticks_on()
            ax[i, 1].hist(c2, orientation='horizontal',
                          range=yrange, bins=50, normed=True,
                          label='Mean: ' + str(avg_c2))
            ax[i, 1].plot(stats.chi2.pdf(y * DoFs[i], DoFs[i]) * DoFs[i], y,
                          label=r'$\tilde{\chi}^2$ dist. (k=' +
                          str(DoFs[i]) + ')')
            ax[i, 1].legend(fontsize=12)
            ax[i, 1].set_ylim(yrange)
            ax[i, 1].minorticks_on()
        ax[4, 0].set_xlabel(r'$\log$(PACS100)', size=12)
        ax[4, 1].set_xlabel('Frequency', size=12)
        fig.tight_layout()
        if self.fake:
            fn = 'output/_FAKE_Residual_Chi2_' + name + '.pdf'
        else:
            fn = 'output/_Residual_Chi2_' + name + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def pdf_profiles_merge(self, name, BF_method='exp'):
        plt.close('all')
        """
        1X1: DGR profile
        """
        print(' --Plotting merged DGR vs. Metallicity...')
        method_abbrs = ['SE', 'FB', 'BE', 'WD', 'PL']
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        DGRs = {}
        DGR16s, DGR84s = {}, {}
        n_zeromasks = {}
        for method_abbr in method_abbrs:
            r = d = w = np.array([])  # radius, dgr, weight
            for i in range(len(self.d[name]['binlist'])):
                temp_G = self.d[name]['aGas'][i]
                temp_R = self.d[name]['aRadius'][i]
                mask = self.d[name][method_abbr]['aPDFs'][i] > \
                    self.d[name][method_abbr]['aPDFs'][i].max() / 1000
                temp_DGR = self.SigmaDs[mask] / temp_G
                temp_P = self.d[name][method_abbr]['aPDFs'][i][mask]
                temp_P = temp_P / np.sum(temp_P) * temp_G * \
                    (self.d[name]['binmap'] ==
                     self.d[name]['binlist'][i]).sum()
                r = np.append(r, [temp_R] * len(temp_P))
                d = np.append(d, temp_DGR)
                w = np.append(w, temp_P)
            nanmask = np.isnan(r + d + w)
            r, d, w = r[~nanmask], d[~nanmask], w[~nanmask]
            rbins = np.linspace(np.min(r), np.max(r), self.rbin)
            dbins = \
                np.logspace(np.min(np.log10(d)), np.max(np.log10(d)),
                            self.dbin)
            # Counting hist2d...
            counts, _, _ = np.histogram2d(r, d, bins=(rbins, dbins), weights=w)
            del r, d, w
            counts = counts.T
            n_zeromask = np.full(counts.shape[1], True, dtype=bool)
            dbins2 = np.sqrt(dbins[:-1] + dbins[1:])
            rbins2 = (rbins[:-1] + rbins[1:]) / 2
            DGR_Median = DGR_LExp = DGR_Max = DGR_16 = DGR_84 = np.array([])
            for i in range(counts.shape[1]):
                if np.sum(counts[:, i]) > 0:
                    counts[:, i] /= np.sum(counts[:, i])
                    csp = np.cumsum(counts[:, i])
                    csp = csp / csp[-1]
                    ssd = np.interp([0.16, 0.5, 0.84], csp, np.log10(dbins2))
                    DGR_Median = np.append(DGR_Median, 10**ssd[1])
                    DGR_LExp = np.append(DGR_LExp,
                                         10**np.sum(np.log10(dbins2) *
                                                    counts[:, i]))
                    DGR_Max = np.append(DGR_Max,
                                        dbins2[np.argmax(counts[:, i])])
                    DGR_16 = np.append(DGR_16, 10**ssd[0])
                    DGR_84 = np.append(DGR_84, 10**ssd[2])
                else:
                    n_zeromask[i] = False
            DGR16s[method_abbr] = DGR_16
            DGR84s[method_abbr] = DGR_84
            n_zeromasks[method_abbr] = n_zeromask
            if BF_method == 'exp':
                DGRs[method_abbr] = DGR_LExp
            elif BF_method == 'max':
                DGRs[method_abbr] = DGR_Max
            elif BF_method == 'median':
                DGRs[method_abbr] = DGR_Median
            #
        # My DGR gradient with Remy-Ruyer data and various models
        fs1 = 24
        fs2 = 20
        xbins2 = (8.715 - 0.027 * rbins2 * self.d[name]['R25'] *
                  7.4 / GD_dist)
        df = pd.read_csv("data/Tables/Remy-Ruyer_2014.csv")
        r_ = (8.715 - df['12+log(O/H)'].values) / 0.027 * GD_dist / 7.4 / \
            self.d[name]['R25']
        r__ = np.linspace(np.nanmin(r_), np.nanmax(r_), 50)
        x__ = (8.715 - 0.027 * r__ * self.d[name]['R25'] * 7.4 / GD_dist -
               solar_oxygen_bundance)
        o__ = x__ + solar_oxygen_bundance
        # New uncertainty
        DGR_unc_dex = np.zeros_like(rbins2)
        # second part: gas zero point uncertainty
        radius_map = np.zeros_like(self.d[name]['binmap']) * np.nan
        gas_map = np.zeros_like(self.d[name]['binmap']) * np.nan
        for i in range(len(self.d[name]['binlist'])):
            b_mask = self.d[name]['binmap'] == self.d[name]['binlist'][i]
            radius_map[b_mask] = self.d[name]['aRadius'][i]
            gas_map[b_mask] = self.d[name]['aGas'][i]
        for i in range(len(DGR_unc_dex)):
            u_mask = (rbins[i] <= radius_map) * (rbins[i + 1] >= radius_map)
            temp_gas = np.nanmean(gas_map[u_mask])
            DGR_unc_dex[i] += np.log10((temp_gas + 1) / temp_gas)
        DGR_unc_mtp = 10**DGR_unc_dex
        #
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.set_xlim([np.nanmin(xbins2), np.nanmax(xbins2)])
        for i in range(len(DGRs.keys())):
            k = list(DGRs.keys())[i]
            xbins2 = (8.715 - 0.027 * rbins2[n_zeromasks[k]] *
                      self.d[name]['R25'] * 7.4 / GD_dist)
            ax.plot(xbins2, DGRs[k], label=k, linewidth=4.0,
                    alpha=0.8)
            ax.fill_between(xbins2, DGR16s[k] / DGR_unc_mtp,
                            DGR84s[k] * DGR_unc_mtp, alpha=0.13)
        ax.fill_between(o__, 10**(o__ - 12) * 16.0 / 1.008 / 0.51 / 1.36,
                        10**(o__ - 12) * 16.0 / 1.008 / 0.445 / 1.36,
                        alpha=0.7, label='MAX', hatch='/')
        ax.legend(fontsize=fs2, ncol=2, loc=4)
        ax.tick_params(axis='both', labelsize=fs2)
        ax.set_yscale('log')
        ax.set_xlabel('12 + log(O/H)', size=fs1)
        ax.set_ylabel('DGR (' + BF_method + ')', size=fs1)
        ax2 = ax.twiny()
        ax2.set_xlabel('Radius (kpc)', size=fs1, color='k')
        ax2.set_xlim([np.nanmax(rbins2) * self.d[name]['R25'],
                      np.nanmin(rbins2) * self.d[name]['R25']])
        ax2.tick_params(axis='both', labelsize=fs2)
        fig.tight_layout()
        fn = 'output/_DGR-vs-Metallicity_' + name + '_z_merged.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def pdf_profiles_talk(self, name, BF_method='exp'):
        plt.close('all')
        """
        1X1: DGR profile
        """
        print(' --Plotting merged DGR vs. Metallicity...')
        method_abbrs = ['SE', 'FB', 'BE', 'WD', 'PL']
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        DGRs = {}
        DGR16s, DGR84s = {}, {}
        n_zeromasks = {}
        for method_abbr in method_abbrs:
            r = d = w = np.array([])  # radius, dgr, weight
            for i in range(len(self.d[name]['binlist'])):
                temp_G = self.d[name]['aGas'][i]
                temp_R = self.d[name]['aRadius'][i]
                mask = self.d[name][method_abbr]['aPDFs'][i] > \
                    self.d[name][method_abbr]['aPDFs'][i].max() / 1000
                temp_DGR = self.SigmaDs[mask] / temp_G
                temp_P = self.d[name][method_abbr]['aPDFs'][i][mask]
                temp_P = temp_P / np.sum(temp_P) * temp_G * \
                    (self.d[name]['binmap'] ==
                     self.d[name]['binlist'][i]).sum()
                r = np.append(r, [temp_R] * len(temp_P))
                d = np.append(d, temp_DGR)
                w = np.append(w, temp_P)
            nanmask = np.isnan(r + d + w)
            r, d, w = r[~nanmask], d[~nanmask], w[~nanmask]
            rbins = np.linspace(np.min(r), np.max(r), self.rbin)
            dbins = \
                np.logspace(np.min(np.log10(d)), np.max(np.log10(d)),
                            self.dbin)
            # Counting hist2d...
            counts, _, _ = np.histogram2d(r, d, bins=(rbins, dbins), weights=w)
            del r, d, w
            counts = counts.T
            n_zeromask = np.full(counts.shape[1], True, dtype=bool)
            dbins2 = np.sqrt(dbins[:-1] * dbins[1:])
            rbins2 = (rbins[:-1] + rbins[1:]) / 2
            DGR_Median = DGR_LExp = DGR_Max = DGR_16 = DGR_84 = np.array([])
            for i in range(counts.shape[1]):
                if np.sum(counts[:, i]) > 0:
                    counts[:, i] /= np.sum(counts[:, i])
                    csp = np.cumsum(counts[:, i])
                    csp = csp / csp[-1]
                    ssd = np.interp([0.16, 0.5, 0.84], csp, np.log10(dbins2))
                    DGR_Median = np.append(DGR_Median, 10**ssd[1])
                    DGR_LExp = np.append(DGR_LExp,
                                         10**np.sum(np.log10(dbins2) *
                                                    counts[:, i]))
                    DGR_Max = np.append(DGR_Max,
                                        dbins2[np.argmax(counts[:, i])])
                    DGR_16 = np.append(DGR_16, 10**ssd[0])
                    DGR_84 = np.append(DGR_84, 10**ssd[2])
                else:
                    n_zeromask[i] = False
            DGR16s[method_abbr] = DGR_16
            DGR84s[method_abbr] = DGR_84
            n_zeromasks[method_abbr] = n_zeromask
            if BF_method == 'exp':
                DGRs[method_abbr] = DGR_LExp
            elif BF_method == 'max':
                DGRs[method_abbr] = DGR_Max
            elif BF_method == 'median':
                DGRs[method_abbr] = DGR_Median
            #
        # My DGR gradient with Remy-Ruyer data and various models
        fs1 = 24
        fs2 = 20
        xbins2 = (8.715 - 0.027 * rbins2 * self.d[name]['R25'] *
                  7.4 / GD_dist)
        df = pd.read_csv("data/Tables/Remy-Ruyer_2014.csv")
        r_ = (8.715 - df['12+log(O/H)'].values) / 0.027 * GD_dist / 7.4 / \
            self.d[name]['R25']
        r__ = np.linspace(np.nanmin(r_), np.nanmax(r_), 50)
        x__ = (8.715 - 0.027 * r__ * self.d[name]['R25'] * 7.4 / GD_dist -
               solar_oxygen_bundance)
        o__ = x__ + solar_oxygen_bundance
        # New uncertainty
        DGR_unc_dex = np.zeros_like(rbins2)
        # second part: gas zero point uncertainty
        radius_map = np.zeros_like(self.d[name]['binmap']) * np.nan
        gas_map = np.zeros_like(self.d[name]['binmap']) * np.nan
        for i in range(len(self.d[name]['binlist'])):
            b_mask = self.d[name]['binmap'] == self.d[name]['binlist'][i]
            radius_map[b_mask] = self.d[name]['aRadius'][i]
            gas_map[b_mask] = self.d[name]['aGas'][i]
        for i in range(len(DGR_unc_dex)):
            u_mask = (rbins[i] <= radius_map) * (rbins[i + 1] >= radius_map)
            temp_gas = np.nanmean(gas_map[u_mask])
            DGR_unc_dex[i] += np.log10((temp_gas + 1) / temp_gas)
        DGR_unc_mtp = 10**DGR_unc_dex
        #
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.set_xlim([np.nanmin(xbins2), np.nanmax(xbins2)])
        for i in range(len(DGRs.keys())):
            k = list(DGRs.keys())[i]
            label = k if k != 'SE' else 'MBB'
            xbins2 = (8.715 - 0.027 * rbins2[n_zeromasks[k]] *
                      self.d[name]['R25'] * 7.4 / GD_dist)
            ax.plot(xbins2, DGRs[k], label=label, linewidth=4.0,
                    alpha=0.8)
            ax.fill_between(xbins2, DGR16s[k] / DGR_unc_mtp,
                            DGR84s[k] * DGR_unc_mtp, alpha=0.13)
        ax.fill_between(o__, 10**(o__ - 12) * 16.0 / 1.008 / 0.51 / 1.36,
                        10**(o__ - 12) * 16.0 / 1.008 / 0.445 / 1.36,
                        alpha=0.7, label='MAX', hatch='/')
        ax.legend(fontsize=fs2, ncol=2, loc=4)
        ax.tick_params(axis='both', labelsize=fs2)
        ax.set_yscale('log')
        ax.set_xlabel('12 + log(O/H)', size=fs1)
        ax.set_ylabel('DGR (' + BF_method + ')', size=fs1)
        ax2 = ax.twiny()
        ax2.set_xlabel('Radius (kpc)', size=fs1, color='k')
        ax2.set_xlim([np.nanmax(rbins2) * self.d[name]['R25'],
                      np.nanmin(rbins2) * self.d[name]['R25']])
        ax2.tick_params(axis='both', labelsize=fs2)
        fig.tight_layout()
        fn = 'output/_DGR-vs-Metallicity_' + name + '_z_talk.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def temperature_profiles_merge(self, name):
        plt.close('all')
        """
         Temperature profile
        """
        print(' --Plotting Temperature profile (merged)...')
        method_abbrs = ['SE', 'FB', 'BE', 'WD', 'PL']
        rs, Ts, TMaxs, TMins = {}, {}, {}, {}
        tbins = self.d[name]['WD']['Teff_bins']
        tbins2 = (tbins[:-1] + tbins[1:]) / 2
        for method_abbr in method_abbrs:
            r = t = w = np.array([])  # radius, temperature, weight
            for i in range(len(self.d[name]['binlist'])):
                temp_G = self.d[name]['aGas'][i]
                temp_R = self.d[name]['aRadius'][i]
                mask = self.d[name][method_abbr]['aPDFs_T'][i] > \
                    self.d[name][method_abbr]['aPDFs_T'][i].max() / 1000
                temp_T = self.Ts[mask]
                temp_P = self.d[name][method_abbr]['aPDFs_T'][i][mask]
                temp_P = temp_P / np.sum(temp_P) * temp_G * \
                    (self.d[name]['binmap'] ==
                     self.d[name]['binlist'][i]).sum()
                r = np.append(r, [temp_R] * len(temp_P))
                t = np.append(t, temp_T)
                w = np.append(w, temp_P)
            nanmask = np.isnan(r + t + w)
            r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
            rbins = np.linspace(np.min(r), np.max(r), self.rbin)
            rbins2 = (rbins[:-1] + rbins[1:]) / 2
            # Counting hist2d...
            counts, _, _ = np.histogram2d(r, t, bins=(rbins, tbins), weights=w)
            del r, t, w
            counts = counts.T
            T_Exp, T_Max, T_Min = np.array([]), np.array([]), np.array([])
            n_zeromask = np.full(counts.shape[1], True, dtype=bool)
            for i in range(counts.shape[1]):
                if np.sum(counts[:, i]) > 0:
                    counts[:, i] /= np.sum(counts[:, i])
                    csp = np.cumsum(counts[:, i])
                    csp = csp / csp[-1]
                    sst = np.interp([0.16, 0.5, 0.84], csp, tbins2)
                    T_Exp = np.append(T_Exp, np.sum(tbins2 * counts[:, i]))
                    T_Max = np.append(T_Max, sst[2])
                    T_Min = np.append(T_Min, sst[0])
                else:
                    n_zeromask[i] = False
            Ts[method_abbr] = T_Exp
            rs[method_abbr] = rbins2[n_zeromask]
            TMaxs[method_abbr] = T_Max
            TMins[method_abbr] = T_Min
        fs1 = 18
        fs2 = 15
        xlim = [np.nanmin(rs['BE']), np.nanmax(rs['BE'])]
        fig, ax = plt.subplots(2, 1, figsize=(6, 7.5),
                               gridspec_kw={'wspace': 0, 'hspace': 0})
        for k in Ts.keys():
            ax[0].plot(rs[k], Ts[k], alpha=0.8, label=k, linewidth=2.0)
            ax[0].fill_between(rs[k], TMins[k], TMaxs[k], alpha=0.13)
        # ax[0].set_xlabel(r'Radius ($R_{25}$)', size=fs1)
        ax[0].set_ylabel(r'T$_d$ (K)', size=fs1)
        ax[0].set_xlim(xlim)
        ax[0].legend(fontsize=fs2, ncol=2, loc=2)
        # ax[0].set_title(r'$T_d$ profile', size=fs1, loc='left', y=0.85)
        ax[0].set_yticks([10, 20, 30, 40])
        ax[0].set_yticklabels([10, 20, 30, 40], fontsize=fs2)
        #
        R_SFR, SFR_profile = self.simple_profile(self.d[name]['SFR'],
                                                 self.d[name]['Radius'],
                                                 self.rbin,
                                                 self.d[name]['SigmaGas'])
        R_SMSD, SMSD_profile = self.simple_profile(self.d[name]['SMSD'],
                                                   self.d[name]['Radius'],
                                                   self.rbin,
                                                   self.d[name]['SigmaGas'])
        #
        ax2 = ax[1].twinx()
        ax[1].plot(R_SMSD, SMSD_profile, c='b')
        ax[1].fill_between(R_SMSD, SMSD_profile * 0.9, SMSD_profile * 1.1,
                           alpha=0.2, color='b')
        ax[1].tick_params('y', colors='b')
        ax2.plot(R_SFR, SFR_profile, 'r')
        ax2.fill_between(R_SFR, SFR_profile * 0.9, SFR_profile * 1.1,
                         alpha=0.2, color='r')
        ax2.set_ylabel(r'$\Sigma_{SFR}$ ($M_\odot kpc^{-2} yr^{-1}$)',
                       size=fs1, color='r')
        ax2.tick_params('y', colors='r')
        ax[1].set_xlim(xlim)
        ax[1].set_xlabel(r'Radius ($r_{25}$)', size=fs1)
        ax[1].set_ylabel(r'$\Sigma_*$ ($M_\odot pc^{-2}$)', size=fs1,
                         color='b')
        ax[1].tick_params(axis='both', labelsize=fs2)
        ax2.tick_params(axis='y', labelsize=fs2)
        ax[1].set_yscale('log')
        ax2.set_yscale('log')
        #
        fig.tight_layout()
        fn = 'output/_T-profile_' + name + '_z_merged.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def temperature_profiles_talk(self, name):
        plt.close('all')
        """
         Temperature profile
        """
        print(' --Plotting Temperature profile (merged)...')
        method_abbrs = ['SE', 'FB', 'BE', 'WD', 'PL']
        rs, Ts, TMaxs, TMins = {}, {}, {}, {}
        tbins = self.d[name]['WD']['Teff_bins']
        tbins2 = (tbins[:-1] + tbins[1:]) / 2
        for method_abbr in method_abbrs:
            r = t = w = np.array([])  # radius, temperature, weight
            for i in range(len(self.d[name]['binlist'])):
                temp_G = self.d[name]['aGas'][i]
                temp_R = self.d[name]['aRadius'][i]
                mask = self.d[name][method_abbr]['aPDFs_T'][i] > \
                    self.d[name][method_abbr]['aPDFs_T'][i].max() / 1000
                temp_T = self.Ts[mask]
                temp_P = self.d[name][method_abbr]['aPDFs_T'][i][mask]
                temp_P = temp_P / np.sum(temp_P) * temp_G * \
                    (self.d[name]['binmap'] ==
                     self.d[name]['binlist'][i]).sum()
                r = np.append(r, [temp_R] * len(temp_P))
                t = np.append(t, temp_T)
                w = np.append(w, temp_P)
            nanmask = np.isnan(r + t + w)
            r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
            rbins = np.linspace(np.min(r), np.max(r), self.rbin)
            rbins2 = (rbins[:-1] + rbins[1:]) / 2
            # Counting hist2d...
            counts, _, _ = np.histogram2d(r, t, bins=(rbins, tbins), weights=w)
            del r, t, w
            counts = counts.T
            T_Exp, T_Max, T_Min = np.array([]), np.array([]), np.array([])
            n_zeromask = np.full(counts.shape[1], True, dtype=bool)
            for i in range(counts.shape[1]):
                if np.sum(counts[:, i]) > 0:
                    counts[:, i] /= np.sum(counts[:, i])
                    csp = np.cumsum(counts[:, i])
                    csp = csp / csp[-1]
                    sst = np.interp([0.16, 0.5, 0.84], csp, tbins2)
                    T_Exp = np.append(T_Exp, np.sum(tbins2 * counts[:, i]))
                    T_Max = np.append(T_Max, sst[2])
                    T_Min = np.append(T_Min, sst[0])
                else:
                    n_zeromask[i] = False
            Ts[method_abbr] = T_Exp
            rs[method_abbr] = rbins2[n_zeromask]
            TMaxs[method_abbr] = T_Max
            TMins[method_abbr] = T_Min
        fs1 = 18
        fs2 = 15
        xlim = [np.nanmin(rs['SE']), np.nanmax(rs['SE'])]
        fig, ax = plt.subplots(2, 1, figsize=(6, 7.5),
                               gridspec_kw={'wspace': 0, 'hspace': 0})
        LABELS = {k: k for k in Ts.keys()}
        LABELS['SE'] = 'MBB'
        for k in Ts.keys():
            ax[0].plot(rs[k], Ts[k], alpha=0.8, label=LABELS[k], linewidth=2.0)
            ax[0].fill_between(rs[k], TMins[k], TMaxs[k], alpha=0.13)
        # ax[0].set_xlabel(r'Radius ($R_{25}$)', size=fs1)
        ax[0].set_ylabel(r'T$_d$ (K)', size=fs1)
        ax[0].set_xlim(xlim)
        ax[0].legend(fontsize=fs2, ncol=2, loc=2)
        # ax[0].set_title(r'$T_d$ profile', size=fs1, loc='left', y=0.85)
        ax[0].set_yticks([10, 20, 30, 40])
        ax[0].set_yticklabels([10, 20, 30, 40], fontsize=fs2)
        #
        R_SFR, SFR_profile = self.simple_profile(self.d[name]['SFR'],
                                                 self.d[name]['Radius'],
                                                 self.rbin,
                                                 self.d[name]['SigmaGas'])
        R_SMSD, SMSD_profile = self.simple_profile(self.d[name]['SMSD'],
                                                   self.d[name]['Radius'],
                                                   self.rbin,
                                                   self.d[name]['SigmaGas'])
        #
        ax2 = ax[1].twinx()
        ax[1].plot(R_SMSD, SMSD_profile, c='b')
        ax[1].fill_between(R_SMSD, SMSD_profile * 0.9, SMSD_profile * 1.1,
                           alpha=0.2, color='b')
        ax[1].tick_params('y', colors='b')
        ax2.plot(R_SFR, SFR_profile, 'r')
        ax2.fill_between(R_SFR, SFR_profile * 0.9, SFR_profile * 1.1,
                         alpha=0.2, color='r')
        ax2.set_ylabel(r'$\Sigma_{SFR}$ ($M_\odot kpc^{-2} yr^{-1}$)',
                       size=fs1, color='r')
        ax2.tick_params('y', colors='r')
        ax[1].set_xlim(xlim)
        ax[1].set_xlabel(r'Radius ($r_{25}$)', size=fs1)
        ax[1].set_ylabel(r'$\Sigma_*$ ($M_\odot pc^{-2}$)', size=fs1,
                         color='b')
        ax[1].tick_params(axis='both', labelsize=fs2)
        ax2.tick_params(axis='y', labelsize=fs2)
        ax[1].set_yscale('log')
        ax2.set_yscale('log')
        #
        fig.tight_layout()
        fn = 'output/_T-profile_' + name + '_z_talk.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def voronoi_plot(self, name='NGC5457', nwl=5, targetSN=5):
        plt.close('all')
        """
        Redo the Voronoi binning plot
        """
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Regrid']
            sed = grp['HERSCHEL_011111'].value
            bkgcov = grp['HERSCHEL_011111_BKGCOV'].value
            diskmask = grp['HERSCHEL_011111_DISKMASK'].value
            D = grp['DIST_MPC'].value
            cosINCL = grp['cosINCL'].value
            dp_radius = grp['RADIUS_KPC'].value
        binmap = np.full_like(diskmask, np.nan, dtype=int)
        print("Start binning " + name + "...")
        noise4snr = np.array([np.sqrt(bkgcov[i, i]) for i in range(nwl)])
        diskmask *= ~np.isnan(np.sum(sed, axis=2))
        temp_snr = sed[diskmask] / noise4snr
        signal_d = np.array([np.min(temp_snr[i]) for i in
                             range(len(temp_snr))])
        signal_d[signal_d < 0] = 0
        del temp_snr
        noise_d = np.ones(signal_d.shape)
        x_d, y_d = np.meshgrid(range(sed.shape[1]), range(sed.shape[0]))
        x_d, y_d = x_d[diskmask], y_d[diskmask]
        # Dividing into layers
        judgement = np.sum(signal_d) / np.sqrt(len(signal_d))
        if judgement < targetSN:
            print(name, 'is having just too small overall SNR. Will not fit')
        fwhm_radius = fwhm_sp500 * D * 1E3 / cosINCL
        nlayers = int(np.nanmax(dp_radius) // fwhm_radius)
        masks = []
        with np.errstate(invalid='ignore'):
            masks.append(dp_radius < fwhm_radius)
            for i in range(1, nlayers - 1):
                masks.append((dp_radius >= i * fwhm_radius) *
                             (dp_radius < (i + 1) * fwhm_radius))
            masks.append(dp_radius >= (nlayers - 1) * fwhm_radius)
        # test image: original layers
        testimage1 = np.full_like(dp_radius, np.nan)
        for i in range(nlayers):
            testimage1[masks[i]] = np.sin(i)
        testimage1[~diskmask] = np.nan
        #
        for i in range(nlayers - 1, -1, -1):
            judgement = np.sum(signal_d[masks[i][diskmask]]) / \
                np.sqrt(len(masks[i][diskmask]))
            if judgement < targetSN:
                if i > 0:
                    masks[i - 1] += masks[i]
                    del masks[i]
                else:
                    masks[0] += masks[1]
                    del masks[1]
        nlayers = len(masks)
        # test image: combined layers #
        testimage2 = np.full_like(dp_radius, np.nan)
        for i in range(nlayers):
            testimage2[masks[i]] = np.sin(i)
        testimage2[~diskmask] = np.nan
        #######################################
        masks = [masks[i][diskmask] for i in range(nlayers)]
        """ Modify radial bins here """
        max_binNum = 0
        binNum = np.full_like(signal_d, np.nan)
        for i in range(nlayers):
            x_l, y_l, signal_l, noise_l = x_d[masks[i]], y_d[masks[i]], \
                                          signal_d[masks[i]], noise_d[masks[i]]
            if np.min(signal_l) > targetSN:
                binNum_l = np.arange(len(signal_l))
            else:
                binNum_l, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
                    voronoi_m(x_l, y_l, signal_l, noise_l, targetSN,
                              pixelsize=1, plot=False, quiet=True)
            binNum_l += max_binNum
            max_binNum = np.max(binNum_l) + 1
            binNum[masks[i]] = binNum_l

        for i in range(len(signal_d)):
            binmap[y_d[i], x_d[i]] = binNum[i]
        temp_snr = sed / noise4snr
        testimage0 = np.empty_like(testimage1, dtype=float)
        for i in range(sed.shape[0]):
            for j in range(sed.shape[1]):
                testimage0[i, j] = \
                    temp_snr[i, j][np.argmin(np.abs(temp_snr[i, j]))]
        testimage0[~diskmask] = np.nan
        testimage3 = np.sin(binmap)
        testimage3[~diskmask] = np.nan
        #
        fig, ax = plt.subplots(2, 2, figsize=(9, 9))
        im = ax[0, 0].imshow(testimage0, origin='lower', vmin=0, vmax=5,
                             cmap='YlOrRd')
        axins = inset_axes(ax[0, 0], width="90%", height="5%", loc=8)
        plt.colorbar(im, cax=axins, orientation="horizontal")
        axins.xaxis.set_ticks_position("top")
        axins.set_xticklabels([0, 1, 2, 3, 4, 5], size=16)
        ax[0, 1].imshow(testimage1, origin='lower', cmap='jet')
        ax[1, 0].imshow(testimage2, origin='lower', cmap='jet')
        ax[1, 1].imshow(testimage3, origin='lower', cmap='jet')
        ax[1, 1].contour(dp_radius, origin='lower', levels=[7.4],
                         colors='white', linewidths=3, linestyles='dashed')
        ax[0, 0].contour(dp_radius, origin='lower', levels=[7.4],
                         colors='white', linewidths=3, linestyles='dashed')
        #
        fig2, ax2 = plt.subplots()
        ax2.imshow(testimage3, origin='lower', cmap='jet')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.tick_params(axis='both', bottom=False, left=False)
        fig2.tight_layout()
        fn = 'output/_Voronoi_poster_' + name + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig2, bbox_inches='tight')
        #
        titles = np.array([['(a)', '(b)'], ['(c)', '(d)']])
        for i in range(2):
            for j in range(2):
                ax[i, j].set_title(titles[i][j], size=22, x=0.9, y=0.9)
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])
        fig.tight_layout()
        fn = 'output/_Voronoi_' + name + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def example_model_merged(self, name):
        plt.close('all')
        print(' --Plotting merged example models...')
        i = np.argmax(self.d[name]['binlist'] ==
                      self.d[name]['binmap'][self.y, self.x])
        print(' --Plotting region', i)
        wl_complete = np.linspace(1, 800, 1000)
        wl_plot = np.linspace(51, 549, 100)
        method_abbrs = ['SE', 'FB', 'BE', 'WD', 'PL']
        num_para = {'SE': 3, 'FB': 2, 'WD': 3, 'BE': 3, 'PL': 4}
        grid = plt.GridSpec(2, 6, wspace=1.0, hspace=0.4)
        fig = plt.figure(figsize=(10, 4))
        for mi in range(5):
            """
            if method_abbrs[mi] != 'BE':
                continue
            """
            if mi == 0:
                ax = plt.subplot(grid[0, 0:2])
            elif mi == 1:
                ax = plt.subplot(grid[0, 2:4])
            elif mi == 2:
                ax = plt.subplot(grid[0, 4:])
            elif mi == 3:
                ax = plt.subplot(grid[1, 1:3])
            elif mi == 4:
                ax = plt.subplot(grid[1, 3:5])
            method_abbr = method_abbrs[mi]
            with File('hdf5_MBBDust/Models.h5', 'r') as hf:
                models = hf[method_abbr].value
            sed = self.d[name]['aSED'][i]
            cov_n1 = self.d[name]['acov_n1'][i]
            unc = np.sqrt(np.linalg.inv(cov_n1).diagonal())
            kappa160 = self.d['kappa160'][method_abbr]
            SigmaD = 10**self.d[name][method_abbr]['alogSigmaD'][i]
            SigmaDerr = self.d[name][method_abbr]['aSigmaD_err'][i]
            if method_abbr not in ['PL']:
                T = self.d[name][method_abbr]['aT'][i]
                Beta = self.d[name][method_abbr]['aBeta'][i]
            #
            # Colour correction factors
            #
            if method_abbr in ['SE', 'FB', 'FBPT', 'PB']:
                model_complete = SEMBB(wl_complete, SigmaD, T, Beta,
                                       kappa160=kappa160)
                ccf = SEMBB(wl, SigmaD, T, Beta, kappa160=kappa160) / \
                    z0mg_RSRF(wl_complete, model_complete, bands)
                sed_best_plot = \
                    SEMBB(wl_plot, SigmaD, T, Beta, kappa160=kappa160)
            elif method_abbr in ['BEMFB', 'BE']:
                lambda_c = self.d[name][method_abbr]['alambda_c'][i]
                beta2 = self.d[name][method_abbr]['abeta2'][i]
                model_complete = \
                    BEMBB(wl_complete, SigmaD, T, Beta, lambda_c, beta2,
                          kappa160=kappa160)
                ccf = BEMBB(wl, SigmaD, T, Beta, lambda_c, beta2,
                            kappa160=kappa160) / \
                    z0mg_RSRF(wl_complete, model_complete, bands)
                sed_best_plot = BEMBB(wl_plot, SigmaD, T, Beta, lambda_c,
                                      beta2, kappa160=kappa160)
            elif method_abbr in ['WD']:
                WDfrac = self.d[name][method_abbr]['aWDfrac'][i]
                model_complete = \
                    WD(wl_complete, SigmaD, T, Beta, WDfrac, kappa160=kappa160)
                ccf = WD(wl, SigmaD, T, Beta, WDfrac, kappa160=kappa160) / \
                    z0mg_RSRF(wl_complete, model_complete, bands)
                sed_best_plot = WD(wl_plot, SigmaD, T, Beta, WDfrac,
                                   kappa160=kappa160)
            elif method_abbr in ['PL']:
                alpha = self.d[name][method_abbr]['aalpha'][i]
                gamma = 10**self.d[name][method_abbr]['aloggamma'][i]
                logUmin = self.d[name][method_abbr]['alogUmin'][i]
                model_complete = \
                    PowerLaw(wl_complete, SigmaD, alpha, gamma, logUmin,
                             kappa160=kappa160)
                ccf = PowerLaw(wl, SigmaD, alpha, gamma, logUmin,
                               kappa160=kappa160) / \
                    z0mg_RSRF(wl_complete, model_complete, bands)
                sed_best_plot = PowerLaw(wl_plot, SigmaD, alpha, gamma,
                                         logUmin, kappa160=kappa160)
            sed_obs_plot = sed * ccf
            #
            # Begin fitting
            #
            if method_abbr == 'SE':
                Sigmas, Ts, Betas = np.meshgrid(self.SigmaDs, self.Ts,
                                                self.betas)
            elif method_abbr == 'FB':
                Sigmas, Ts = np.meshgrid(self.SigmaDs, self.Ts)
                Betas = np.full(Ts.shape, Beta)
            elif method_abbr == 'FBPT':
                Sigmas = self.SigmaDs
                Ts, Betas = \
                    np.full(Sigmas.shape, T), np.full(Sigmas.shape, Beta)
            elif method_abbr == 'PB':
                Ts, Sigmas = np.meshgrid(self.Ts, self.SigmaDs)
                Betas = np.full(Ts.shape, Beta)
            elif method_abbr == 'BEMFB':
                Sigmas, Ts, lambda_cs, beta2s = \
                    np.meshgrid(self.SigmaDs, self.Ts, self.lambda_cs,
                                self.beta2s)
            elif method_abbr == 'BE':
                Sigmas, Ts, beta2s = \
                    np.meshgrid(self.SigmaDs, self.Ts, self.beta2s)
                lambda_cs = np.full(Ts.shape, lambda_c)
                Betas = np.full(Ts.shape, Beta)
            elif method_abbr == 'WD':
                Sigmas, Ts, WDfracs = \
                    np.meshgrid(self.SigmaDs, self.Ts, self.WDfracs)
                Betas = np.full(Ts.shape, Beta)
            elif method_abbr == 'PL':
                Sigmas, alphas, gammas, logUmins = \
                    np.meshgrid(self.SigmaDs, self.alphas, self.gammas,
                                self.logUmins)
            diff = (models - sed)
            temp_matrix = np.empty_like(diff)
            for j in range(5):
                temp_matrix[..., j] = np.sum(diff * cov_n1[:, j], axis=-1)
            chi2 = np.sum(temp_matrix * diff, axis=-1)
            chi2 /= (5.0 - num_para[method_abbr])
            del temp_matrix, diff
            #
            # Plot corner
            #
            if method_abbr == 'BE':
                samplesBE = np.array([np.log10(Sigmas).flatten(),
                                      Ts.flatten(), beta2s.flatten()])
                tempchi2 = (chi2 * (5.0 - num_para['BE'])).flatten()
                weightsBE = 10**(-tempchi2)
                weightsBE /= np.sum(weightsBE)
            #
            # Selecting samples for plotting
            #
            chi2_threshold = 3.0
            chi2 -= np.nanmin(chi2)
            mask = chi2 < chi2_threshold
            if method_abbr not in ['PL']:
                chi2, Sigmas, Ts, Betas = \
                    chi2[mask], Sigmas[mask], Ts[mask], Betas[mask]
            else:
                chi2, Sigmas, alphas, gammas, logUmins = \
                    chi2[mask], Sigmas[mask], alphas[mask], gammas[mask], \
                    logUmins[mask]
            if method_abbr in ['BE']:
                lambda_cs, beta2s = lambda_cs[mask], beta2s[mask]
            elif method_abbr in ['WD']:
                WDfracs = WDfracs[mask]
            num = 50
            if len(Sigmas) > num:
                print(' ----Method', method_abbr, 'is having', str(num),
                      'points')
                mask = np.array([True] * num + [False] * (len(Sigmas) - num))
                np.random.shuffle(mask)
                if method_abbr not in ['PL']:
                    chi2, Sigmas, Ts, Betas = \
                        chi2[mask], Sigmas[mask], Ts[mask], Betas[mask]
                else:
                    chi2, Sigmas, alphas, gammas, logUmins = \
                        chi2[mask], Sigmas[mask], alphas[mask], gammas[mask], \
                        logUmins[mask]
                if method_abbr in ['BEMFB', 'BE']:
                    lambda_cs, beta2s = lambda_cs[mask], beta2s[mask]
                elif method_abbr in ['WD']:
                    WDfracs = WDfracs[mask]
            else:
                print(' ----Method', method_abbr, 'is having',
                      str(len(Sigmas)), 'points')
            transp = np.exp(-0.5 * chi2) * 0.2
            #
            # Begin plotting
            #
            for j in range(len(Sigmas)):
                if method_abbr in ['SE', 'FB']:
                    model_plot = SEMBB(wl_plot, Sigmas[j], Ts[j], Betas[j],
                                       kappa160=kappa160)
                elif method_abbr in ['BEMFB', 'BE']:
                    model_plot = BEMBB(wl_plot, Sigmas[j], Ts[j], Betas[j],
                                       lambda_cs[j], beta2s[j],
                                       kappa160=kappa160)
                elif method_abbr in ['WD']:
                    model_plot = WD(wl_plot, Sigmas[j], Ts[j], Betas[j],
                                    WDfracs[j], kappa160=kappa160)
                elif method_abbr in ['PL']:
                    model_plot = PowerLaw(wl_plot, Sigmas[j], alphas[j],
                                          gammas[j], logUmins[j],
                                          kappa160=kappa160)
                ax.plot(wl_plot, model_plot, alpha=transp[j], color='k')
            ax.plot(wl_plot, sed_best_plot, color='orange', linestyle='dashed',
                    linewidth=2)
            ax.errorbar(wl, sed, yerr=unc, fmt='_', ls=None, markersize=20,
                        color='red', capsize=10)
            ax.scatter(wl, sed_obs_plot, marker='o', zorder=5,
                       color='limegreen')
            ax.set_ylim([1.0, 22.0])
            ax.set_ylabel(r'SED ($MJy$ $sr^{-1}$)')
            ax.set_xticks([100, 300, 500])
            plt.minorticks_on()
            ax.set_xlabel(r'Wavelength ($\mu m$)')
            ax.set_title(method_abbr, x=0.95, y=0.6,
                         horizontalalignment='right')
            ax.text(0.95, 0.85,
                    r'$\log_{10}\Sigma_d$=' +
                    str(round(np.log10(SigmaD), 1)) + r'$\pm$' +
                    str(round(SigmaDerr, 1)),
                    horizontalalignment='right', verticalalignment='center',
                    transform=ax.transAxes)
        # fig.tight_layout()
        fn = 'output/_Model_' + str(self.x) + str(self.y) + '_' + \
            name + '_z_merged.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        #
        # Plot corner
        #
        logsigmas = np.log10(self.SigmaDs)
        s_max, s_min = -1.0, -1.8
        bin_s = np.sum((s_max >= logsigmas) * (logsigmas >= s_min))
        range_s = (np.nanmin(logsigmas[logsigmas >= s_min]),
                   np.nanmax(logsigmas[logsigmas <= s_max]))
        T_max, T_min = 24, 18
        bin_T = np.sum((T_max >= self.Ts) * (self.Ts >= T_min))
        range_T = (np.nanmin(self.Ts[self.Ts >= T_min]),
                   np.nanmax(self.Ts[self.Ts <= T_max]))
        b_max, b_min = 2.0, 0.5
        bin_beta = np.sum((b_max >= self.beta2s) * (self.beta2s >= b_min))
        range_beta = (np.nanmin(self.beta2s[self.beta2s >= b_min]),
                      np.nanmax(self.beta2s[self.beta2s <= b_max]))
        labels = [r'$\log(\Sigma_d)$', r'$T_d$', r'$\beta_2$']
        fig2 = corner_m(samplesBE.T, labels=labels, weights=weightsBE,
                        bins=[bin_s, bin_T, bin_beta], title_fmt=None,
                        range=[range_s, range_T, range_beta],
                        logscale=True)
        fn = 'output/_Corner_' + str(self.x) + str(self.y) + '_' + \
            name + '_BE.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig2, bbox_inches='tight')

    def metallicity_contour(self, name='NGC5457'):
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        aMetal = 8.715 - 0.027 * self.d[name]['aRadius'] * \
            self.d[name]['R25'] * 7.4 / GD_dist
        Metal = list2bin(aMetal, self.d[name]['binlist'],
                         self.d[name]['binmap'])
        DGR = list2bin(self.d[name]['BE']['alogSigmaD'] -
                       np.log10(self.d[name]['aGas']), self.d[name]['binlist'],
                       self.d[name]['binmap'])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(DGR, origin='lower', cmap='Reds')
        CS = ax.contour(Metal, cmap='prism',
                        levels=[8.5, 8.3, 8.1, 8.05, 8.0, 7.95, 7.9][::-1])
        ax.clabel(CS, inline=1, fontsize=10)
        fig.tight_layout()
        fn = 'output/_metal_ct_' + name + '_.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def kappa160_fH2(self, name='NGC5457', method_abbr='BE', quiet_four=True):
        plt.close('all')
        """
        1X1: kappa_160 vs H2
        """
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Regrid']
            heracles = grp['HERACLES'].value
        min_heracles = np.abs(np.nanmin(heracles))
        heracles[heracles < min_heracles] = np.nan
        aH2 = np.array([np.mean(heracles[self.d[name]['binmap'] == bin_])
                        for bin_ in self.d[name]['binlist']])
        afH2 = aH2 / self.d[name]['aGas']
        # 1. Want: kappa_160 vs. fH2
        # 2. Option 1: get kappa_160 vs. fH2 of each bin, and fit
        # 3. Option 2: get gas-weighted profiles of them independently,
        #    and matches them up together
        # Start with Option 1 first.
        # Need: DGR at each bin, D14 value at each bin (thus need metallicity
        #       at each bin)
        aMetal = (8.715 - 0.027 * self.d[name]['aRadius'] *
                  self.d[name]['R25'] * 7.4 / GD_dist)
        aD14_DGR = 10**(aMetal - solar_oxygen_bundance) / 150
        aDGR = 10**self.d[name][method_abbr]['alogSigmaD'] / \
            self.d[name]['aGas']
        arelDGR = aDGR / aD14_DGR
        k160 = self.d['kappa160'][method_abbr]
        ak160 = k160 * arelDGR
        #
        """
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(aMetal[aMetal >= 8.6], afH2[aMetal >= 8.6])
        fn = 'output/_afH2_vs_aMetal_' + name + '_.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        """
        #
        del aH2, heracles, aMetal, aD14_DGR, aDGR
        n_nanmask = ~np.isnan(afH2 + ak160 + arelDGR)
        afH2, ak160, arelDGR = \
            afH2[n_nanmask], ak160[n_nanmask], arelDGR[n_nanmask]
        afH2_log, ak160_log = np.log10(afH2), np.log10(ak160)
        ak160_fit = [[-1, -1], [-1, -1]]
        #
        yerr_log = self.d[name][method_abbr]['aSigmaD_err'][n_nanmask]
        yerr = np.empty_like(yerr_log)
        for i in range(len(yerr)):
            yerr[i] = max((10**yerr_log[i] - 1) * ak160[i],
                          (1 - 10**(1 - yerr_log[i])) * ak160[i])
        #
        arelDGR_fit, coef_ = fit_DataY(afH2_log, np.log10(arelDGR), yerr_log,
                                       quiet=False)
        arelDGR_fit = 10**arelDGR_fit
        ak160_fit[0][0], coef_ = fit_DataY(afH2, ak160, yerr, quiet=True)
        ak160_fit[0][1], coef_ = fit_DataY(afH2_log, ak160, yerr, quiet=True)
        ak160_fit[1][0], coef_ = fit_DataY(afH2, ak160_log, yerr_log,
                                           quiet=True)
        ak160_fit[1][1], coef_ = fit_DataY(afH2_log, ak160_log, yerr_log,
                                           quiet=True)
        ak160_fit[1][0] = 10**ak160_fit[1][0]
        ak160_fit[1][1] = 10**ak160_fit[1][1]
        titles = np.array([['(a)', '(b)'], ['(c)', '(d)']])
        #
        if not quiet_four:
            fig, ax = plt.subplots(2, 2,
                                   gridspec_kw={'wspace': 0, 'hspace': 0})
            for i in range(2):
                for j in range(2):
                    ax[i, j].scatter(afH2, ak160, s=1)
                    ax[i, j].plot(afH2, ak160_fit[i][j], 'r')
                    if j == 1:
                        ax[i, j].set_xscale('log')
                        ax[i, j].set_yticklabels([])
                    else:
                        ax[i, j].set_ylabel(r'$\kappa_{160}$ $(cm^2g^{-1})$')
                    if i == 1:
                        ax[i, j].set_yscale('log')
                        ax[i, j].set_xlabel(r'$f_{H_2}$')
                    else:
                        ax[i, j].set_xticklabels([])
                    ax[i, j].set_title(titles[i, j], x=0.1, y=0.8)
            fig.tight_layout()
            fn = 'output/_ak160_vs_afH2_4_' + name + '_.pdf'
            with PdfPages(fn) as pp:
                pp.savefig(fig, bbox_inches='tight')
        # 2018/07/26 old value
        DTM_unc_simple = self.d[name][method_abbr]['aSigmaD_err'][n_nanmask] +\
            np.log10((self.d[name]['aGas'] + 1) /
                     self.d[name]['aGas'])[n_nanmask]
        tempe_simple = 10**(np.mean(DTM_unc_simple))
        # 2018/07/04 new value
        DTM_unc_dex = 0.06 + \
            (0.023 + (0.001 * self.d[name]['aRadius'] *
                      self.d[name]['R25'] * 7.4 / GD_dist))[n_nanmask] + \
            np.log10((self.d[name]['aGas'] + 1) /
                     self.d[name]['aGas'])[n_nanmask] + \
            self.d[name][method_abbr]['aSigmaD_err'][n_nanmask]
        tempe = 10**np.mean(DTM_unc_dex)
        #
        err_y = 1.4
        yerr_eg = np.array([err_y * (1 - 1 / tempe),
                            err_y * (tempe - 1)]).reshape(2, 1)
        err_y_simple = err_y * tempe / tempe_simple
        yerr_eg_simple = np.array([err_y_simple * (1 - 1 / tempe_simple),
                                   err_y_simple *
                                   (tempe_simple - 1)]).reshape(2, 1)
        #
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(afH2, arelDGR, s=1)
        ax.errorbar(4E-2, err_y, yerr=yerr_eg, fmt='o', color='g', ms=2,
                    elinewidth=1)
        ax.errorbar(5E-2, err_y_simple, yerr=yerr_eg_simple,
                    fmt='o', color='c', ms=2, elinewidth=1)
        ax.plot(afH2, arelDGR_fit, 'r')
        ax.set_ylabel('DTM (result) / DTM (MW)', size=12)
        ax.set_yscale('log')
        ax.set_xlabel(r'$f_{H_2}$', size=12)
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.set_yticks([num / 10 for num in range(6, 25)], minor=True)
        ax.set_yticklabels([], minor=True)
        ax.set_yticks([1, 2], minor=False)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        fn = 'output/_relDGR_vs_afH2_' + name + '_.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        print('Max and min Norm. DGR:', max(arelDGR_fit),
              min(arelDGR_fit))
        print('Max and min kappa_160:', max(arelDGR_fit) * k160,
              min(arelDGR_fit) * k160)
        #
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(afH2, ak160, s=1)
        ax.plot(afH2, ak160_fit[1][1], 'r')
        ax.set_ylabel(r'$\kappa_{160}$ $(cm^2g^{-1})$', size=12)
        ax.set_yscale('log')
        ax.set_xlabel(r'$f_{H_2}$', size=12)
        ax.set_xscale('log')
        ax.set_yticks([num for num in range(10, 40)], minor=True)
        ax.set_yticklabels([], minor=True)
        ax.set_yticks([10, 20, 30], minor=False)
        """
        fn = 'output/_ak160_vs_afH2_' + name + '_.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        print('Max and min kappa_160:', max(ak160_fit[1][1]),
              min(ak160_fit[1][1]))
        """
        print('Min fH2 considered:', min(afH2))

    def residual_trend(self, name='NGC5457', method_abbr='BE'):
        plt.close('all')
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Regrid']
            heracles = grp['HERACLES'].value
            things = grp['THINGS'].value
        min_heracles = np.abs(np.nanmin(heracles))
        heracles[heracles < min_heracles] = np.nan
        aH2 = np.array([np.mean(heracles[self.d[name]['binmap'] == bin_])
                        for bin_ in self.d[name]['binlist']])
        aHI = np.array([np.mean(things[self.d[name]['binmap'] == bin_])
                        for bin_ in self.d[name]['binlist']])
        aGas = col2sur * H2HaHe * aHI + aH2
        alogfH2 = np.log10(aH2 / aGas)
        alogSFR = np.log10(bin2list(self.d[name]['SFR'],
                                    self.d[name]['binlist'],
                                    self.d[name]['binmap']))
        alogSMSD = np.log10(bin2list(self.d[name]['SMSD'],
                                     self.d[name]['binlist'],
                                     self.d[name]['binmap']))
        aRadius = self.d[name]['aRadius'] * self.d[name]['R25']
        alogMetal = (8.715 - 0.027 * aRadius * 7.4 / GD_dist)
        aDGR = 10**self.d[name][method_abbr]['alogSigmaD'] / \
            self.d[name]['aGas']
        aDTM = aDGR / self.max_possible_DGR(alogMetal)
        #
        mask = ~np.isnan(alogfH2 + aDTM)
        yerr_log = self.d[name][method_abbr]['aSigmaD_err'][mask]
        logfH2 = alogfH2[mask]
        logDTM = np.log10(aDTM[mask])
        Radius = aRadius[mask]
        ##
        mask2 = ~np.isnan(alogfH2 + alogSFR + alogSMSD)
        logfH22 = alogfH2[mask2]
        logSMSD = alogSMSD[mask2]
        logSFR = alogSFR[mask2]
        Radius2 = aRadius[mask2]
        logDTM2 = np.log10(aDTM[mask2])
        # logMetal = alogMetal[mask2]
        #
        df = pd.DataFrame()
        df[r'$f({\rm H}_2)$'] = logfH22
        df[r'$\Sigma_\star$'] = logSMSD
        df[r'$\Sigma_{\rm SFR}$'] = logSFR
        df[r'DTM'] = logDTM2
        fig = plt.figure()
        sns.heatmap(df.corr(), annot=True, cmap='Reds')
        fn = 'output/_direct_corr_' + name + '_' + method_abbr + '.pdf'
        # with PdfPages(fn) as pp:
        #     pp.savefig(fig, bbox_inches='tight')
        #
        df = pd.DataFrame()
        temp, coef_ = fit_DataY(Radius2, logfH22, np.full_like(logfH22, 0.05))
        df[r'$f({\rm H}_2)$'] = logfH22 - temp
        temp, coef_ = fit_DataY(Radius2, logSMSD, np.full_like(logSMSD, 0.05))
        df[r'$\Sigma_\star$'] = logSMSD - temp
        temp, coef_ = fit_DataY(Radius2, logSFR, np.full_like(logSFR, 0.05))
        df[r'$\Sigma_{\rm SFR}$'] = logSFR - temp
        temp, coef_ = fit_DataY(Radius2, logDTM2, np.full_like(logDTM2, 0.05))
        df[r'DTM'] = logDTM2 - temp
        fig = plt.figure()
        sns.heatmap(df.corr(), annot=True, cmap='Reds')
        fn = 'output/_residual_corr_' + name + '_' + method_abbr + '.pdf'
        # with PdfPages(fn) as pp:
        #     pp.savefig(fig, bbox_inches='tight')
        print('f(H2)')
        print('Direct:', stats.spearmanr(logfH22, logDTM2))
        print('Residual:', stats.spearmanr(df[r'$f({\rm H}_2)$'], df[r'DTM']))
        print('SMSD')
        print('Direct:', stats.spearmanr(logSMSD, logDTM2))
        print('Residual:', stats.spearmanr(df[r'$\Sigma_\star$'], df[r'DTM']))
        print('SFR')
        print('Direct:', stats.spearmanr(logSFR, logDTM2))
        print('Residual:', stats.spearmanr(df[r'$\Sigma_{\rm SFR}$'],
                                           df[r'DTM']))
        print('f(H2) again (residual)')
        print('DTM vs. radius:', stats.spearmanr(logDTM2, Radius2))
        print('f(H2) vs. radius:', stats.spearmanr(logfH22, Radius2))
        print('SMSD vs. radius:', stats.spearmanr(logSMSD, Radius2))
        print('SFR vs. radius:', stats.spearmanr(logSFR, Radius2))
        del aDGR, aDTM, alogfH2, alogSMSD, alogSFR
        #
        logfH2_RaT, coef_ = \
            fit_DataY(Radius, logfH2, np.full_like(logfH2, 0.05))
        logfH2_ReT = logfH2 - logfH2_RaT
        logDTM_RaT, coef2_ = fit_DataY(Radius, logDTM, yerr_log)
        logDTM_ReT = logDTM - logDTM_RaT
        #
        fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        DATAY = [logDTM2, df[r'DTM']]
        DATAX = [[logfH22, logSMSD, logSFR],
                 [df[r'$f({\rm H}_2)$'], df[r'$\Sigma_\star$'],
                  df[r'$\Sigma_{\rm SFR}$']]]
        LABELY = [r'log$_{10}$DTM', r'log$_{10}$DTM Residual']
        LABELX = [[r'log$_{10}$f$_{H_2}$',
                   r'log$_{10}\Sigma_\star$',
                   r'log$_{10}\Sigma_{\rm SFR}$'],
                  [r'log$_{10}$f$_{H_2}$ Residual',
                   r'log$_{10}\Sigma_\star$ Residual',
                   r'log$_{10}\Sigma_{\rm SFR}$ Residual']]
        LIMY = [(-0.42, 0.13), (-0.23, 0.15)]
        titles = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
        #
        for i in range(2):
            for j in range(3):
                ax[i, j].scatter(DATAX[i][j], DATAY[i], s=1, label='data')
                # ax[i, j].errorbar(0.3, -0.15, fmt='o',
                #                   yerr=np.nanmean(yerr_log),
                #                   color='g', ms=4, elinewidth=1,
                #                   label='mean unc')
                ax[i, j].set_xlabel(LABELX[i][j], size=11)
                ax[i, j].set_ylim(LIMY[i])
                ax[i, j].set_title(titles[i][j], size=16, x=0.9, y=0.07)
                """
                ax[i, j].text(0.58, 0.19,
                              r'$\rho_S=$' + RHOS[i][j],
                              horizontalalignment='left',
                              verticalalignment='bottom', zorder=5,
                              transform=ax[i, j].transAxes,
                              fontdict={'fontsize': 14})
                ax[i, j].text(0.58, 0.09,
                              PVALUES[i][j],
                              horizontalalignment='left',
                              verticalalignment='bottom', zorder=5,
                              transform=ax[i, j].transAxes, color=COLORS[i][j],
                              fontdict={'fontsize': 14})
                """
                if j == 0:
                    ax[i, j].set_ylabel(LABELY[i], size=11)
                else:
                    ax[i, j].set_yticklabels([])
                # ax[i, j].minorticks_on()
        # ax.legend(fontsize=9)
        """
        ax[1, 0].text(0.0, 0.09,
                      'The only significant correlation in residuals',
                      horizontalalignment='left',
                      verticalalignment='bottom', zorder=5,
                      transform=ax[i, j].transAxes, color=COLORS[i][j],
                      fontdict={'fontsize': 14})
        """
        fig.tight_layout()
        fn = 'output/_residual_trends_' + name + '_' + \
            method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        print(np.nanmean(yerr_log))
        return
        #
        # Pearson and p-values
        corr = stats.pearsonr(logfH2_ReT, logDTM_ReT)
        ax.text(1.01, 0.9,
                'Pearson=' + str(round(corr[0], 3)),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.text(1.01, 0.85,
                'p-value=' + str(corr[1]),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        #
        # Shuffle Pearson
        temp_x = np.array([num for num in logfH2_ReT])
        corrs = []
        for j in range(10000):
            np.random.shuffle(temp_x)
            corrs.append(stats.pearsonr(temp_x, logDTM_ReT)[0])
        mean_, std_ = np.mean(corrs), np.std(corrs)
        ax.text(1.01, 0.8,
                'Shuffle STD=' + str(round(std_, 3)),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.text(1.01, 0.75,
                'Shuffle Mean=' + '{0:.2E}'.format(Decimal(str(mean_))),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        #
        # Spearman and p-values
        corr = stats.spearmanr(logfH2_ReT, logDTM_ReT)
        ax.text(1.01, 0.65,
                'Spearman=' + str(round(corr[0], 3)),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.text(1.01, 0.6,
                'p-value=' + str(round(corr[1], 3)),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        # Y-error
        ax.text(1.01, 0.5,
                'Mean Y unc=' + str(round(np.nanmean(yerr_log), 2)),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        # Random noise
        corrs = []
        for j in range(10000):
            temp_y = np.random.lognormal(mean=logDTM_ReT,
                                         sigma=yerr_log)
            corrs.append(stats.pearsonr(logfH2_ReT, temp_y)[0])
        ax.text(1.01, 0.4,
                'Noise Pearson Max=' + str(round(np.max(corrs), 2)),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        #
        fn = 'output/_residual_trend_fH2_FULLTEXT' + name + '_' + \
            method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def residual_trend_half(self, name='NGC5457', method_abbr='BE'):
        plt.close('all')
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Regrid']
            heracles = grp['HERACLES'].value
            things = grp['THINGS'].value
        min_heracles = np.abs(np.nanmin(heracles))
        heracles[heracles < min_heracles] = np.nan
        aH2 = np.array([np.mean(heracles[self.d[name]['binmap'] == bin_])
                        for bin_ in self.d[name]['binlist']])
        aHI = np.array([np.mean(things[self.d[name]['binmap'] == bin_])
                        for bin_ in self.d[name]['binlist']])
        aGas = col2sur * H2HaHe * aHI + aH2
        alogfH2 = np.log10(aH2 / aGas)
        alogSFR = np.log10(bin2list(self.d[name]['SFR'],
                                    self.d[name]['binlist'],
                                    self.d[name]['binmap']))
        alogSMSD = np.log10(bin2list(self.d[name]['SMSD'],
                                     self.d[name]['binlist'],
                                     self.d[name]['binmap']))
        aRadius = self.d[name]['aRadius'] * self.d[name]['R25']
        alogMetal = (8.715 - 0.027 * aRadius * 7.4 / GD_dist)
        aDGR = 10**self.d[name][method_abbr]['alogSigmaD'] / \
            self.d[name]['aGas']
        aDTM = aDGR / self.max_possible_DGR(alogMetal)
        #
        mask = ~np.isnan(alogfH2 + aDTM)
        yerr_log = self.d[name][method_abbr]['aSigmaD_err'][mask]
        logfH2 = alogfH2[mask]
        Radius = aRadius[mask]
        ##
        mask2 = ~np.isnan(alogfH2 + alogSFR + alogSMSD)
        logfH22 = alogfH2[mask2]
        logSMSD = alogSMSD[mask2]
        logSFR = alogSFR[mask2]
        Radius2 = aRadius[mask2]
        logDTM2 = np.log10(aDTM[mask2])
        # logMetal = alogMetal[mask2]
        #
        df = pd.DataFrame()
        df[r'$f({\rm H}_2)$'] = logfH22
        df[r'$\Sigma_\star$'] = logSMSD
        df[r'$\Sigma_{\rm SFR}$'] = logSFR
        df[r'DTM'] = logDTM2
        fig = plt.figure()
        sns.heatmap(df.corr(), annot=True, cmap='Reds')
        fn = 'output/_direct_corr_' + name + '_' + method_abbr + '.pdf'
        # with PdfPages(fn) as pp:
        #     pp.savefig(fig, bbox_inches='tight')
        #
        df = pd.DataFrame()
        temp, coef_ = fit_DataY(Radius2, logfH22, np.full_like(logfH22, 0.05))
        df[r'$f({\rm H}_2)$'] = logfH22 - temp
        temp, coef_ = fit_DataY(Radius2, logSMSD, np.full_like(logSMSD, 0.05))
        df[r'$\Sigma_\star$'] = logSMSD - temp
        temp, coef_ = fit_DataY(Radius2, logSFR, np.full_like(logSFR, 0.05))
        df[r'$\Sigma_{\rm SFR}$'] = logSFR - temp
        temp, coef_ = fit_DataY(Radius2, logDTM2, np.full_like(logDTM2, 0.05))
        df[r'DTM'] = logDTM2 - temp
        fig = plt.figure()
        sns.heatmap(df.corr(), annot=True, cmap='Reds')
        fn = 'output/_residual_corr_' + name + '_' + method_abbr + '.pdf'
        # with PdfPages(fn) as pp:
        #     pp.savefig(fig, bbox_inches='tight')
        RHOS = [[]]
        PVALUES = [[]]
        print('f(H2)')
        print('Direct:', stats.spearmanr(logfH22, logDTM2))
        print('Residual:', stats.spearmanr(df[r'$f({\rm H}_2)$'], df[r'DTM']))
        r, p = stats.spearmanr(df[r'$f({\rm H}_2)$'], df[r'DTM'])
        RHOS[0].append(str(round(r, 2)))
        if p < 0.01:
            PVALUES[0].append(r'p-value $\ll$ 1')
        else:
            PVALUES[0].append(r'p-value = ' + str(round(p, 2)))
        print('SMSD')
        print('Direct:', stats.spearmanr(logSMSD, logDTM2))
        print('Residual:', stats.spearmanr(df[r'$\Sigma_\star$'], df[r'DTM']))
        r, p = stats.spearmanr(df[r'$\Sigma_\star$'], df[r'DTM'])
        RHOS[0].append(str(round(r, 2)))
        if p < 0.01:
            PVALUES[0].append(r'p-value $\ll$ 1')
        else:
            PVALUES[0].append(r'p-value = ' + str(round(p, 2)))
        print('SFR')
        print('Direct:', stats.spearmanr(logSFR, logDTM2))
        print('Residual:', stats.spearmanr(df[r'$\Sigma_{\rm SFR}$'],
                                           df[r'DTM']))
        r, p = stats.spearmanr(df[r'$\Sigma_{\rm SFR}$'], df[r'DTM'])
        RHOS[0].append(str(round(r, 2)))
        if p < 0.01:
            PVALUES[0].append(r'p-value $\ll$ 1')
        else:
            PVALUES[0].append(r'p-value = ' + str(round(p, 2)))
        del aDGR, aDTM, alogfH2, alogSMSD, alogSFR
        #
        logfH2_RaT, coef_ = \
            fit_DataY(Radius, logfH2, np.full_like(logfH2, 0.05))
        #
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        DATAY = [df[r'DTM']]
        DATAX = [[df[r'$f({\rm H}_2)$'], df[r'$\Sigma_\star$'],
                  df[r'$\Sigma_{\rm SFR}$']]]
        LABELY = [r'log$_{10}$DTM res.']
        LABELX = [[r'log$_{10}$f$_{H_2}$ res.',
                   r'log$_{10}\Sigma_\star$ res.',
                   r'log$_{10}\Sigma_{\rm SFR}$ res.']]
        LIMY = [(-0.23, 0.15)]
        # titles = [['(a)', '(b)', '(c)']]
        #
        for i in range(1):
            for j in range(3):
                ax[j].scatter(DATAX[i][j], DATAY[i], s=1, label='data')
                # ax[i, j].errorbar(0.3, -0.15, fmt='o',
                #                   yerr=np.nanmean(yerr_log),
                #                   color='g', ms=4, elinewidth=1,
                #                   label='mean unc')
                ax[j].set_xlabel(LABELX[i][j], size=20)
                ax[j].set_ylim(LIMY[i])
                # ax[j].set_title(titles[i][j], size=16, x=0.9, y=0.07)
                ax[j].text(0.58, 0.19,
                           r'$\rho_S=$' + RHOS[i][j],
                           horizontalalignment='left',
                           verticalalignment='bottom', zorder=5,
                           transform=ax[j].transAxes,
                           fontdict={'fontsize': 13})
                ax[j].text(0.58, 0.09,
                           PVALUES[i][j],
                           horizontalalignment='left',
                           verticalalignment='bottom', zorder=5,
                           transform=ax[j].transAxes,
                           fontdict={'fontsize': 13})
                if j == 0:
                    ax[j].set_ylabel(LABELY[i], size=20)
                else:
                    ax[j].set_yticklabels([])
                # ax[i, j].minorticks_on()
        # ax.legend(fontsize=9)
        """
        ax[1, 0].text(0.0, 0.09,
                      'The only significant correlation in residuals',
                      horizontalalignment='left',
                      verticalalignment='bottom', zorder=5,
                      transform=ax[i, j].transAxes, color=COLORS[i][j],
                      fontdict={'fontsize': 14})
        """
        fig.tight_layout()
        fn = 'output/_residual_trends_half_' + name + '_' + \
            method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        print(np.nanmean(yerr_log))

    def alpha_CO_test(self, name='NGC5457'):
        plt.close('all')
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Regrid']
            heracles = grp['HERACLES'].value
            things = grp['THINGS'].value
        temp_s = pd.read_csv('data/Tables/galaxy_data.csv',
                             index_col=0).loc[name]
        alpha_CO = temp_s['ACO']
        del temp_s
        min_heracles = np.abs(np.nanmin(heracles))
        heracles[heracles < min_heracles] = np.nan
        HERA = map2bin(heracles, self.d[name]['binlist'],
                       self.d[name]['binmap']) / alpha_CO
        HI = map2bin(things, self.d[name]['binlist'],
                     self.d[name]['binmap']) * col2sur * H2HaHe
        Radius = self.d[name]['Radius'] * self.d[name]['R25']
        MetalOH = (8.715 - 0.027 * Radius * 7.4 / GD_dist)
        Z_prime = 10**(MetalOH - solar_oxygen_bundance)
        #
        gas_S13 = HI + HERA * alpha_CO
        #
        MW_alpha_CO = alpha_CO
        AV_MW = 5
        Delta_AV = 0.53 + 0.35 - 0.097 * np.log(Z_prime)
        alpha_CO_W10 = MW_alpha_CO * \
            np.exp(4 * Delta_AV * (1 / Z_prime - 1) / AV_MW)
        # Wolfire et al. (2010) model
        # Bolatto et al. (2013) Eq. 28.
        # A constant DTM is assumed. Delta_AV
        # MO/MZ = 51%
        # MW_alpha_CO = alpha_CO
        # The mean extinction through a GMC at Milky Way metallicity AV_MW = 5
        gas_W10 = HI + HERA * alpha_CO_W10
        #
        gas_S13[np.isnan(gas_W10)] = np.nan
        #
        fig, ax = plt.subplots()
        ax.plot(np.log10(Z_prime.flatten()),
                [alpha_CO] * len(MetalOH.flatten()),
                label='Sandstrom et al. (2013)')
        mask = np.argsort(Z_prime.flatten())
        ax.plot(np.log10(Z_prime.flatten())[mask],
                alpha_CO_W10.flatten()[mask], label='Wolfire et al. (2010)')
        ax.set_xlabel(r"log(Z') = log(Z/Z$_\odot$)")
        ax.set_ylabel(r'$\alpha_{\rm CO}$ $(\rm M_{\odot} pc^{-2}$' +
                      r'$ (K km s^{-1})^{-1}$')
        ax.grid(b=True, which='both', axis='both')
        ax.legend()
        ax.set_yscale('log')
        ax2 = fig.add_axes([.55, .45, .35, .35])
        mask = MetalOH.flatten() > 8.35
        ax2.plot(np.log10(Z_prime.flatten())[mask],
                 [alpha_CO] * len(MetalOH.flatten()[mask]),
                 label='Sandstrom et al. (2013)')
        ax2.plot(np.log10(Z_prime.flatten())[mask],
                 alpha_CO_W10.flatten()[mask],
                 label='Wolfire et al. (2010)')
        ax2.set_yscale('log')
        ax2.grid(b=True, which='both', axis='both')
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%d"))
        fig.tight_layout()
        fn = 'output/alphaCO_S13-vs-Wolfire.png'
        fig.savefig(fn)
        #
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(3, 9))
        images = [gas_S13, gas_W10, np.log10(gas_S13 / gas_W10)]
        titles = [r'$\Sigma_{\rm gas}$ ($\alpha_{\rm CO, Sandstrom+2013})$',
                  r'$\Sigma_{\rm gas}$ $\alpha_{\rm CO, Wolfire+2010}$',
                  'Change (dex)']
        for i in range(3):
            im = ax[i].imshow(images[i], origin='lower')
            ax[i].set_title(titles[i])
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            plt.colorbar(im, ax=ax[i])
        fig.tight_layout()
        fn = 'output/alphaCOgas_S13-vs-Wolfire.png'
        fig.savefig(fn)
        diff = np.log10(gas_S13 / gas_W10)
        diff = np.sort(diff[~np.isnan(diff)])
        counts = np.arange(len(diff)) / len(diff)
        ssd = np.interp([0.16, 0.5, 0.84], counts, diff)
        print('median', np.median(diff), ssd)

    def pdf_profiles_four_beta(self, name='NGC5457', m='FB'):
        plt.close('all')
        """
        1X1: DGR profile
        """
        print(' --Plotting merged DGR vs. Metallicity...')
        betas = ['1.6', '1.8', '2.0', '2.2']
        # betas = ['1.8', '2.0', '2.2']
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        DGRs = {}
        DGR16s, DGR84s = {}, {}
        n_zeromasks = {}
        for beta in betas:
            with File('hdf5_MBBDust/0710_beta=' + beta + '_backup/' + name +
                      '.h5', 'r') as hf:
                grp = hf['Fitting_results']
                subgrp = grp[m]
                aPDFs = subgrp['PDF'].value
            r = d = w = np.array([])  # radius, dgr, weight
            for i in range(len(self.d[name]['binlist'])):
                temp_G = self.d[name]['aGas'][i]
                temp_R = self.d[name]['aRadius'][i]
                mask = aPDFs[i] > aPDFs[i].max() / 1000
                temp_DGR = self.SigmaDs[mask] / temp_G
                temp_P = aPDFs[i][mask]
                temp_P = temp_P / np.sum(temp_P) * temp_G * \
                    (self.d[name]['binmap'] ==
                     self.d[name]['binlist'][i]).sum()
                r = np.append(r, [temp_R] * len(temp_P))
                d = np.append(d, temp_DGR)
                w = np.append(w, temp_P)
            nanmask = np.isnan(r + d + w)
            r, d, w = r[~nanmask], d[~nanmask], w[~nanmask]
            rbins = np.linspace(np.min(r), np.max(r), self.rbin)
            dbins = \
                np.logspace(np.min(np.log10(d)), np.max(np.log10(d)),
                            self.dbin)
            # Counting hist2d...
            counts, _, _ = np.histogram2d(r, d, bins=(rbins, dbins), weights=w)
            del r, d, w
            counts = counts.T
            n_zeromask = np.full(counts.shape[1], True, dtype=bool)
            dbins2 = np.sqrt(dbins[:-1] + dbins[1:])
            rbins2 = (rbins[:-1] + rbins[1:]) / 2
            DGR_Median = DGR_LExp = DGR_Max = DGR_16 = DGR_84 = np.array([])
            for i in range(counts.shape[1]):
                if np.sum(counts[:, i]) > 0:
                    counts[:, i] /= np.sum(counts[:, i])
                    csp = np.cumsum(counts[:, i])
                    csp = csp / csp[-1]
                    ssd = np.interp([0.16, 0.5, 0.84], csp, np.log10(dbins2))
                    DGR_Median = np.append(DGR_Median, 10**ssd[1])
                    DGR_LExp = np.append(DGR_LExp,
                                         10**np.sum(np.log10(dbins2) *
                                                    counts[:, i]))
                    DGR_Max = np.append(DGR_Max,
                                        dbins2[np.argmax(counts[:, i])])
                    DGR_16 = np.append(DGR_16, 10**ssd[0])
                    DGR_84 = np.append(DGR_84, 10**ssd[2])
                else:
                    n_zeromask[i] = False
            DGR16s[beta] = DGR_16
            DGR84s[beta] = DGR_84
            n_zeromasks[beta] = n_zeromask
            DGRs[beta] = DGR_LExp
            #
        # My DGR gradient with Remy-Ruyer data and various models
        fs1 = 24
        fs2 = 20
        xbins2 = (8.715 - 0.027 * rbins2 * self.d[name]['R25'] *
                  7.4 / GD_dist)
        df = pd.read_csv("data/Tables/Remy-Ruyer_2014.csv")
        r_ = (8.715 - df['12+log(O/H)'].values) / 0.027 * GD_dist / 7.4 / \
            self.d[name]['R25']
        r__ = np.linspace(np.nanmin(r_), np.nanmax(r_), 50)
        x__ = (8.715 - 0.027 * r__ * self.d[name]['R25'] * 7.4 / GD_dist -
               solar_oxygen_bundance)
        o__ = x__ + solar_oxygen_bundance
        # New uncertainty
        DGR_unc_dex = np.zeros_like(rbins2)
        # second part: gas zero point uncertainty
        radius_map = np.zeros_like(self.d[name]['binmap']) * np.nan
        gas_map = np.zeros_like(self.d[name]['binmap']) * np.nan
        for i in range(len(self.d[name]['binlist'])):
            b_mask = self.d[name]['binmap'] == self.d[name]['binlist'][i]
            radius_map[b_mask] = self.d[name]['aRadius'][i]
            gas_map[b_mask] = self.d[name]['aGas'][i]
        for i in range(len(DGR_unc_dex)):
            u_mask = (rbins[i] <= radius_map) * (rbins[i + 1] >= radius_map)
            temp_gas = np.nanmean(gas_map[u_mask])
            DGR_unc_dex[i] += np.log10((temp_gas + 1) / temp_gas)
        DGR_unc_mtp = 10**DGR_unc_dex
        #
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.set_xlim([np.nanmin(xbins2), np.nanmax(xbins2)])
        for i in range(len(DGRs.keys())):
            k = list(DGRs.keys())[i]
            xbins2 = (8.715 - 0.027 * rbins2[n_zeromasks[k]] *
                      self.d[name]['R25'] * 7.4 / GD_dist)
            ax.plot(xbins2, DGRs[k], label=r'$\beta$=' + k, linewidth=4.0,
                    alpha=0.8)
            ax.fill_between(xbins2, DGR16s[k] / DGR_unc_mtp,
                            DGR84s[k] * DGR_unc_mtp, alpha=0.13)
        ax.fill_between(o__, 10**(o__ - 12) * 16.0 / 1.008 / 0.51 / 1.36,
                        10**(o__ - 12) * 16.0 / 1.008 / 0.445 / 1.36,
                        alpha=0.7, label='MAX', hatch='/')
        ax.legend(fontsize=fs2, ncol=2, loc=4)
        ax.tick_params(axis='both', labelsize=fs2)
        ax.set_ylim([0.0003, 0.02])
        ax.set_yscale('log')
        ax.set_xlabel('12 + log(O/H)', size=fs1)
        ax.set_ylabel('DGR from the ' + m + ' model', size=fs1)
        ax2 = ax.twiny()
        ax2.set_xlabel('Radius (kpc)', size=fs1, color='k')
        ax2.set_xlim([np.nanmax(rbins2) * self.d[name]['R25'],
                      np.nanmin(rbins2) * self.d[name]['R25']])
        ax2.tick_params(axis='both', labelsize=fs2)
        fig.tight_layout()
        fn = 'output/_DGR-vs-Metallicity_' + name + '_beta_merged.png'
        fig.savefig(fn)
        for beta in betas:
            with File('hdf5_MBBDust/0710_beta=' + beta + '_backup/' +
                      'Calibration.h5', 'r') as hf:
                grp = hf[m]
                print(beta, grp['kappa160'].value)
        # with PdfPages(fn) as pp:
        #     pp.savefig(fig, bbox_inches='tight')

    def unbinned_and_binned_gas_masp(self, name='NGC5457'):
        binned = self.d[name]['SigmaGas']
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Regrid']
            unbinned = grp['TOTAL_GAS'].value
            unbinned[np.isnan(binned)] = np.nan
        fig, ax = plt.subplots(ncols=2, figsize=(7, 3))
        im = ax[0].imshow(unbinned, origin='lower', norm=LogNorm(),
                          cmap='viridis', vmax=39, vmin=0.1)
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title(r'Unbinned $\Sigma_{\rm gas}$')
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        im = ax[1].imshow(binned, origin='lower', norm=LogNorm(),
                          cmap='viridis', vmax=39, vmin=0.1)
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title(r'Binned $\Sigma_{\rm gas}$')
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        fig.tight_layout()
        fn = 'output/_0713_gas.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')

    def realize_vs_simple_sum(self, name='NGC5457', method_abbr='BE',
                              rbin_num=19, nop=10):
        plt.close('all')
        binlist = self.d[name]['binlist']
        aRadius = self.d[name]['aRadius']
        aGas = self.d[name]['aGas']
        aPDFs = self.d[name][method_abbr]['aPDFs']
        aArea = np.array([(self.d[name]['binmap'] == binlist[i]).sum()
                          for i in range(len(binlist))])
        for i in range(len(aPDFs)):
            aPDFs[i] /= np.nansum(aPDFs[i])
        #
        # Pick out the i's and calculate simple sum PDF.
        #
        rbins = np.linspace(np.min(aRadius), np.max(aRadius), self.rbin)
        rmin, rmax = rbins[rbin_num], rbins[rbin_num + 1]
        is_ = []
        d, w = np.array([]), np.array([])
        total_gas = 0
        for i in range(len(binlist)):
            if rmin <= aRadius[i] <= rmax:
                is_.append(i)
                mask = aPDFs[i] > aPDFs[i].max() / 1000
                temp_P = aPDFs[i][mask]
                temp_P = temp_P / np.sum(temp_P) * aGas[i] * aArea[i]
                d = np.append(d, self.SigmaDs[mask] / aGas[i])
                w = np.append(w, temp_P)
                total_gas += aGas[i] * aArea[i]
        logd = np.log10(d)
        #
        # Calculate random realize PDF.
        #
        logsigma_step = 0.025
        min_logsigma = -4.
        max_logsigma = 1.
        half_step = logsigma_step / 2
        rand_diff = max_logsigma - min_logsigma
        rand_min = min_logsigma - half_step
        n = 10000

        def mp_realize(pid, total_step, output):
            DGRs = []
            for step in range(total_step):
                if step % 1000 == 0:
                    print('Processor', pid, 'at step', step)
                total_dust = 0
                for i in is_:
                    while(True):
                        rand_logSigmaD, rand_p = np.random.rand(2)
                        rand_logSigmaD = rand_min + rand_diff * rand_logSigmaD
                        pos = int((rand_logSigmaD - rand_min) // logsigma_step)
                        if aPDFs[i][pos] > rand_p:
                            total_dust += (10**rand_logSigmaD) * aArea[i]
                            break
                DGRs.append(total_dust / total_gas)
            output.put((pid, DGRs))

        total_step = int(n // nop)
        timeout = 1e-6
        q = mp.Queue()
        processes = [mp.Process(target=mp_realize,
                     args=(pid, total_step, q))
                     for pid in range(nop)]
        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout)
        DGRs = []
        for p in processes:
            pid, result = q.get()
            print("--Got result from process", pid)
            DGRs += result
            del pid, result
        #
        xlim = (-4.5, 1.5)
        dbins = 50
        num_samples = len(is_)
        titles = [['n=' + str(int(n // 100)), 'n=' + str(int(n // 10))],
                  ['n=' + str(n), r'$M_{\rm gas}$-weighted']]
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6))
        ax[0, 0].hist(np.log10(DGRs[:int(n // 100)]), bins=dbins, normed=True,
                      range=xlim)
        ax[0, 1].hist(np.log10(DGRs[:int(n // 10)]), bins=dbins, normed=True,
                      range=xlim)
        ax[1, 0].hist(np.log10(DGRs), bins=dbins, normed=True, range=xlim)
        ax[1, 1].hist(logd, weights=w, bins=dbins, normed=True, range=xlim)
        for i in range(2):
            for j in range(2):
                ax[i, j].set_xlabel(r'$\log_{10}$[' +
                                    r'$\Sigma_d (M_\odot pc^{-2})$]')
                ax[i, j].set_yticks([])
                ax[i, j].set_title(titles[i][j])
                ax[i, j].set_xlim(xlim)
        fig.tight_layout()
        temp = str(rbin_num)
        if len(temp) == 1:
            temp = '0' + temp
        fn = 'output/realize_vs_simple_sum' + temp + '_num_samples=' + \
            str(num_samples) + '.png'
        fig.savefig(fn)

    def realize_PDF(self, name='NGC5457', method_abbr='BE', nop=10, n=20000,
                    timeout=1e-6):
        np.random.seed(time.time())
        plt.close('all')
        binlist = self.d[name]['binlist']
        aRadius = self.d[name]['aRadius']
        aGas = self.d[name]['aGas']
        aPDFs = self.d[name][method_abbr]['aPDFs']
        aArea = np.array([(self.d[name]['binmap'] == binlist[i]).sum()
                          for i in range(len(binlist))])
        for i in range(len(aPDFs)):
            aPDFs[i] /= np.nansum(aPDFs[i])
        rbins = np.linspace(np.min(aRadius), np.max(aRadius), self.rbin)
        rbins2 = (rbins[:-1] + rbins[1:]) / 2
        dbins = np.logspace(np.log10(3E-5), np.log10(6E-2), self.dbin)
        dbins2 = np.sqrt(dbins[:-1] * dbins[1:])
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        xbins2 = (8.715 - 0.027 * rbins2 * self.d[name]['R25'] * 7.4 / GD_dist)
        logsigma_step = 0.025
        min_logsigma = -4.
        max_logsigma = 1.
        half_step = logsigma_step / 2
        rand_diff = max_logsigma - min_logsigma
        rand_min = min_logsigma - half_step
        total_step = int(n // nop)

        def mp_realize(pid, total_step, total_gas, output):
            DGRs = []
            for step in range(total_step):
                if step % 1000 == 0:
                    print('Processor', pid, 'at step', step)
                total_dust = 0
                for i in is_:
                    while(True):
                        rand_logSigmaD, rand_p = np.random.rand(2)
                        rand_logSigmaD = \
                            rand_min + rand_diff * rand_logSigmaD
                        pos = int((rand_logSigmaD - rand_min) //
                                  logsigma_step)
                        if aPDFs[i][pos] > rand_p:
                            total_dust += (10**rand_logSigmaD) * aArea[i]
                            break
                DGRs.append(total_dust / total_gas)
            output.put((pid, DGRs))
        #
        # Pick out the i's and calculate simple sum PDF.
        #
        realize_DGR_EXP = []
        realize_DGR_16 = []
        realize_DGR_84 = []
        Mgas_DGR_EXP = []
        Mgas_DGR_16 = []
        Mgas_DGR_84 = []
        for rbin_num in range(len(rbins2)):
            rmin, rmax = rbins[rbin_num], rbins[rbin_num + 1]
            is_ = []
            d, w = np.array([]), np.array([])
            total_gas = 0
            for i in range(len(binlist)):
                if rmin <= aRadius[i] <= rmax:
                    is_.append(i)
                    mask = aPDFs[i] > aPDFs[i].max() / 1000
                    temp_P = aPDFs[i][mask]
                    temp_P = temp_P / np.sum(temp_P) * aGas[i] * aArea[i]
                    d = np.append(d, self.SigmaDs[mask] / aGas[i])
                    w = np.append(w, temp_P)
                    total_gas += aGas[i] * aArea[i]
            logd = np.log10(d)
            #
            # Calculate random realize PDF.
            #
            q = mp.Queue()
            processes = [mp.Process(target=mp_realize,
                         args=(pid, total_step, total_gas, q))
                         for pid in range(nop)]
            for p in processes:
                p.start()
            for p in processes:
                p.join(timeout)
            DGRs = []
            for p in processes:
                pid, result = q.get()
                print("--Got result from process", pid)
                DGRs += result
                del pid, result
            del processes, q
            # Calculate and save Realize 16-EXP-84 to array
            counts, _ = np.histogram(DGRs, bins=dbins)
            realize_DGR_EXP.append(10**(np.sum(counts * np.log10(dbins2)) /
                                        np.sum(counts)))
            csp = np.cumsum(counts)
            csp = csp / csp[-1]
            logDGR16, logDGR84 = np.interp([0.16, 0.84], csp, np.log10(dbins2))
            realize_DGR_16.append(10**logDGR16)
            realize_DGR_84.append(10**logDGR84)
            # Calculate and save M_gas-weighted 16-EXP-84 to array
            counts2, _ = np.histogram(d, bins=dbins, weights=w)
            Mgas_DGR_EXP.append(10**(np.sum(counts2 * np.log10(dbins2)) /
                                     np.sum(counts2)))
            csp = np.cumsum(counts2)
            csp = csp / csp[-1]
            logDGR16, logDGR84 = np.interp([0.16, 0.84], csp, np.log10(dbins2))
            Mgas_DGR_16.append(10**logDGR16)
            Mgas_DGR_84.append(10**logDGR84)
            #
            # Plot them
            #
            xlim = (-4.5, 1.5)
            num_samples = len(is_)
            titles = ['n=' + str(n), r'$M_{\rm gas}$-weighted']
            fig, ax = plt.subplots(ncols=2, figsize=(7, 3))
            h = ax[0].hist(np.log10(DGRs), bins=np.log10(dbins), normed=True,
                           range=xlim)
            max_ = np.nanmax(h[0])
            ax[0].plot([np.log10(realize_DGR_16[-1])] * 2,
                       [0.0, max_], lw=0.7,
                       label='16%:'+str(round(np.log10(realize_DGR_16[-1]),
                                              2)))
            ax[0].plot([np.log10(realize_DGR_84[-1])] * 2,
                       [0.0, max_], lw=0.7,
                       label='84%:'+str(round(np.log10(realize_DGR_84[-1]),
                                              2)))
            ax[0].plot([np.log10(realize_DGR_EXP[-1])] * 2,
                       [0.0, max_], lw=0.7,
                       label='EXP:'+str(round(np.log10(realize_DGR_EXP[-1]),
                                              2)))
            h = ax[1].hist(logd, weights=w, bins=np.log10(dbins), normed=True,
                           range=xlim)
            max_ = np.nanmax(h[0])
            ax[1].plot([np.log10(Mgas_DGR_16[-1])] * 2,
                       [0.0, max_], lw=0.7,
                       label='16%:'+str(round(np.log10(Mgas_DGR_16[-1]), 2)))
            ax[1].plot([np.log10(Mgas_DGR_84[-1])] * 2,
                       [0.0, max_], lw=0.7,
                       label='84%:'+str(round(np.log10(Mgas_DGR_84[-1]), 2)))
            ax[1].plot([np.log10(Mgas_DGR_EXP[-1])] * 2,
                       [0.0, max_], lw=0.7,
                       label='EXP:'+str(round(np.log10(Mgas_DGR_EXP[-1]), 2)))
            for i in range(2):
                ax[i].set_xlabel(r'$\log_{10}$DGR')
                ax[i].set_yticks([])
                ax[i].set_title(titles[i])
                ax[i].set_xlim(xlim)
                ax[i].legend()
            fig.tight_layout()
            temp = str(rbin_num)
            if len(temp) == 1:
                temp = '0' + temp
            fn = 'output/realize_vs_simple_sum_' + method_abbr + temp + \
                '_num_samples=' + str(num_samples) + '.png'
            fig.savefig(fn)
        # Combine arrays to data frame. Save to .csv.
        df = pd.DataFrame()
        df['12+log(O/H)'] = xbins2
        df['realize_DGR_EXP'] = realize_DGR_EXP
        df['realize_DGR_16'] = realize_DGR_16
        df['realize_DGR_84'] = realize_DGR_84
        df['Mgas_DGR_EXP'] = Mgas_DGR_EXP
        df['Mgas_DGR_16'] = Mgas_DGR_16
        df['Mgas_DGR_84'] = Mgas_DGR_84
        df.to_csv('data/Tables/DGR-to-metals_' + method_abbr + '.csv',
                  index=False)

    def realize_vs_Mgas_PDF2(self, name='NGC5457', method_abbr='BE', nop=10,
                             n=20000, timeout=1e-6, rbin_num=0):
        np.random.seed(int(time.time()))
        plt.close('all')
        binlist = self.d[name]['binlist']
        aRadius = self.d[name]['aRadius']
        aGas = self.d[name]['aGas']
        aPDFs = self.d[name][method_abbr]['aPDFs']
        aArea = np.array([(self.d[name]['binmap'] == binlist[i]).sum()
                          for i in range(len(binlist))])
        for i in range(len(aPDFs)):
            aPDFs[i] /= np.nansum(aPDFs[i])
        rbins = np.linspace(np.min(aRadius), np.max(aRadius), self.rbin)
        dbins = np.logspace(np.log10(3E-5), np.log10(6E-2), self.dbin)
        dbins2 = np.sqrt(dbins[:-1] * dbins[1:])
        logsigma_step = 0.025
        min_logsigma = -4.
        max_logsigma = 1.
        half_step = logsigma_step / 2
        rand_diff = max_logsigma - min_logsigma
        rand_min = min_logsigma - half_step
        total_step = int(n // nop)

        #
        rmin, rmax = rbins[rbin_num], rbins[rbin_num + 1]
        is_fixed = []
        for i in range(len(binlist)):
            if rmin <= aRadius[i] <= rmax:
                is_fixed.append(i)
        #
        for j in range(1, len(is_fixed) + 1):
            is_ = is_fixed[:j]
            d, w = np.array([]), np.array([])
            total_gas = 0
            for i in is_:
                mask = aPDFs[i] > aPDFs[i].max() / 1000
                temp_P = aPDFs[i][mask]
                temp_P = temp_P / np.sum(temp_P) * aGas[i] * aArea[i]
                d = np.append(d, self.SigmaDs[mask] / aGas[i])
                w = np.append(w, temp_P)
                total_gas += aGas[i] * aArea[i]
            logd = np.log10(d)
            #
            # Calculate random realize PDF.
            #

            def mp_realize(pid, total_step, total_gas, output):
                DGRs = []
                for step in range(total_step):
                    if step % 1000 == 0:
                        print('Processor', pid, 'at step', step)
                    total_dust = 0
                    for i in is_:
                        while(True):
                            rand_logSigmaD, rand_p = np.random.rand(2)
                            rand_logSigmaD = \
                                rand_min + rand_diff * rand_logSigmaD
                            pos = int((rand_logSigmaD - rand_min) //
                                      logsigma_step)
                            if aPDFs[i][pos] > rand_p:
                                total_dust += (10**rand_logSigmaD) * aArea[i]
                                break
                    DGRs.append(total_dust / total_gas)
                output.put((pid, DGRs))

            q = mp.Queue()
            processes = [mp.Process(target=mp_realize,
                         args=(pid, total_step, total_gas, q))
                         for pid in range(nop)]
            for p in processes:
                p.start()
            for p in processes:
                p.join(timeout)
            DGRs = []
            for p in processes:
                pid, result = q.get()
                print("--Got result from process", pid)
                DGRs += result
                del pid, result
            del processes, q
            # Calculate and save Realize 16-EXP-84 to array
            counts, _ = np.histogram(DGRs, bins=dbins)
            rlogDGREXP = np.sum(counts * np.log10(dbins2)) / np.sum(counts)
            csp = np.cumsum(counts)
            csp = csp / csp[-1]
            rlogDGR16, rlogDGR84 = \
                np.interp([0.16, 0.84], csp, np.log10(dbins2))
            # Calculate and save M_gas-weighted 16-EXP-84 to array
            counts2, _ = np.histogram(d, bins=dbins, weights=w)
            glogDGREXP = np.sum(counts2 * np.log10(dbins2)) / np.sum(counts2)
            csp = np.cumsum(counts2)
            csp = csp / csp[-1]
            glogDGR16, glogDGR84 = \
                np.interp([0.16, 0.84], csp, np.log10(dbins2))
            #
            # Plot them
            #
            xlim = (-4.5, 1.5)
            num_samples = len(is_)
            titles = ['n=' + str(n), r'$M_{\rm gas}$-weighted']
            fig, ax = plt.subplots(ncols=2, figsize=(7, 3))
            h = ax[0].hist(np.log10(DGRs), bins=np.log10(dbins), normed=True,
                           range=xlim)
            max_ = np.nanmax(h[0])
            ax[0].plot([rlogDGR16] * 2,
                       [0.0, max_], lw=0.7,
                       label='16%:'+str(round(rlogDGR16, 2)))
            ax[0].plot([rlogDGR84] * 2,
                       [0.0, max_], lw=0.7,
                       label='84%:'+str(round(rlogDGR84, 2)))
            ax[0].plot([rlogDGREXP] * 2,
                       [0.0, max_], lw=0.7,
                       label='EXP:'+str(round(rlogDGREXP, 2)))
            h = ax[1].hist(logd, weights=w, bins=np.log10(dbins), normed=True,
                           range=xlim)
            max_ = np.nanmax(h[0])
            ax[1].plot([glogDGR16] * 2,
                       [0.0, max_], lw=0.7,
                       label='16%:'+str(round(glogDGR16, 2)))
            ax[1].plot([glogDGR84] * 2,
                       [0.0, max_], lw=0.7,
                       label='84%:'+str(round(glogDGR84, 2)))
            ax[1].plot([glogDGREXP] * 2,
                       [0.0, max_], lw=0.7,
                       label='EXP:'+str(round(glogDGREXP, 2)))
            for i in range(2):
                ax[i].set_xlabel(r'$\log_{10}$DGR')
                ax[i].set_yticks([])
                ax[i].set_title(titles[i])
                ax[i].set_xlim(xlim)
                ax[i].legend()
            fig.tight_layout()
            temp, temp2 = str(rbin_num), str(num_samples)
            if len(temp) == 1:
                temp = '0' + temp
            if len(temp2) == 1:
                temp2 = '0' + temp2
            fn = 'output/realize_vs_simple_sum_' + method_abbr + temp + \
                '_num_samples=' + temp2 + '.png'
            fig.savefig(fn)
