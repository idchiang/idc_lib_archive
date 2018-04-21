from time import ctime
from h5py import File
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from corner import corner
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
from .idc_fitting import fwhm_sp500, fit_DataY, cali_mat2
from .idc_voronoi import voronoi_m
from decimal import Decimal
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

    def Load_and_Sum(self, names, method_abbrs):
        for name in names:
            for method_abbr in method_abbrs:
                self.Load_Data(name, method_abbr)
                self.Method_Summary(name, method_abbr)

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

    def Method_Summary(self, name, method_abbr):
        print('')
        print('################################################')
        print('Summary plots for', name, '-', method_abbr + ' (' + ctime() +
              ')')
        print('################################################')
        #
        # \Sigma_D, T, \beta, reduced_\chi2
        #
        self.STBC(name, method_abbr)
        #
        # DGR PDF Profiles & vs. metallicity
        #
        self.pdf_profiles(name, method_abbr)
        #
        # Temperature PDF Profiles & vs. metallicity
        #
        self.temperature_profiles(name, method_abbr)
        #
        # Residual and reduced_\chi2
        #
        self.residual_maps(name, method_abbr)
        #
        # Single pixel model & PDF
        #
        self.example_model(name, method_abbr)
        #
        # Parameter corner plots
        #
        self.corner_plots(name, method_abbr)

    def extra_plots(self, name):
        print('')
        print('################################################')
        print('Extra plots for', name)
        print('################################################')
        #
        # Radial profile of S_SFR and S_*
        #
        self.SFR_and_starlight(name)
        #
        # Correlation plots
        #
        self.corr(name, 'WD')
        #
        # Chi^2 version of residual plot
        #
        self.residual_chi2(name)

    def STBC(self, name, method_abbr, err_selc=0.3):
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

    def pdf_profiles(self, name, method_abbr):
        plt.close('all')
        """
        1X1: DGR profile
        """
        # print(' --Plotting DGR/(O/H) profile...')
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        r = d = w = dm = np.array([])  # radius, dgr, weight
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
                      GD_dist) - 12.0)
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
        dbins2 = (dbins[:-1] + dbins[1:]) / 2
        rbins2 = (rbins[:-1] + rbins[1:]) / 2
        DGR_Median = DGR_LExp = DGR_Max = DGR_16 = DGR_84 = np.array([])
        n_zeromask = np.full(counts.shape[1], True, dtype=bool)
        for i in range(counts.shape[1]):
            if np.sum(counts[:, i]) > 0:
                counts[:, i] /= np.sum(counts[:, i])
                counts3[:, i] /= np.sum(counts3[:, i])
                csp = np.cumsum(counts[:, i])[:-1]
                csp = np.append(0, csp / csp[-1])
                ssd = np.interp([0.16, 0.5, 0.84], csp, dbins2)
                DGR_Median = np.append(DGR_Median, ssd[1])
                DGR_LExp = np.append(DGR_LExp, 10**np.sum(np.log10(dbins2) *
                                                          counts[:, i]))
                DGR_Max = np.append(DGR_Max, dbins2[np.argmax(counts[:, i])])
                DGR_16 = np.append(DGR_16, ssd[0])
                DGR_84 = np.append(DGR_84, ssd[2])
            else:
                n_zeromask[i] = False
        #
        """
        xbins2 = (8.715 - 0.027 * rbins2 * self.d[name]['R25'] * 7.4 / GD_dist)
        xbins2 = 10**(xbins2 - 12.0)
        #
        fig = plt.figure(figsize=(10, 7.5))
        plt.pcolormesh(rbins, dmbins, counts3, norm=LogNorm(),
                       cmap=self.cmap1, vmin=1E-3)
        plt.yscale('log')
        plt.colorbar()
        plt.plot(rbins2[n_zeromask], DGR_Median / xbins2[n_zeromask], 'r',
                 label='Median')
        plt.plot(rbins2[n_zeromask], DGR_LExp / xbins2[n_zeromask], 'g',
                 label='Expectation')
        plt.plot(rbins2[n_zeromask], DGR_Max / xbins2[n_zeromask], 'b',
                 label='Max')
        plt.ylim([10**(-1.2), 10**(2.2)])
        plt.xlabel(r'Radius ($R_{25}$)', size=16)
        plt.ylabel(r'DGR / (O/H)', size=16)
        plt.legend(fontsize=16)
        plt.title('Gas mass weighted DGR/(O/H) profile', size=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if self.fake:
            fn = 'output/_FAKE_DGR-metal-profile_' + name + '_' + \
                method_abbr + '.pdf'
        else:
            fn = 'output/_DGR-metal-profile_' + name + '_' + method_abbr + \
                '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        """
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
        oH2003 = 8.715 - 0.027 * funcH2(0.03) * self.d[name]['R25'] * 7.4 / \
            GD_dist
        #
        # My own fitting and data points
        #
        print(' --Plotting DGR vs. Metallicity...')
        fig, ax = plt.subplots(nrows=2, figsize=(5, 8))
        with np.errstate(invalid='ignore'):
            tempo = (8.715 - 0.027 * self.d[name]['aRadius'] *
                     self.d[name]['R25'] * 7.4 / GD_dist)
            tempd = 10**self.d[name][method_abbr]['alogSigmaD'] / \
                self.d[name]['aGas']
            tempe = 10**self.d[name][method_abbr]['aSigmaD_err']
        nonnanmask = ~np.isnan(tempo + tempd + tempe)
        tempo, tempd, tempe = \
            tempo[nonnanmask], tempd[nonnanmask], tempe[nonnanmask]
        yerr = np.array([tempd * (tempe - 1),
                         tempd * (1 - 1 / tempe)])
        yerr = np.array([tempd * (1 - 1 / tempe),
                         tempd * (tempe - 1)])
        ax[0].errorbar(tempo, tempd, yerr=yerr, alpha=0.3, ms=1, fmt='o',
                       elinewidth=0.5, label='Results')
        ax[0].fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                           color='lightgray',
                           label=r'one-$\sigma$ of combined PDF')
        #
        ax[1].fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                           color='lightgray')
        ax[1].plot(xbins2[n_zeromask], DGR_LExp, label='Exp', linewidth=3.0)
        ax[1].plot(xbins2[n_zeromask], DGR_full, '-.',
                   label='Fit', linewidth=3.0)
        ax[1].plot(xbins2[n_zeromask], DGR_part, '--',
                   label='Fit (High Z)', linewidth=3.0)
        xlim = ax[1].get_xlim()
        ylim = [3E-5, 6E-2]
        ax[1].set_xlabel('12 + log(O/H)', size=12)
        titles = ['(a)', '(b)']
        for i in range(2):
            ax[i].set_yscale('log')
            ax[i].set_ylabel('DGR', size=12)
            ax[i].set_xlim(xlim)
            ax[i].set_ylim(ylim)
            ax[i].legend(fontsize=12, framealpha=1.0, loc=4)
            ax[i].set_title(titles[i], size=16, x=0.1, y=0.85)
        fn = 'output/_DTM_C18_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        #
        # Independent R14 plot
        #
        print(' --Plotting DTM with R14...')
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                        color='lightgray')
        ax.plot(xbins2[n_zeromask], DGR_LExp, linewidth=3.0)
        ax.plot(o__, 10**(1.62 * x__ - 2.21), '--', linewidth=3.0,
                alpha=0.8, label='R14 power')
        ax.plot(o__, self.BPL_DGR(x__, 'MW'), ':', linewidth=3.0,
                alpha=0.8, label='R14 broken')
        ax.scatter(df['12+log(O/H)'], df['DGR_MW'], c='b', s=15,
                   label='R14 data')
        ax.set_yscale('log')
        ax.set_ylabel('DGR', size=12)
        ax.set_xlim([np.nanmin(df['12+log(O/H)']),
                     np.nanmax(df['12+log(O/H)'])])
        ax.set_ylim(ylim)
        ax.legend(fontsize=12, framealpha=0.5, loc=4)
        ax.set_xlabel('12 + log(O/H)', size=12)
        fn = 'output/_DTM_R14_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        #
        # Independent D14 plot
        #
        print(' --Plotting DTM with D14...')
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                        color='lightgray')
        ax.fill_between([zl + solar_oxygen_bundance,
                         zu + solar_oxygen_bundance], 3E-5, 6E-2,
                        color='green', alpha=0.2, label='D14 Z')
        ax.fill_between([oH2003, oH202], 3E-5, 6E-2, color='red', alpha=0.2,
                        label=r'D14 f(H$_2$)')
        ax.plot(xbins2[n_zeromask], DGR_LExp, linewidth=3.0)
        ax.plot(o__, 10**(x__) / 150, 'k', linewidth=3.0,
                alpha=0.6, label='D11')
        ax.set_yscale('log')
        ax.set_ylabel('DGR', size=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(fontsize=12, framealpha=1.0, loc=4)
        ax.set_xlabel('12 + log(O/H)', size=12)
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
        ax.plot(xbins2[n_zeromask], DGR_LExp, linewidth=2.0)
        fs = [0, 0.36, 1, 1E9]
        fs_str = ['0.00', '0.36', '1.00', 'inf']
        for f in range(4):
            y = self.Jenkins_F_DGR(o__, fs[f])
            text = r'F$_*$=' + fs_str[f]
            ax.plot(o__, y, alpha=0.7, linewidth=1.0, color='grey', ls='--')
            n = 13
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
        ax.set_yscale('log')
        ax.set_ylabel('DGR', size=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('12 + log(O/H)', size=12)
        fig.tight_layout()
        fn = 'output/_DTM_J09_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        """
        #
        # kappa160 vs. metallicity
        #
        oH2bd = 8.715 - 0.027 * funcH2(0.05) * self.d[name]['R25'] * 7.4 / \
            GD_dist
        fig, ax = plt.subplots(figsize=(5, 4))
        # ax[s].fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
        #                    color='lightgray')
        ratio = self.d['kappa160'][method_abbr] / \
            (10**(xbins2[n_zeromask] - solar_oxygen_bundance) / 150)
        ax.fill_between(xbins2[n_zeromask], DGR_16 * ratio, DGR_84 * ratio,
                        color='lightgray')
        ax.plot(xbins2[n_zeromask],
                DGR_LExp * ratio,
                linewidth=3.0)
        ax.set_xlim([oH2bd, np.nanmax(xbins2[n_zeromask])])
        ax.set_ylim([10, 40])
        ax.set_xlabel('12 + log(O/H)', size=12)
        ax.set_ylabel(r'$\kappa_{160}$ $(cm^2g^{-1})$')
        fig.tight_layout()
        fn = 'output/_k160-vs-Metallicity_' + name + '_' + method_abbr + \
            '.pdf'
        # with PdfPages(fn) as pp:
        #     pp.savefig(fig, bbox_inches='tight')
        mask = xbins2[n_zeromask] > oH2bd
        print('Max kappa_160:', np.nanmax((DGR_LExp * ratio)[mask]))
        print('Min kappa_160:', np.nanmin((DGR_LExp * ratio)[mask]))
        """

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
                    csp = np.cumsum(counts[:, i])[:-1]
                    csp = np.append(0, csp / csp[-1])
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
        yranges = [-0.3, 0.5]
        xranges = [[-0.5, 2.5], [0.0, 2.5], [0.0, 2.5], [-0.5, 2.0],
                   [-0.5, 1.5]]
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
                ax[i, m].hist2d(logdata, Res_d_data, cmap='Blues',
                                range=[xranges[i], yranges], bins=[50, 50])
                ax[i, m].set_ylim(yranges)
                if m == 0:
                    ax[i, m].set_ylabel('(Obs. - Fit) / Obs.', size=14)
                else:
                    ax[i, m].set_yticklabels([])
                if i < 4:
                    ax[i, m].set_xticklabels([])
                ax[i, m].text(0.1, 0.85, titles[i], size=14,
                              horizontalalignment='left',
                              transform=ax[i, m].transAxes)
                ax[i, m].text(0.9, 0.85, '(' + ms[m] + ')', size=14,
                              horizontalalignment='right',
                              transform=ax[i, m].transAxes)
                ax[i, m].minorticks_on()
            ax[4, m].set_xlabel(r'$\log$(Obs.)', size=14)
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
        """
        ax.set_ylim([1.0, np.nanmax(sed_best_plot) * 1.2])
        ax.set_yscale('log')
        if self.fake:
            fn = 'output/_FAKE_Model-log_' + str(self.x) + str(self.y) + \
                '_' + name + '_' + method_abbr + '.pdf'
        else:
            fn = 'output/_Model-log_' + str(self.x) + str(self.y) + '_' + \
                name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        """

    def corner_plots(self, name, method_abbr):
        plt.close('all')
        with np.errstate(invalid='ignore'):
            pacs100 = np.log10(self.d[name]['aSED'][:, 0])
            mask = ~np.isnan(pacs100)
        print(' --Plotting corner plot...')
        if method_abbr == 'SE':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['aBeta'][mask],
                                self.d[name][method_abbr]['archi2'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$T_d$', r'$\beta$',
                      r'$\tilde{\chi}^2$']
        elif method_abbr == 'FB':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['archi2'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$T_d$', r'$\tilde{\chi}^2$']
        elif method_abbr == 'BE':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['abeta2'][mask],
                                self.d[name][method_abbr]['archi2'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$T_d$', r'$\beta_2$',
                      r'$\tilde{\chi}^2$']
        elif method_abbr == 'WD':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['aWDfrac'][mask],
                                self.d[name][method_abbr]['archi2'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$T_d$', r'$f_w$',
                      r'$\tilde{\chi}^2$']
        elif method_abbr == 'PL':
            temp = self.d[name][method_abbr]['aalpha'][mask]
            temp[0] += 0.01
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                temp,
                                self.d[name][method_abbr]['aloggamma'][mask],
                                self.d[name][method_abbr]['alogUmin'][mask],
                                self.d[name][method_abbr]['archi2'][mask]])
            labels = [r'$\log(\Sigma_d)$', r'$\alpha$', r'$\log(\gamma)$',
                      r'$\log(U)_{min}$', r'$\tilde{\chi}^2$']
        mask2 = np.sum(~np.isnan(samples), axis=0).astype(bool)
        fig = corner(samples.T[mask2], labels=labels, quantities=(0.16, 0.84),
                     show_titles=True, title_kwargs={"fontsize": 12})
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
            dbins2 = (dbins[:-1] + dbins[1:]) / 2
            rbins2 = (rbins[:-1] + rbins[1:]) / 2
            DGR_Median = DGR_LExp = DGR_Max = DGR_16 = DGR_84 = np.array([])
            for i in range(counts.shape[1]):
                if np.sum(counts[:, i]) > 0:
                    counts[:, i] /= np.sum(counts[:, i])
                    csp = np.cumsum(counts[:, i])[:-1]
                    csp = np.append(0, csp / csp[-1])
                    ssd = np.interp([0.16, 0.5, 0.84], csp, dbins2)
                    DGR_Median = np.append(DGR_Median, ssd[1])
                    DGR_LExp = np.append(DGR_LExp,
                                         10**np.sum(np.log10(dbins2) *
                                                    counts[:, i]))
                    DGR_Max = np.append(DGR_Max,
                                        dbins2[np.argmax(counts[:, i])])
                    DGR_16 = np.append(DGR_16, ssd[0])
                    DGR_84 = np.append(DGR_84, ssd[2])
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
        #
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.set_xlim([np.nanmin(xbins2), np.nanmax(xbins2)])
        for i in range(len(DGRs.keys())):
            k = list(DGRs.keys())[i]
            xbins2 = (8.715 - 0.027 * rbins2[n_zeromasks[k]] *
                      self.d[name]['R25'] * 7.4 / GD_dist)
            ax.plot(xbins2, DGRs[k], label=k, linewidth=4.0,
                    alpha=0.8)
            ax.fill_between(xbins2, DGR16s[k], DGR84s[k],
                            alpha=0.13)
        ax.fill_between(o__, 10**(o__ - 12) * 16.0 / 1.008 / 0.51 / 1.36,
                        10**(o__ - 12) * 16.0 / 1.008 / 0.445 / 1.36,
                        alpha=0.7, label='MAX', hatch='/')
        ax.legend(fontsize=fs2, ncol=2, loc=4)
        ax.tick_params(axis='both', labelsize=fs2)
        ax.set_yscale('log')
        ax.set_xlabel('12 + log(O/H)', size=fs1)
        ax.set_ylabel('DGR (' + BF_method + ')', size=fs1)
        #
        """
        fig, ax = plt.subplots(2, 1, figsize=(6, 7.5))
        # ax[0].set_ylim([1E-5, 1E-1])
        ax[0].set_xlim([7.75, 8.75])
        ax[0].set_yscale('log')
        for k in DGRs.keys():
            ax[0].plot(xbins2, DGRs[k], label=k)
        # ax[0].set_xlabel('12 + log(O/H)', size=fs1)
        ax[0].set_ylabel('DGR (' + BF_method + ')', size=fs1)
        ax[0].plot(o__, self.max_possible_DGR(o__), alpha=0.6, label='MAX',
                   linestyle='--')
        ax[0].legend(fontsize=fs2, ncol=2)
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=fs2)
        #
        # ax[1].set_ylim([1E-5, 1E-1])
        ax[1].set_xlim([7.75, 8.75])
        ax[1].set_yscale('log')
        mbins2 = self.max_possible_DGR(xbins2)
        for k in DGRs.keys():
            ax[1].plot(xbins2, DGRs[k] / mbins2, label=k)
        zs = self.max_possible_DGR(solar_oxygen_bundance)
        line, = ax[1].plot(xbins2, [1. / 150 / zs] * len(xbins2), label='D14',
                           linestyle='--')
        ax[1].set_xlabel('12 + log(O/H)', size=fs1)
        ax[1].set_ylabel('Dust-to-metal (' + BF_method + ')', size=fs1)
        ax[1].legend(handles=[line], fontsize=fs2, loc=4)
        ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=fs2)
        ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=fs2)
        """
        fig.tight_layout()
        fn = 'output/_DGR-vs-Metallicity_' + name + '_z_merged.pdf'
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
                    csp = np.cumsum(counts[:, i])[:-1]
                    csp = np.append(0, csp / csp[-1])
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
        ax[0].set_ylabel(r'T (K)', size=fs1)
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
        ax2 = ax[1].twinx()
        ax[1].plot(R_SMSD, SMSD_profile, c='b')
        ax[1].tick_params('y', colors='b')
        ax2.plot(R_SFR, SFR_profile, 'r')
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

    def voronoi_plot(self, name, nwl=5, targetSN=5):
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
        bin_s = np.sum((-1.26 >= logsigmas) * (logsigmas >= -1.56))
        range_s = (np.nanmin(logsigmas[logsigmas >= -1.56]),
                   np.nanmax(logsigmas[logsigmas <= -1.26]))
        bin_T = np.sum((22.2 >= self.Ts) * (self.Ts >= 19.2))
        range_T = (np.nanmin(self.Ts[self.Ts >= 19.2]),
                   np.nanmax(self.Ts[self.Ts <= 22.2]))
        bin_beta = np.sum((1.6 >= self.beta2s) * (self.beta2s >= 0.9))
        range_beta = (np.nanmin(self.beta2s[self.beta2s >= 0.9]),
                      np.nanmax(self.beta2s[self.beta2s <= 1.6]))
        labels = [r'$\log(\Sigma_d)$', r'$T_d$', r'$\beta_2$']
        fig2 = corner(samplesBE.T, labels=labels, weights=weightsBE,
                      quantities=(0.16, 0.84), show_titles=True,
                      plot_contours=False,
                      title_kwargs={"fontsize": 12},
                      bins=[bin_s, bin_T, bin_beta],
                      range=[range_s, range_T, range_beta],
                      norm=LogNorm())
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
        yerr_eg = np.array([2E0 * (1 - 1 / 10**np.mean(yerr_log)),
                            2E0 * (10**np.mean(yerr_log) - 1)]).reshape(2, 1)
        #
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(afH2, arelDGR, s=1)
        ax.errorbar(4E-2, 2E0, yerr=yerr_eg, fmt='o', color='g', ms=2,
                    elinewidth=1)
        ax.plot(afH2, arelDGR_fit, 'r')
        ax.set_ylabel('DGR (result) / DGR (MW)', size=12)
        ax.set_yscale('log')
        ax.set_xlabel(r'$f_{H_2}$', size=12)
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.set_yticks([num / 10 for num in range(6, 25)], minor=True)
        ax.set_yticklabels([], minor=True)
        ax.set_yticks([1, 2], minor=False)
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
        min_heracles = np.abs(np.nanmin(heracles))
        heracles[heracles < min_heracles] = np.nan
        aH2 = np.array([np.mean(heracles[self.d[name]['binmap'] == bin_])
                        for bin_ in self.d[name]['binlist']])
        afH2 = aH2 / self.d[name]['aGas']
        aSFR = bin2list(self.d[name]['SFR'], self.d[name]['binlist'],
                        self.d[name]['binmap'])
        aSMSD = bin2list(self.d[name]['SMSD'], self.d[name]['binlist'],
                         self.d[name]['binmap'])
        logxdata = np.log10(np.array([afH2, aSFR, aSMSD]))
        xnames = [r'log$_{10}$f(H$_2$) Residual',
                  r'log$_{10}$ $\Sigma_{\rm SFR}$ Residual',
                  r'log$_{10}$ $\Sigma_\star$ Residual']
        xtitles = ['fH2', 'SFR', 'SMSD']
        aRadius = self.d[name]['aRadius'] * self.d[name]['R25']
        alogMetal = (8.715 - 0.027 * aRadius * 7.4 / GD_dist)
        aDGR = 10**self.d[name][method_abbr]['alogSigmaD'] / \
            self.d[name]['aGas']
        aDTM = aDGR / (10**alogMetal)
        del alogMetal
        #
        mask = ~np.isnan(np.sum(logxdata, axis=0) + aDGR + aDTM)
        yerr_log = self.d[name][method_abbr]['aSigmaD_err'][mask]
        logxdata = logxdata[:, mask]
        logDGR = np.log10(aDGR[mask])
        logDTM = np.log10(aDTM[mask])
        Radius = aRadius[mask]
        del aDGR, aDTM, afH2, aSFR, aSMSD
        #
        df = pd.DataFrame()
        for i in range(len(logxdata)):
            temp, coef_ = fit_DataY(Radius, logxdata[i], logxdata[i] * 0.1)
            df[xnames[i]] = logxdata[i] - temp
        logDGR_R, coef2_ = fit_DataY(Radius, logDGR, logDGR * 0.1)
        logDTM_R, coef2_ = fit_DataY(Radius, logDTM, logDTM * 0.1)
        df[r'log$_{10}$DGR'] = logDGR - logDGR_R
        df[r'log$_{10}$DTM'] = logDTM - logDTM_R
        #
        """
        fig = plt.figure()
        sns.heatmap(df.corr(), annot=True, cmap='Reds')
        fn = 'output/_residual_corr_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        """
        #
        for i in range(len(xnames)):
            x = xnames[i]
            fig, ax = plt.subplots()
            ax.scatter(df[x], df[r'log$_{10}$DTM'], s=1, label='data')
            ax.errorbar(0.9 * np.nanmax(df[x]), -0.15, fmt='o',
                        yerr=np.nanmean(yerr_log),
                        color='g', ms=2, elinewidth=1, label='mean unc')
            ax.set_xlabel(x)
            ax.set_ylabel(r'log$_{10}$DTM')
            ax.legend()
            #
            # Pearson and p-values
            corr = stats.pearsonr(df[x], df[r'log$_{10}$DTM'])
            ax.text(1.01, 0.9,
                    'Pearson=' + str(round(corr[0], 3)),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(1.01, 0.85,
                    'p-value=' + str(round(corr[1], 3)),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            #
            # Shuffle Pearson
            temp_x = df[x].values
            corrs = []
            for j in range(1000):
                np.random.shuffle(temp_x)
                corrs.append(stats.pearsonr(temp_x, df[r'log$_{10}$DTM'])[0])
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
            corr = stats.spearmanr(df[x].values, df[r'log$_{10}$DTM'].values)
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
            for j in range(1000):
                temp_y = np.random.lognormal(mean=df[r'log$_{10}$DTM'].values,
                                             sigma=yerr_log)
                corrs.append(stats.pearsonr(df[x], temp_y)[0])
            ax.text(1.01, 0.4,
                    'Noise Pearson Max=' + str(round(np.max(corrs), 2)),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            #
            fn = 'output/_residual_trend_' + xtitles[i] + '_' + name + '_' + \
                method_abbr + '.pdf'
            with PdfPages(fn) as pp:
                pp.savefig(fig, bbox_inches='tight')

    def chi2_experiments(self, name='NGC5457', method_abbr='BE'):
        plt.close('all')
        DoFs = {'SE': 2, 'FB': 3, 'BE': 2, 'WD': 2, 'PL': 1}
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Regrid']
            bkgcov = grp['HERSCHEL_011111_BKGCOV'].value
            grp = hf['Fitting_results']
            subgrp = grp[method_abbr]
            aSED_fit = subgrp['Best_fit_sed'].value
        aSED = self.d[name]['aSED']
        #
        print(' --Plotting chi2 by varying \sigma_{bkg}...')
        bkgdn = [1, 2, 3, 4, 5, 6]
        achi2_bkgdn = []
        for i in range((len(bkgdn))):
            achi2_bkgdn.append([])
        for i in range(len(aSED)):
            sed_vec = aSED[i].reshape(1, 5)
            calcov = sed_vec.T * cali_mat2 * sed_vec
            bin_ = self.d[name]['binmap'] == self.d[name]['binlist'][i]
            bkgcov_avg = bkgcov / np.sum(bin_)
            diff = aSED_fit[i] - aSED[i]
            for j in range(len(bkgdn)):
                bkgcov_avg_dn = bkgcov_avg / (bkgdn[j]**2)
                cov_n1 = np.linalg.inv(bkgcov_avg_dn + calcov)
                temp = np.empty(5)
                for k in range(5):
                    temp[k] = np.sum(diff * cov_n1[:, k])
                chi2 = np.sum(temp * diff)
                achi2_bkgdn[j].append(chi2)
        #
        rows, columns = len(bkgdn), 2
        fig, ax = plt.subplots(rows, columns, figsize=(6, 12))
        yrange = [0, 6.0]
        y = np.linspace(0, 6.0)
        xranges = [[-0.5, 2.5], [0.0, 1.0]]
        with np.errstate(invalid='ignore'):
            logPACS100 = np.log10(self.d[name]['aSED'][:, 0])
        for i in range(rows):
            c2 = np.array(achi2_bkgdn[i]) / DoFs[method_abbr]
            d = str(bkgdn[i])
            avg_c2 = round(np.nanmean(c2), 2)
            nonnanmask = ~np.isnan(logPACS100 + c2)
            ax[i, 0].hist2d(logPACS100[nonnanmask],
                            c2[nonnanmask], cmap='Blues',
                            range=[xranges[0], yrange], bins=[50, 50])
            ax[i, 0].set_ylabel(r'$\tilde{\chi}^2$', size=12)
            ax[i, 0].set_title(method_abbr + ' BKG / ' + d,
                               x=0.3, y=0.8, size=12)
            ax[i, 0].minorticks_on()
            ax[i, 1].hist(c2, orientation='horizontal',
                          range=yrange, bins=50, normed=True,
                          label='Mean: ' + str(avg_c2))
            ax[i, 1].plot(stats.chi2.pdf(y * DoFs[method_abbr],
                                         DoFs[method_abbr]) *
                          DoFs[method_abbr], y,
                          label=r'$\tilde{\chi}^2$ dist. (k=' +
                          str(DoFs[method_abbr]) + ')')
            ax[i, 1].legend(fontsize=12)
            ax[i, 1].set_ylim(yrange)
            ax[i, 1].minorticks_on()
        ax[4, 0].set_xlabel(r'$\log$(PACS100)', size=12)
        ax[4, 1].set_xlabel('Frequency', size=12)
        fig.tight_layout()
        fn = 'output/_Chi2Exp_ReduceBKG_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
        #
        print(' --Plotting chi2 contributions...')
        achi2_100 = []
        achi2_160 = []
        achi2_250 = []
        achi2_350 = []
        achi2_500 = []
        achi2_od = []
        achi2 = []
        for i in range(len(aSED)):
            sed_vec = aSED[i].reshape(1, 5)
            calcov = sed_vec.T * cali_mat2 * sed_vec
            bin_ = self.d[name]['binmap'] == self.d[name]['binlist'][i]
            bkgcov_avg = bkgcov / np.sum(bin_)
            diff = aSED_fit[i] - aSED[i]
            cov = bkgcov_avg + calcov
            cov_n1 = np.linalg.inv(cov)
            temp = np.empty(5)
            for k in range(5):
                temp[k] = np.sum(diff * cov_n1[:, k])
            chi2 = np.sum(temp * diff) / DoFs[method_abbr]
            achi2_100.append(diff[0]**2 / cov[0, 0] / DoFs[method_abbr])
            achi2_160.append(diff[1]**2 / cov[1, 1] / DoFs[method_abbr])
            achi2_250.append(diff[2]**2 / cov[2, 2] / DoFs[method_abbr])
            achi2_350.append(diff[3]**2 / cov[3, 3] / DoFs[method_abbr])
            achi2_500.append(diff[4]**2 / cov[4, 4] / DoFs[method_abbr])
            achi2_od.append(chi2 - achi2_100[-1] - achi2_160[-1] -
                            achi2_250[-1] - achi2_350[-1] - achi2_500[-1])
            achi2.append(chi2)
        achi2_100 = np.array(achi2_100)
        achi2_160 = np.array(achi2_160)
        achi2_250 = np.array(achi2_250)
        achi2_350 = np.array(achi2_350)
        achi2_500 = np.array(achi2_500)
        achi2_od = np.array(achi2_od)
        achi2 = np.array(achi2)
        print('PACS100:', np.sum(achi2_100) / np.sum(achi2))
        print('PACS160:', np.sum(achi2_160) / np.sum(achi2))
        print('SPIRE_250:', np.sum(achi2_250) / np.sum(achi2))
        print('SPIRE_350:', np.sum(achi2_350) / np.sum(achi2))
        print('SPIRE_500:', np.sum(achi2_500) / np.sum(achi2))
        print('Off-diag:', np.sum(achi2_od) / np.sum(achi2))
        #
        rows, columns = 3, 2
        fig, ax = plt.subplots(rows, columns, figsize=(4, 6))
        images = [achi2_100 / achi2,
                  achi2_160 / achi2,
                  achi2_250 / achi2,
                  achi2_350 / achi2,
                  achi2_500 / achi2,
                  achi2_od / achi2]
        titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350',
                  'SPIRE500', 'Off-diagonal']
        p = [0, 0, 1, 1, 2, 2]
        q = [0, 1, 0, 1, 0, 1]
        for i in range(6):
            cax = ax[p[i], q[i]].imshow(list2bin(images[i],
                                        self.d[name]['binlist'],
                                        self.d[name]['binmap']),
                                        origin='lower', vmax=0.5)
            plt.colorbar(cax, ax=ax[p[i], q[i]])
            ax[p[i], q[i]].set_title(titles[i])
            ax[p[i], q[i]].set_xticks([])
            ax[p[i], q[i]].set_yticks([])
        fig.tight_layout()
        fn = 'output/_Chi2Exp_Contribution_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig, bbox_inches='tight')
