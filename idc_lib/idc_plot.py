from time import ctime
from h5py import File
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
from .idc_functions import map2bin, list2bin, SEMBB, BEMBB, WD, PowerLaw
from .idc_fitting import fwhm_sp500, fit_DataY
from .idc_voronoi import voronoi_m
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
            cd['logSigmaD'] = \
                list2bin(cd['alogSigmaD'], self.d[name]['binlist'],
                         self.d[name]['binmap'])
            cd['aSigmaD_err'] = subgrp['Dust_surface_density_err_dex'].value
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

    def STBC(self, name, method_abbr, use_mask=False):
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
        errs = {'logSigmaD': 'SigmaD_err',
                'T': 'T_err',
                'Beta': 'Beta_err',
                'beta2': 'beta2_err',
                'WDfrac': 'WDfrac_err',
                'alpha': 'alpha_err',
                'loggamma': 'loggamma_err',
                'logUmin': 'logUmin_err'}
        """
        maxs = {'logSigmaD': -0.5,
                'SigmaD_err': 1.5,
                'T': 35.0,
                'T_err': 20.0,
                'Beta': 3.5,
                'Beta_err': 2.0,
                'rchi2': 10.0,
                'beta2': 3.5,
                'beta2_err': 2.0,
                'WDfrac': 0.05,
                'WDfrac_err': 0.01,
                'alpha': 3.5,
                'alpha_err': 1.5,
                'loggamma': 0.0,
                'loggamma_err': 2.0,
                'logUmin': 1.1,
                'logUmin_err': 5.0}
        mins = {'logSigmaD': -3.5,
                'SigmaD_err': 0.0,
                'T': 10.0,
                'T_err': 0.0,
                'Beta': -0.5,
                'Beta_err': 0.0,
                'rchi2': 0.0,
                'beta2': -0.5,
                'beta2_err': 0.0,
                'WDfrac': 0.0,
                'WDfrac_err': 0.0,
                'alpha': 0.5,
                'alpha_err': 0.0,
                'loggamma': -4.0,
                'loggamma_err': 0.0,
                'logUmin': -3.7,
                'logUmin_err': 0.0}
        """
        #
        mask = np.zeros_like(self.d[name][method_abbr][params[0]])\
            .astype(bool)
        if use_mask:
            for p in params:
                image = self.d[name][method_abbr][p]
                err = self.d[name][method_abbr][errs[p]]
                if (p == 'loggamma') or (p == 'WDfrac'):
                    pass
                elif p[:3] == 'log':
                    mask += (err > 1.0)
                else:
                    mask += ((err / image) > 10.0)
        #
        rows, columns = 2, len(params)
        fig, ax = plt.subplots(rows, columns, figsize=(1.5 * columns, 3),
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
        cmap = 'viridis_r'
        cm = plt.cm.get_cmap(cmap)
        fitting_region = \
            (~np.isnan(self.d[name][method_abbr][params[0]])).astype(float)
        fitting_region[fitting_region == 0] = np.nan
        for i in range(columns):
            image = self.d[name][method_abbr][params[i]]
            image[mask] = np.nan
            max_, min_ = np.nanmax(image), np.nanmin(image)
            ax[0, i].imshow(fitting_region, origin='lower', cmap='Greys_r',
                            alpha=0.5)
            ax[0, i].imshow(image,
                            origin='lower', cmap=cmap,
                            vmax=max_, vmin=min_)
            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])
            ax[0, i].set_title(titles[params[i]], size=10)
            bins = np.linspace(min_, max_, 12)
            mask2 = ~np.isnan(image)
            image = image[mask2]
            n, bins, patches = \
                ax[1, i].hist(image, bins=bins,
                              weights=self.d[name]['SigmaGas'][mask2])
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)
            for j in range(len(col)):
                plt.setp(patches[j], 'facecolor', cm(col[j]))
            ax[1, i].set_yticks([])
            # ax[1, i].grid(True)
        fig.tight_layout()
        if self.fake:
            fn = 'output/_FAKE_STBC_' + name + '_' + method_abbr + '.pdf'
        else:
            fn = 'output/_STBC_' + name + '_' + method_abbr + '.pdf'
        with np.errstate(invalid='ignore'):
            with PdfPages(fn) as pp:
                pp.savefig(fig)

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
            pp.savefig(fig)
        """
        #
        # My linear fitting
        #
        xbins2 = (8.715 - 0.027 * rbins2 * self.d[name]['R25'] * 7.4 / GD_dist)
        DataX = xbins2[n_zeromask]
        DataY = np.log10(DGR_LExp)
        yerr = np.zeros_like(DGR_LExp)
        for yi in range(len(yerr)):
            yerr[yi] = max(np.log10(DGR_84[yi]) - np.log10(DGR_LExp[yi]),
                           np.log10(DGR_LExp[yi]) - np.log10(DGR_16[yi]))
        DGR_full, coef_ = fit_DataY(DataX, DataY, yerr)
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
        z_ = np.linspace(zl, zu, 50)
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
        oH205 = 8.715 - 0.027 * funcH2(0.5) * self.d[name]['R25'] * 7.4 / \
            GD_dist
        oH203 = 8.715 - 0.027 * funcH2(0.3) * self.d[name]['R25'] * 7.4 / \
            GD_dist
        """
        1X?: DGR vs. metallicity with other papers
        """
        print(' --Plotting DGR vs. Metallicity...')
        nrows = 3
        fig, ax = plt.subplots(nrows, 1, figsize=(5, 6),
                               gridspec_kw={'wspace': 0, 'hspace': 0})
        #
        # ax[0]: For results part. Full PDF, my fittings.
        #
        s = 0
        """
        ax[s].pcolor(xbins, dbins, counts, norm=LogNorm(),
                     cmap='Reds')
        """
        ax[s].fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                           color='lightgray')
        ax[s].plot(xbins2[n_zeromask], DGR_LExp, label='Exp', linewidth=3.0)
        ax[s].plot(xbins2[n_zeromask], DGR_full, '-.',
                   label='Fit', linewidth=3.0)
        ax[s].plot(xbins2[n_zeromask], DGR_part, '--',
                   label='Fit (High Z)', linewidth=3.0)
        #
        # ax[1]: Draine
        #
        s = 1
        ax[s].fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                           color='lightgray')
        ax[s].plot(xbins2[n_zeromask], DGR_LExp, linewidth=3.0)
        ax[s].plot(o__, 10**(x__) / 150, 'k', linewidth=3.0,
                   alpha=0.6, label='D11')
        ax[s].plot(z_ + solar_oxygen_bundance, 10**z_ / 150, 'c',
                   label='D14 Fit', linewidth=3.0)
        #
        # ax[2]: Remy-Ruyer
        #
        s = 2
        ax[s].fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                           color='lightgray')
        ax[s].plot(xbins2[n_zeromask], DGR_LExp, linewidth=3.0)
        ax[s].plot(o__, 10**(1.62 * x__ - 2.21), '--', linewidth=3.0,
                   alpha=0.6, label='R14 power')
        ax[s].plot(o__, self.BPL_DGR(x__, 'MW'), ':', linewidth=3.0,
                   alpha=0.6, label='R14 broken')
        # ax[s].scatter(df['12+log(O/H)'], df['DGR_MW'], c='b', s=15,
        #               label='R14 data')
        #
        titles = np.array(['(a)', '(b)', '(c)'])
        xlim = ax[0].get_xlim()
        ylim = [3E-5, 6E-2]
        for s in range(nrows):
            ax[s].plot([oH205] * 2, [3E-5, 6E-2], 'r',
                       label=r'50% H$_2$')
            ax[s].set_yscale('log')
            ax[s].set_ylabel('DGR', size=12)
            ax[s].set_xlim(xlim)
            ax[s].set_ylim(ylim)
            # ax[s].ticklabel_format(style='sci', axis='y')
            # ax[s].set_yticklabels(ax[s].get_yticks(), fontsize=12)
            ax[s].legend(fontsize=12, loc='center left',
                         bbox_to_anchor=(1.0, 0.5))
            ax[s].set_title(titles[s], size=14, x=0.1, y=0.8)
            if s < nrows - 1:
                ax[s].set_xticklabels([])
            else:
                # ax[s].set_xticklabels(ax[s].get_xticks(), fontsize=12)
                ax[s].set_xlabel('12 + log(O/H)', size=12)
        fig.tight_layout(rect=[0, 0, 0.7, 1])
        # ax.plot(o__, self.max_possible_DGR(o__),
        #         'r', alpha=0.6, label='Max possible DGR')
        if self.fake:
            fn = 'output/_FAKE_DGR-vs-Metallicity_' + name + '_' + \
                method_abbr + '.pdf'
        else:
            fn = 'output/_DGR-vs-Metallicity_' + name + '_' + method_abbr + \
                '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)
        #
        # Independent Jenkins plot
        #
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.fill_between(xbins2[n_zeromask], DGR_16, DGR_84,
                        color='lightgray')
        ax.plot(xbins2[n_zeromask], DGR_LExp, linewidth=2.0)
        ax.plot(o__, self.Jenkins_F_DGR(o__, 0),
                alpha=0.6, label=r'J09 F$_*$=0', linewidth=2.0)
        ax.plot(o__, self.Jenkins_F_DGR(o__, 0.36),
                alpha=0.6, label=r'J09 F$_*$=0.36', linewidth=2.0)
        ax.plot(o__, self.Jenkins_F_DGR(o__, 1),
                alpha=0.6, label=r'J09 F$_*$=1', linewidth=2.0)
        ax.plot(o__, self.Jenkins_F_DGR(o__, 100000000),
                alpha=0.6, label=r'J09 F$_*$=inf', linewidth=2.0)
        ax.plot([oH205] * 2, [3E-5, 6E-2], 'r',
                label=r'50% H$_2$')
        ax.plot([oH203] * 2, [3E-5, 6E-2],
                label=r'30% H$_2$')
        ax.set_yscale('log')
        ax.set_ylabel('DGR', size=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax[s].ticklabel_format(style='sci', axis='y')
        # ax[s].set_yticklabels(ax[s].get_yticks(), fontsize=12)
        ax.legend(fontsize=12)
        ax.set_xlabel('12 + log(O/H)', size=12)
        fig.tight_layout()
        fn = 'output/_DGR-vs-Metallicity_' + name + '_' + method_abbr + \
            '_Jenkins.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)
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
        ax.legend()
        fig.tight_layout()
        fn = 'output/_k160-vs-Metallicity_' + name + '_' + method_abbr + \
            '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)
        mask = xbins2[n_zeromask] > oH2bd
        print('Max kappa_160:', np.nanmax((DGR_LExp * ratio)[mask]))
        print('Min kappa_160:', np.nanmin((DGR_LExp * ratio)[mask]))

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
            pp.savefig(fig)

    def residual_maps(self, name, method_abbr):
        plt.close('all')
        print(' --Plotting residual maps...')
        titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']
        rows, columns = 5, 2
        fig, ax = plt.subplots(rows, columns, figsize=(8, 12))
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
            ax[i, 0].set_xlabel(r'$\log$(Obs.)')
            ax[i, 0].set_ylabel('(Obs. - Fit) / Obs.')
            ax[i, 0].set_title(titles[i])
            ax[i, 1].hist(Res_d_data, orientation='horizontal',
                          range=yranges, bins=50)
            ax[i, 1].set_xlabel('Count')
            ax[i, 1].set_ylim(yranges)
            ax[i, 1].set_title(titles[i])
        fig.tight_layout()
        if self.fake:
            fn = 'output/_FAKE_Residual_' + name + '_' + method_abbr + '.pdf'
        else:
            fn = 'output/_Residual_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)

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
            pp.savefig(fig)
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
            pp.savefig(fig)
        """

    def corner_plots(self, name, method_abbr):
        plt.close('all')
        with np.errstate(invalid='ignore'):
            pacs100 = np.log10(self.d[name]['aSED'][:, 0])
            mask = ~np.isnan(pacs100)
            pacs100 = pacs100[mask]
        print(' --Plotting corner plot...')
        if method_abbr == 'SE':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['aBeta'][mask],
                                self.d[name][method_abbr]['archi2'][mask],
                                pacs100])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$\beta$',
                      r'$\tilde{\chi}^2$', r'\log(PACS100)']
        elif method_abbr == 'FB':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['archi2'][mask],
                                pacs100])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$\tilde{\chi}^2$',
                      r'\log(PACS100)']
        elif method_abbr == 'FBPT':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['archi2'][mask],
                                pacs100])
            labels = [r'$\log(\Sigma_d)$', r'$\tilde{\chi}^2$',
                      r'\log(PACS100)']
        elif method_abbr == 'PB':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['archi2'][mask],
                                pacs100])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$\tilde{\chi}^2$',
                      r'\log(PACS100)']
        elif method_abbr == 'BEMFB':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['alambda_c'][mask],
                                self.d[name][method_abbr]['abeta2'][mask],
                                self.d[name][method_abbr]['archi2'][mask],
                                pacs100])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$\lambda_c$',
                      r'$\beta_2$', r'$\tilde{\chi}^2$', r'\log(PACS100)']
        elif method_abbr == 'BE':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['abeta2'][mask],
                                self.d[name][method_abbr]['archi2'][mask],
                                pacs100])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$\beta_2$',
                      r'$\tilde{\chi}^2$', r'\log(PACS100)']
        elif method_abbr == 'WD':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                self.d[name][method_abbr]['aT'][mask],
                                self.d[name][method_abbr]['aWDfrac'][mask],
                                self.d[name][method_abbr]['archi2'][mask],
                                pacs100])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$f_w$',
                      r'$\tilde{\chi}^2$', r'$\log(PACS100)$']
        elif method_abbr == 'PL':
            temp = self.d[name][method_abbr]['aalpha'][mask]
            temp[0] += 0.01
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'][mask],
                                temp,
                                self.d[name][method_abbr]['aloggamma'][mask],
                                self.d[name][method_abbr]['alogUmin'][mask],
                                self.d[name][method_abbr]['archi2'][mask],
                                pacs100])
            labels = [r'$\log(\Sigma_d)$', r'$\alpha$', r'$\log(\gamma)$',
                      r'$\log(U)_{min}$', r'$\tilde{\chi}^2$',
                      r'\log(PACS100)']
        fig = corner(samples.T, labels=labels, quantities=(0.16, 0.84),
                     show_titles=True, title_kwargs={"fontsize": 12})
        if self.fake:
            fn = 'output/_FAKE_Corner_' + name + '_' + method_abbr + '.pdf'
        else:
            fn = 'output/_Corner_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)

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
            pp.savefig(fig)

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
            pp.savefig(fig)
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
            pp.savefig(fig)

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
        fig, ax = plt.subplots(rows, columns, figsize=(7.5, 12))
        yrange = [0, 6.0]
        y = np.linspace(0, 6.0)
        xranges = [[-0.5, 2.5], [0.0, 1.0]]
        with np.errstate(invalid='ignore'):
            logPACS100 = np.log10(self.d[name]['aSED'][:, 0])
        for i in range(len(method_abbrs)):
            c2 = self.d[name][method_abbrs[i]]['archi2']
            avg_c2 = round(np.nanmean(c2), 2)
            exp_k = int(round(avg_c2 * DoFs[i]))
            if self.fake:
                with np.errstate(invalid='ignore'):
                    logPACS100 = \
                        np.log10(self.d[name][method_abbrs[i]]['aSED'][:, 0])
            nonnanmask = ~np.isnan(logPACS100 + c2)
            ax[i, 0].hist2d(logPACS100[nonnanmask],
                            c2[nonnanmask], cmap='Blues',
                            range=[xranges[0], yrange], bins=[50, 50])
            ax[i, 0].set_xlabel(r'$\log$(PACS100)')
            ax[i, 0].set_ylabel(r'$\tilde{\chi}^2$')
            ax[i, 0].set_title(method_abbrs[i] + ' (DoF=' + str(DoFs[i]) + ')')
            ax[i, 1].hist(c2, orientation='horizontal',
                          range=yrange, bins=50, normed=True,
                          label='Mean: ' + str(avg_c2) + '(Cal. k=' +
                          str(exp_k) + ')')
            ax[i, 1].plot(stats.chi2.pdf(y * DoFs[i], DoFs[i]) * DoFs[i], y,
                          label=r'$\tilde{\chi}^2$ dist. (k=' +
                          str(DoFs[i]) + ')')
            ax[i, 1].plot(stats.chi2.pdf(y * exp_k, exp_k) * exp_k, y,
                          label=r'$\tilde{\chi}^2$ dist. (k=' +
                          str(exp_k) + ')')
            ax[i, 1].legend()
            ax[i, 1].set_xlabel('Frequency')
            ax[i, 1].set_ylim(yrange)
        fig.tight_layout()
        if self.fake:
            fn = 'output/_FAKE_Residual_Chi2_' + name + '.pdf'
        else:
            fn = 'output/_Residual_Chi2_' + name + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)

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
            try:
                n_zeromask
            except NameError:
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
        xbins2 = (8.715 - 0.027 * rbins2[n_zeromask] * self.d[name]['R25'] *
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
            ax.plot(xbins2, DGRs[k], label=k, linewidth=4.0,
                    alpha=0.8)
            ax.fill_between(xbins2, DGR16s[k], DGR84s[k],
                            alpha=0.13)
        ax.plot(o__, self.max_possible_DGR(o__), 'b', label='MAX',
                linestyle='--', linewidth=4.0, alpha=0.8)
        ax.legend(fontsize=fs2, ncol=2, loc=4)
        ax.set_xticklabels(ax.get_xticks(), fontsize=fs2)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fs2)
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
            pp.savefig(fig)

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
        ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=fs2)
        ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=fs2)
        ax2.set_yticklabels(ax2.get_yticks(), fontsize=fs2)
        ax[1].set_yscale('log')
        ax2.set_yscale('log')
        #
        fig.tight_layout()
        fn = 'output/_T-profile_' + name + '_z_merged.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)

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
            pp.savefig(fig)

    def example_model_merged(self, name):
        i = np.argmax(self.d[name]['binlist'] ==
                      self.d[name]['binmap'][self.y, self.x])
        print(' --Plotting example model...')
        wl_complete = np.linspace(1, 800, 1000)
        wl_plot = np.linspace(51, 549, 100)
        method_abbrs = ['SE', 'FB', 'BE', 'WD', 'PL']
        fig, ax = plt.subplots(1, 5, figsize=(15, 3),
                               gridspec_kw={'wspace': 0, 'hspace': 0})
        for mi in range(5):
            method_abbr = method_abbrs[mi]
            with File('hdf5_MBBDust/Models.h5', 'r') as hf:
                models = hf[method_abbr].value
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
            unc_obs_plot = unc * ccf
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
            del temp_matrix, diff
            #
            # Selecting samples for plotting
            #
            chi2_threshold = 2.0
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
            transp = np.exp(-0.5 * chi2) * 0.3
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
                ax[mi].plot(wl_plot, model_plot, alpha=transp[j], color='k')
            ax[mi].plot(wl_plot, sed_best_plot, 'orange', linewidth=3,
                        label=method_abbr + ' exp')
            ax[mi].errorbar(wl, sed_obs_plot, yerr=unc_obs_plot, fmt='o',
                            color='red', capsize=10)
        for mi in range(5):
            ax[mi].set_ylim([1.0, 22.0])
            if mi > 0:
                ax[mi].set_yticklabels([])
            else:
                ax[mi].set_yticklabels(ax[mi].get_yticks(), fontsize=14)
                ax[mi].set_ylabel(r'SED ($MJy$ $sr^{-1}$)', size=18)
            ax[mi].set_xticks([200, 400])
            ax[mi].set_xticklabels([200, 400], fontsize=14)
            ax[mi].set_xlabel(r'Wavelength ($\mu m$)', size=18)
            ax[mi].legend(fontsize=16)
        fig.tight_layout()
        fn = 'output/_Model_' + str(self.x) + str(self.y) + '_' + \
            name + '_z_merged.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)

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
            pp.savefig(fig)

    def kappa160_fH2(self, name='NGC5457', method_abbr='BE'):
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
        ak160 = self.d['kappa160'][method_abbr] * aDGR / aD14_DGR
        #
        del aH2, heracles, aMetal, aD14_DGR, aDGR
        n_nanmask = ~np.isnan(afH2 + ak160)
        afH2, ak160 = afH2[n_nanmask], ak160[n_nanmask]
        afH2_log, ak160_log = np.log10(afH2), np.log10(ak160)
        ak160_fit = [[-1, -1], [-1, -1]]
        #
        yerr_log = self.d[name][method_abbr]['aSigmaD_err'][n_nanmask]
        yerr = np.empty_like(yerr_log)
        for i in range(len(yerr)):
            yerr[i] = max((10**yerr_log[i] - 1) * ak160[i],
                          (1 - 10**(1 - yerr_log[i])) * ak160[i])
        #
        ak160_fit[0][0], coef_ = fit_DataY(afH2, ak160, yerr)
        ak160_fit[0][1], coef_ = fit_DataY(afH2_log, ak160, yerr)
        ak160_fit[1][0], coef_ = fit_DataY(afH2, ak160_log, yerr_log)
        ak160_fit[1][1], coef_ = fit_DataY(afH2_log, ak160_log, yerr_log)
        ak160_fit[1][0] = 10**ak160_fit[1][0]
        ak160_fit[1][1] = 10**ak160_fit[1][1]
        titles = np.array([['(a)', '(b)'], ['(c)', '(d)']])
        #
        fig, ax = plt.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0})
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
            pp.savefig(fig)
        #
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(afH2, ak160, s=1)
        ax.plot(afH2, ak160_fit[1][1], 'r')
        ax.set_ylabel(r'$\kappa_{160}$ $(cm^2g^{-1})$', size=12)
        ax.set_yscale('log')
        ax.set_xlabel(r'$f_{H_2}$', size=12)
        ax.set_xscale('log')
        fig.tight_layout()
        fn = 'output/_ak160_vs_afH2_' + name + '_.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)
        print(max(ak160_fit[1][1]), min(ak160_fit[1][1]))
        print(min(afH2))
