from h5py import File
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from corner import corner
import astropy.units as u
from astropy.constants import c
from astropy.io import fits
from astropy.wcs import WCS
from .gal_data import gal_data
from .z0mg_RSRF import z0mg_RSRF
from .idc_functions import map2bin, list2bin, SEMBB, BEMBB, WD
plt.ioff()
solar_oxygen_bundance = 8.69  # (O/H)_\odot, ZB12
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
bands = ['PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', 'SPIRE_500']


class Dust_Plots(object):
    def __init__(self):
        self.d = {}
        self.cmap0 = 'gist_heat'
        self.cmap1 = 'Greys'
        self.cmap2 = 'seismic'
        self.rbin = 51
        self.dbin = 100
        self.tbin = 80
        self.x, self.y = 55, 55

    def Load_and_Sum(self, names, method_abbrs):
        for name in names:
            for method_abbr in method_abbrs:
                self.Load_Data(name, method_abbr)
                self.Method_Summary(name, method_abbr)

    def Load_Data(self, name, method_abbr):
        # Maybe change this to "load a method"
        try:
            self.d[name]
        except KeyError:
            cd = {}
            with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
                grp = hf['Bin']
                cd['binlist'] = grp['BINLIST'].value
                cd['binmap'] = grp['BINMAP'].value
                cd['aGas'] = grp['GAS_AVG'].value
                cd['SigmaGas'] = \
                    list2bin(cd['aGas'], cd['binlist'], cd['binmap'])
                cd['aSED'] = grp['Herschel_SED'].value
                cd['aRadius'] = grp['Radius_avg'].value
                cd['acov_n1'] = grp['Herschel_covariance_matrix'].value
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
        num_para = {'EF': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4,
                    'FBWD': 3, 'BEMFBFL': 3}
        with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
            grp = hf['Fitting_results']
            subgrp = grp[method_abbr]
            cd['alogSigmaD'] = subgrp['Dust_surface_density_log'].value
            cd['logSigmaD'] = \
                list2bin(cd['alogSigmaD'], self.d[name]['binlist'],
                         self.d[name]['binmap'])
            cd['SigmaD_err'] = \
                list2bin(subgrp['Dust_surface_density_err_dex'].value,
                         self.d[name]['binlist'], self.d[name]['binmap'])
            cd['aT'] = subgrp['Dust_temperature'].value
            cd['T'] = list2bin(cd['aT'], self.d[name]['binlist'],
                               self.d[name]['binmap'])
            cd['T_err'] = \
                list2bin(subgrp['Dust_temperature_err'].value,
                         self.d[name]['binlist'], self.d[name]['binmap'])
            cd['aBeta'] = subgrp['beta'].value
            cd['Beta'] = \
                list2bin(cd['aBeta'], self.d[name]['binlist'],
                         self.d[name]['binmap'])
            cd['Beta_err'] = \
                list2bin(subgrp['beta_err'].value, self.d[name]['binlist'],
                         self.d[name]['binmap'])
            cd['rchi2'] = \
                list2bin(subgrp['Chi2'].value, self.d[name]['binlist'],
                         self.d[name]['binmap']) / \
                (5.0 - num_para[method_abbr])
            cd['aSED'] = subgrp['Best_fit_sed'].value
            cd['aPDFs'] = subgrp['PDF'].value
            self.SigmaDs = 10**subgrp['logsigmas'].value
            if method_abbr in ['EF', 'FB', 'BEMFB', 'PB', 'FBWD', 'BEMFBFL']:
                cd['aPDFs_T'] = subgrp['PDF_T'].value
                self.Ts = subgrp['Ts'].value
            elif method_abbr in ['FBPT']:
                self.FBPT_Ts = subgrp['Ts'].value
            if method_abbr in ['EF']:
                cd['aPDFs_b'] = subgrp['PDF_B'].value
                self.betas = subgrp['betas'].value
            elif method_abbr in['PB']:
                self.PB_betas = subgrp['betas'].value
            if method_abbr in ['BEMFB', 'BEMFBFL']:
                cd['alambda_c'] = subgrp['Critical_wavelength'].value
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
            if method_abbr in ['FBWD']:
                cd['aWDfrac'] = subgrp['WDfrac'].value
                cd['WDfrac'] = list2bin(cd['aWDfrac'], self.d[name]['binlist'],
                                        self.d[name]['binmap'])
                cd['WDfrac_err'] = \
                    list2bin(subgrp['WDfrac_err'].value,
                             self.d[name]['binlist'], self.d[name]['binmap'])
                self.WDfracs = subgrp['WDfracs'].value
                cd['aPDFs_Wf'] = subgrp['PDF_Wf'].value
            if method_abbr in ['FBPT', 'PB']:
                cd['coef_'] = subgrp['coef_'].value
        self.d[name][method_abbr] = cd

    def Method_Summary(self, name, method_abbr):
        print('')
        print('################################################')
        print('Summary plots for', name, '-', method_abbr)
        print('################################################')
        #
        # \Sigma_D, T, \beta, reduced_\chi2
        #
        # self.STBC(name, method_abbr)
        #
        # DGR PDF Profiles & vs. metallicity
        #
        # self.pdf_profiles(name, method_abbr)
        #
        # Temperature PDF Profiles & vs. metallicity
        #
        # self.temperature_profiles(name, method_abbr)
        #
        # Residual and reduced_\chi2
        #
        # self.residual_maps(name, method_abbr)
        #
        # Single pixel model & PDF
        #
        self.example_model(name, method_abbr)
        #
        # Parameter corner plots
        #
        self.corner_plots(name, method_abbr)

    def STBC(self, name, method_abbr):
        plt.close('all')
        """
        4X2: \Sigma_D, T, \beta, reduced_\chi2
        """
        print(' --Plotting \Sigma_D, T, \\beta & \\tilde{\chi}^2...')
        rows, columns = 2, 4
        fig = plt.figure(figsize=(8, 12))
        titles = [r'$\log(\Sigma_D)$ $\log(M_\odot/pc^2)$',
                  r'$\log(\Sigma_D)$ error',
                  r'$T_D$ (K)', r'$T_D$ error',
                  r'$\beta$', r'$\beta$ error',
                  r'$\tilde{\chi}^2$']
        images = ['logSigmaD', 'SigmaD_err',
                  'T', 'T_err',
                  'Beta', 'Beta_err',
                  'rchi2']
        maxs = [-0.5, 1.5,
                30.0, 15.0,
                3.5, 2.0,
                10.0]
        mins = [-3.5, 0.0,
                10.0, 0.0,
                -0.5, 0.0,
                0.0]
        for i in range(7):
            sub_ = columns * 100 + rows * 10 + i + 1
            ax = fig.add_subplot(sub_, projection=self.d[name]['WCS'])
            cax = ax.imshow(self.d[name][method_abbr][images[i]],
                            origin='lower', cmap=self.cmap0,
                            vmax=maxs[i], vmin=mins[i])
            ax.coords[0].set_major_formatter('hh:mm')
            ax.coords[1].set_major_formatter('dd:mm')
            plt.colorbar(cax, ax=ax)
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[0].set_ticks_visible(False)
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_ticks_visible(False)
            plt.title(titles[i], fontdict={'fontsize': 16})
            if i in [5, 6]:
                plt.xlabel('RA')
                ax.coords[0].set_ticklabel_visible(True)
                ax.coords[0].set_ticks_visible(True)
            if i in [0, 2, 4, 6]:
                plt.ylabel('Dec')
                ax.coords[1].set_ticklabel_visible(True)
                ax.coords[1].set_ticks_visible(True)
        fig.tight_layout(pad=5.0, w_pad=2.0, h_pad=3.5)
        fn = 'output/_STBC_' + name + '_' + method_abbr + '.pdf'
        with np.errstate(invalid='ignore'):
            with PdfPages(fn) as pp:
                pp.savefig(fig)
        """
        3X2: \almbda_c, \beta_2, reduced_\chi2
        """
        if method_abbr[:3] == 'BEM':
            print(' --Plotting \lambda_c, \\beta_2 & \\tilde{\chi}^2...')
            rows, columns = 2, 3
            fig = plt.figure(figsize=(8, 10))
            titles = [r'$\lambda_c$ $(\mu m)$', r'$\lambda_c$ error',
                      r'$\beta_2$', r'$\beta_2$ error',
                      r'$\tilde{\chi}^2$']
            images = ['lambda_c', 'lambda_c_err',
                      'beta2', 'beta2_err',
                      'rchi2']
            maxs = [460, 100,
                    3.5, 2.0,
                    10.0]
            mins = [100, 0.0,
                    -0.5, 0.0,
                    0.0]
            for i in range(5):
                sub_ = columns * 100 + rows * 10 + i + 1
                ax = fig.add_subplot(sub_, projection=self.d[name]['WCS'])
                cax = ax.imshow(self.d[name][method_abbr][images[i]],
                                origin='lower', cmap=self.cmap0,
                                vmax=maxs[i], vmin=mins[i])
                ax.coords[0].set_major_formatter('hh:mm')
                ax.coords[1].set_major_formatter('dd:mm')
                plt.colorbar(cax, ax=ax)
                ax.coords[0].set_ticklabel_visible(False)
                ax.coords[0].set_ticks_visible(False)
                ax.coords[1].set_ticklabel_visible(False)
                ax.coords[1].set_ticks_visible(False)
                plt.title(titles[i], fontdict={'fontsize': 16})
                if i in [3, 4]:
                    plt.xlabel('RA')
                    ax.coords[0].set_ticklabel_visible(True)
                    ax.coords[0].set_ticks_visible(True)
                if i in [0, 2, 4]:
                    plt.ylabel('Dec')
                    ax.coords[1].set_ticklabel_visible(True)
                    ax.coords[1].set_ticks_visible(True)
            fig.tight_layout(pad=5.0, w_pad=2.0, h_pad=3.5)
            fn = 'output/_STBC_BEM_' + name + '_' + method_abbr + '.pdf'
            with np.errstate(invalid='ignore'):
                with PdfPages(fn) as pp:
                    pp.savefig(fig)
        elif method_abbr[-2:] == 'WD':
            print(' --Plotting WDfrac & \\tilde{\chi}^2...')
            rows, columns = 2, 3
            fig = plt.figure(figsize=(8, 6))
            titles = [r'$WD fraction$', r'$WD fraction$ error',
                      r'$\tilde{\chi}^2$']
            images = ['WDfrac', 'WDfrac_err',
                      'rchi2']
            maxs = [0.05, 0.01,
                    10.0]
            mins = [0, 0.0,
                    0.0]
            for i in range(3):
                sub_ = columns * 100 + rows * 10 + i + 1
                ax = fig.add_subplot(sub_, projection=self.d[name]['WCS'])
                cax = ax.imshow(self.d[name][method_abbr][images[i]],
                                origin='lower', cmap=self.cmap0,
                                vmax=maxs[i], vmin=mins[i])
                ax.coords[0].set_major_formatter('hh:mm')
                ax.coords[1].set_major_formatter('dd:mm')
                plt.colorbar(cax, ax=ax)
                ax.coords[0].set_ticklabel_visible(False)
                ax.coords[0].set_ticks_visible(False)
                ax.coords[1].set_ticklabel_visible(False)
                ax.coords[1].set_ticks_visible(False)
                plt.title(titles[i], fontdict={'fontsize': 16})
                if i in [1, 2]:
                    plt.xlabel('RA')
                    ax.coords[0].set_ticklabel_visible(True)
                    ax.coords[0].set_ticks_visible(True)
                if i in [0, 2]:
                    plt.ylabel('Dec')
                    ax.coords[1].set_ticklabel_visible(True)
                    ax.coords[1].set_ticks_visible(True)
            fig.tight_layout(pad=5.0, w_pad=2.0, h_pad=3.5)
            fn = 'output/_STBC_WD_' + name + '_' + method_abbr + '.pdf'
            with np.errstate(invalid='ignore'):
                with PdfPages(fn) as pp:
                    pp.savefig(fig)

    def pdf_profiles(self, name, method_abbr):
        plt.close('all')
        """
        1X1: DGR profile
        """
        print(' --Plotting DGR profile...')
        r = d = w = np.array([])  # radius, dgr, weight
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
            d = np.append(d, temp_DGR)
            w = np.append(w, temp_P)
        nanmask = np.isnan(r + d + w)
        r, d, w = r[~nanmask], d[~nanmask], w[~nanmask]
        rbins = np.linspace(np.min(r), np.max(r), self.rbin)
        dbins = \
            np.logspace(np.min(np.log10(d)), np.max(np.log10(d)), self.dbin)
        # Counting hist2d...
        counts, _, _ = np.histogram2d(r, d, bins=(rbins, dbins), weights=w)
        counts2, _, _ = np.histogram2d(r, d, bins=(rbins, dbins))
        del r, d, w
        counts, counts2 = counts.T, counts2.T
        dbins2 = (dbins[:-1] + dbins[1:]) / 2
        rbins2 = (rbins[:-1] + rbins[1:]) / 2
        DGR_Median = DGR_LExp = DGR_Max = np.array([])
        n_zeromask = np.full(counts.shape[1], True, dtype=bool)
        for i in range(counts.shape[1]):
            if np.sum(counts2[:, i]) > 0:
                counts[:, i] /= np.sum(counts[:, i])
                counts2[:, i] /= np.sum(counts2[:, i])
                csp = np.cumsum(counts[:, i])[:-1]
                csp = np.append(0, csp / csp[-1])
                ssd = np.interp([0.16, 0.5, 0.84], csp, dbins2)
                DGR_Median = np.append(DGR_Median, ssd[1])
                DGR_LExp = np.append(DGR_LExp, 10**np.sum(np.log10(dbins2) *
                                                          counts[:, i]))
                DGR_Max = np.append(DGR_Max, dbins2[np.argmax(counts[:, i])])
            else:
                n_zeromask[i] = False
        #
        fig = plt.figure(figsize=(10, 7.5))
        plt.pcolormesh(rbins2, dbins2, counts, norm=LogNorm(), cmap=self.cmap1,
                       vmin=1E-3)
        plt.yscale('log')
        plt.colorbar()
        plt.plot(rbins2[n_zeromask], DGR_Median, 'r', label='Median')
        plt.plot(rbins2[n_zeromask], DGR_LExp, 'g', label='Log Expectation')
        plt.plot(rbins2[n_zeromask], DGR_Max, 'b', label='Max likelihhod')
        plt.ylim([1E-5, 1E-2])
        plt.xlabel(r'Radius ($R_{25}$)', size=16)
        plt.ylabel(r'DGR', size=16)
        plt.legend(fontsize=16)
        plt.title('Gas mass weighted median DGR', size=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        fn = 'output/_DGR-profile_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)
        """
        1X1: DGR vs. metallicity with other papers
        """
        print(' --Plotting DGR vs. Metallicity...')
        GD_dist = gal_data(name,
                           galdata_dir='data/gal_data').field('DIST_MPC')[0]
        df = pd.read_csv("data/Tables/Remy-Ruyer_2014.csv")
        # My DGR gradient with Remy-Ruyer data and various models
        fig, ax = plt.subplots(figsize=(10, 7.5))
        xbins2 = (8.715 - 0.027 * rbins2 * self.d[name]['R25'] * 7.4 / GD_dist)
        ax.pcolor(xbins2, dbins2, counts, norm=LogNorm(),
                  cmap='Reds', vmin=1E-3)
        ax.set_ylim([1E-5, 1E0])
        ax.set_yscale('log')
        ax.plot(xbins2[n_zeromask], DGR_LExp, 'g',
                label='This work (Log Expectation)')
        ax.set_xlabel('12 + log(O/H)', size=16)
        ax.set_ylabel('DGR', size=16)
        r_ = (8.715 - df['12+log(O/H)'].values) / 0.027 * GD_dist / 7.4 / \
            self.d[name]['R25']
        r__ = np.linspace(np.nanmin(r_), np.nanmax(r_), 50)
        x__ = (8.715 - 0.027 * r__ * self.d[name]['R25'] * 7.4 / GD_dist -
               solar_oxygen_bundance)
        o__ = x__ + solar_oxygen_bundance
        ax.plot(o__, 10**(1.62 * x__ - 2.21),
                'k--', alpha=0.6, label='R14 power law')
        ax.plot(o__, self.BPL_DGR(x__, 'MW'), 'k:',
                alpha=0.6, label='R14 broken power law')
        ax.plot(o__, 10**(x__) / 150, 'k', alpha=0.6,
                label='D14 power law')
        ax.plot(o__, self.max_possible_DGR(o__),
                'r', alpha=0.6, label='Max possible DGR')
        ax.scatter(df['12+log(O/H)'], df['DGR_MW'], c='b', s=15,
                   label='R14 data (MW)')
        zl = np.log10(1.81 * np.exp(-18 / 19))
        zu = np.log10(1.81 * np.exp(-8 / 19))
        z_ = np.linspace(zl, zu, 50)
        ax.plot(z_ + solar_oxygen_bundance, 10**z_ / 150, 'c',
                label='ZB12 range', linewidth=3.0)
        ax.legend(fontsize=16)
        ax.set_xticklabels(ax.get_xticks(), fontsize=12)
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)
        fn = 'output/_DGR-vs-Metallicity_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)

    def max_possible_DGR(self, O_p_H):
        return 10**(O_p_H - 12) * 15.9994 / 1.0079 / 0.4279 / 1.36

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
            tbins = \
                np.logspace(np.min(np.log10(t)), np.max(np.log10(t)),
                            self.tbin)
            # Counting hist2d...
            counts, _, _ = np.histogram2d(r, t, bins=(rbins, tbins), weights=w)
            counts2, _, _ = np.histogram2d(r, t, bins=(rbins, tbins))
            del r, t, w
            counts, counts2 = counts.T, counts2.T
            tbins2 = (tbins[:-1] + tbins[1:]) / 2
            rbins2 = (rbins[:-1] + rbins[1:]) / 2
            T_Median = T_LExp = T_Max = np.array([])
            n_zeromask = np.full(counts.shape[1], True, dtype=bool)
            for i in range(counts.shape[1]):
                if np.sum(counts2[:, i]) > 0:
                    counts[:, i] /= np.sum(counts[:, i])
                    counts2[:, i] /= np.sum(counts2[:, i])
                    csp = np.cumsum(counts[:, i])[:-1]
                    csp = np.append(0, csp / csp[-1])
                    sst = np.interp([0.16, 0.5, 0.84], csp, tbins2)
                    T_Median = np.append(T_Median, sst[1])
                    T_LExp = np.append(T_LExp, 10**np.sum(np.log10(tbins2) *
                                                          counts[:, i]))
                    T_Max = np.append(T_Max, tbins2[np.argmax(counts[:, i])])
                else:
                    n_zeromask[i] = False
            #
            fig = plt.figure(figsize=(10, 7.5))
            plt.pcolormesh(rbins2, tbins2, counts, norm=LogNorm(),
                           cmap=self.cmap1, vmin=1E-3)
            plt.colorbar()
            plt.plot(rbins2[n_zeromask], T_Median, 'r', label='Median')
            plt.plot(rbins2[n_zeromask], T_LExp, 'g', label='Log Expectation')
            plt.plot(rbins2[n_zeromask], T_Max, 'b', label='Max likelihhod')
        plt.ylim([5, 40])
        plt.xlabel(r'Radius ($R_{25}$)', size=16)
        plt.ylabel(r'T (K)', size=16)
        plt.legend(fontsize=16)
        plt.title(r'Gas mass weighted $T_d$', size=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
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
            ax[i, 0].set_ylabel('(Obs. - Model) / Obs.')
            ax[i, 0].set_title(titles[i])
            ax[i, 1].hist(Res_d_data, orientation='horizontal',
                          range=yranges, bins=50)
            ax[i, 1].set_xlabel('Count')
            ax[i, 1].set_ylim(yranges)
            ax[i, 1].set_title(titles[i])
        fig.tight_layout()
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
            if method_abbr == 'BEMFBFL':
                models = hf['BEMFB'].value
                try:
                    self.lambda_cs
                except AttributeError:
                    self.Load_Data(name, 'BEMFB')
                lambda_cf = 300.0
                models = models[:, :, self.lambda_cs == lambda_cf, :]
            else:
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
        SigmaD = 10**self.d[name][method_abbr]['alogSigmaD'][i]
        T = self.d[name][method_abbr]['aT'][i]
        Beta = self.d[name][method_abbr]['aBeta'][i]
        #
        # Colour correction factors
        #
        if method_abbr in ['EF', 'FB', 'FBPT', 'PB']:
            model_complete = SEMBB(wl_complete, SigmaD, T, Beta)
            ccf = SEMBB(wl, SigmaD, T, Beta) / \
                z0mg_RSRF(wl_complete, model_complete, bands)
            sed_best_plot = SEMBB(wl_plot, SigmaD, T, Beta)
        elif method_abbr in ['BEMFB', 'BEMFBFL']:
            lambda_c = self.d[name][method_abbr]['alambda_c'][i]
            beta2 = self.d[name][method_abbr]['abeta2'][i]
            model_complete = \
                BEMBB(wl_complete, SigmaD, T, Beta, lambda_c, beta2)
            ccf = BEMBB(wl, SigmaD, T, Beta, lambda_c, beta2) / \
                z0mg_RSRF(wl_complete, model_complete, bands)
            sed_best_plot = BEMBB(wl_plot, SigmaD, T, Beta, lambda_c, beta2)
        elif method_abbr in ['FBWD']:
            WDfrac = self.d[name][method_abbr]['aWDfrac'][i]
            model_complete = \
                WD(wl_complete, SigmaD, T, Beta, WDfrac)
            ccf = WD(wl, SigmaD, T, Beta, WDfrac) / \
                z0mg_RSRF(wl_complete, model_complete, bands)
            sed_best_plot = WD(wl_plot, SigmaD, T, Beta, WDfrac)
        sed_obs_plot = sed * ccf
        unc_obs_plot = unc * ccf
        #
        # Begin fitting
        #
        if method_abbr == 'EF':
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
        elif method_abbr in ['BEMFB', 'BEMFBFL']:
            Sigmas, Ts, lambda_cs, beta2s = \
                np.meshgrid(self.SigmaDs, self.Ts, self.lambda_cs,
                            self.beta2s)
            if method_abbr == 'BEMFBFL':
                Sigmas = Sigmas[:, :, self.lambda_cs == lambda_cf, :]
                Ts = Ts[:, :, self.lambda_cs == lambda_cf, :]
                beta2s = beta2s[:, :, self.lambda_cs == lambda_cf, :]
                lambda_cs = lambda_cs[:, :, self.lambda_cs == lambda_cf, :]
            Betas = np.full(Ts.shape, Beta)
        elif method_abbr == 'FBWD':
            Sigmas, Ts, WDfracs = \
                np.meshgrid(self.SigmaDs, self.Ts, self.WDfracs)
            Betas = np.full(Ts.shape, Beta)
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
        chi2, Sigmas, Ts, Betas = \
            chi2[mask], Sigmas[mask], Ts[mask], Betas[mask]
        if method_abbr in ['BEMFB', 'BEMFBFL']:
            lambda_cs, beta2s = lambda_cs[mask], beta2s[mask]
        elif method_abbr in ['FBWD']:
            WDfracs = WDfracs[mask]
        num = 100
        if len(Ts) > num:
            mask = np.array([True] * num + [False] * (len(Ts) - num))
            np.random.shuffle(mask)
            chi2, Sigmas, Ts, Betas = \
                chi2[mask], Sigmas[mask], Ts[mask], Betas[mask]
            if method_abbr in ['BEMFB', 'BEMFBFL']:
                lambda_cs, beta2s = lambda_cs[mask], beta2s[mask]
            elif method_abbr in ['FBWD']:
                WDfracs = WDfracs[mask]
        alpha = np.exp(-0.5 * chi2) * 0.2
        #
        # Begin plotting
        #
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.set_ylim([0.0, np.nanmax(sed_best_plot) * 1.2])
        for j in range(len(Ts)):
            if method_abbr in ['EF', 'FB', 'FBPT', 'PB']:
                model_plot = SEMBB(wl_plot, Sigmas[j], Ts[j], Betas[j])
            elif method_abbr in ['BEMFB', 'BEMFBFL']:
                model_plot = BEMBB(wl_plot, Sigmas[j], Ts[j], Betas[j],
                                   lambda_cs[j], beta2s[j])
            elif method_abbr in ['FBWD']:
                model_plot = WD(wl_plot, Sigmas[j], Ts[j], Betas[j],
                                WDfracs[j])
            ax.plot(wl_plot, model_plot, alpha=alpha[j], color='k')
        ax.plot(wl_plot, sed_best_plot, linewidth=3,
                label=method_abbr + ' best fit')
        ax.errorbar(wl, sed_obs_plot, yerr=unc_obs_plot, fmt='o',
                    color='red', capsize=10, label='Herschel data')
        ax.legend(fontsize=12)
        ax.set_xlabel(r'Wavelength ($\mu m$)', size=12)
        ax.set_ylabel(r'SED ($MJy$ $sr^{-1}$)', size=12)
        fig.tight_layout()
        fn = 'output/_Model_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)
        ax.set_ylim([1.0, np.nanmax(sed_best_plot) * 1.2])
        ax.set_yscale('log')
        fn = 'output/_Model-log_' + name + '_' + method_abbr + '.pdf'
        with PdfPages(fn) as pp:
            pp.savefig(fig)

    def corner_plots(self, name, method_abbr):
        plt.close('all')
        print(' --Plotting corner plot...')
        if method_abbr == 'EF':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'],
                                self.d[name][method_abbr]['aT'],
                                self.d[name][method_abbr]['aBeta']])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$\beta$']
        elif method_abbr == 'FB':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'],
                                self.d[name][method_abbr]['aT']])
            labels = [r'$\log(\Sigma_d)$', r'$T$']
        elif method_abbr == 'FBPT':
            samples = self.d[name][method_abbr]['alogSigmaD'].reshape(1, -1)
            labels = [r'$\log(\Sigma_d)$']
        elif method_abbr == 'PB':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'],
                                self.d[name][method_abbr]['aT']])
            labels = [r'$\log(\Sigma_d)$', r'$T$']
        elif method_abbr == 'BEMFB':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'],
                                self.d[name][method_abbr]['aT'],
                                self.d[name][method_abbr]['alambda_c'],
                                self.d[name][method_abbr]['abeta2']])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$\lambda_c$',
                      r'$\beta_2$']
        elif method_abbr == 'BEMFBFL':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'],
                                self.d[name][method_abbr]['aT'],
                                self.d[name][method_abbr]['abeta2']])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$\beta_2$']
        elif method_abbr == 'FBWD':
            samples = np.array([self.d[name][method_abbr]['alogSigmaD'],
                                self.d[name][method_abbr]['aT'],
                                self.d[name][method_abbr]['aWDfrac']])
            labels = [r'$\log(\Sigma_d)$', r'$T$', r'$WD$ $frac$']
        fig = corner(samples.T, labels=labels, quantities=(0.16, 0.84),
                     show_titles=True, title_kwargs={"fontsize": 12})
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


"""
def plots_for_paper(name='NGC5457', rbin=51, dbin=100, tbin=90, SigmaDoff=2.,
                    Toff=20, dr25=0.025, method='011111',
                    cmap0='gist_heat', cmap1='Greys', cmap2='seismic',
                    cmap3='Reds', fitting_model='SEMBB'):
    #
    # 3X1: Temperature gradient & Sigma_SFR + Sigma_* profile
    #
    print('3X1: Temperature gradient & Sigma_SFR + Sigma_* profile')
    r, t, w = np.array([]), np.array([]), np.array([])
    for i in range(lbl):
        temp_GM = aGas[i] * (binmap == binlist[i]).sum()
        mask = a_T_PDFs_AF[i] > a_T_PDFs_AF[i].max() / 1000
        temp_T, temp_P = Ts_AF[mask], a_T_PDFs_AF[i][mask]
        temp_P = temp_P / np.sum(temp_P) * temp_GM
        r = np.append(r, [aRadius[i]] * len(temp_T))
        for j in range(len(temp_T)):
            t = np.append(t, temp_T[j])
            w = np.append(w, temp_P[j])
    nanmask = np.isnan(r + t + w)
    r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
    rbins = np.linspace(np.min(r), np.max(r), rbin)
    tbins = np.linspace(np.min(t), np.max(t), tbin)
    # Counting hist2d
    counts, _, _ = np.histogram2d(r, t, bins=(rbins, tbins), weights=w)
    del r, t, w
    # Fixing temperature
    r, t, w = np.array([]), np.array([]), np.array([])
    for i in range(lbl):
        temp_GM = aGas[i] * (binmap == binlist[i]).sum()
        mask = a_T_PDFs_FB[i] > a_T_PDFs_FB[i].max() / 1000
        temp_T, temp_P = Ts_FB[mask], a_T_PDFs_FB[i][mask]
        temp_P = temp_P / np.sum(temp_P) * temp_GM
        r = np.append(r, [aRadius[i]] * len(temp_T))
        for j in range(len(temp_T)):
            t = np.append(t, temp_T[j])
            w = np.append(w, temp_P[j])
    nanmask = np.isnan(r + t + w)
    r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
    tbins_FB = np.linspace(np.min(t), np.max(t), tbin)
    # Counting hist2d
    counts_FB, _, _ = np.histogram2d(r, t, bins=(rbins, tbins_FB), weights=w)
    del r, t, w
    counts, counts_FB = counts.T, counts_FB.T
    tbins2 = (tbins[:-1] + tbins[1:]) / 2
    tbins2_FB = (tbins[:-1] + tbins[1:]) / 2
    rbins2 = (rbins[:-1] + rbins[1:]) / 2
    T_Exp, T_Exp_FB = np.array([]), np.array([])
    n_zeromask = np.full(counts.shape[1], True, dtype=bool)
    n_zeromask_FB = np.full(counts_FB.shape[1], True, dtype=bool)
    # Calculating PDFs at each radial bin...
    assert counts.shape[1] == counts_FB.shape[1]
    for i in range(counts.shape[1]):
        if np.sum(counts[:, i]) > 0:
            counts[:, i] /= np.sum(counts[:, i])
            T_Exp = np.append(T_Exp, np.sum(counts[:, i] * tbins2))
        else:
            n_zeromask[i] = False
    for i in range(counts_FB.shape[1]):
        if np.sum(counts_FB[:, i]) > 0:
            counts_FB[:, i] /= np.sum(counts_FB[:, i])
            T_Exp_FB = np.append(T_Exp_FB, np.sum(counts_FB[:, i] * tbins2_FB))
        else:
            n_zeromask_FB[i] = False
    R_SFR, SFR_profile = simple_profile(SFR, Radius, 100, SigmaGas)
    R_SMSD, SMSD_profile = simple_profile(SMSD, Radius, 100, SigmaGas)
    #
    rows, columns = 3, 1
    fig, ax = plt.subplots(rows, columns, figsize=(10, 16))
    titles = ['Temperature radial profile',
              r'Temperature radial profile ($\beta=2$)',
              r'$\Sigma_{SFR}$ and $\Sigma_*$ radial profile']
    maxs = [np.nanmax(np.append(rbins2, np.append(R_SFR, R_SMSD)))]
    mins = [np.nanmin(np.append(rbins2, np.append(R_SFR, R_SMSD)))]
    temp1 = [tbins2, tbins2_FB]
    temp2 = [counts, counts_FB]
    temp3 = [T_Exp, T_Exp_FB]
    temp4 = [n_zeromask, n_zeromask_FB]
    del tbins2, tbins2_FB, counts, counts_FB, T_Exp, T_Exp_FB
    for i in range(2):
        ax[i].pcolormesh(rbins2, temp1[i], temp2[i], norm=LogNorm(),
                         cmap=cmap3, vmin=1E-3)
        ax[i].plot(rbins2[temp4[i]], temp3[i], 'b', label='Expectation')
        ax[i].set_title(titles[i])
        ax[i].set_xlabel(r'Radius ($R_{25}$)', size=16)
        ax[i].set_ylabel(r'Temperature ($K$)', size=16)
        ax[i].set_xlim([mins[0], maxs[0]])
        ax[i].set_xticklabels(ax[i].get_xticks(), fontsize=12)
        ax[i].set_yticklabels(ax[i].get_yticks(), fontsize=12)
        ax[i].legend(fontsize=14)
    del temp1, temp2, temp3, temp4
    # IMWH
    ax[2].semilogy(R_SFR, SFR_profile, 'k')
    ax[2].set_xlabel(r'Radius ($R25$)', size=16)
    ax[2].set_ylabel(r'$\Sigma_{SFR}$ ($M_\odot kpc^{-2} yr^{-1}$)',
                     size=16, color='k')
    ax[2].tick_params('y', colors='k')
    ax[2].set_xticklabels(ax[2].get_xticks(), fontsize=12)
    ax[2].set_yticklabels(ax[2].get_yticks(), fontsize=12)
    ax2 = ax[2].twinx()
    ax2.semilogy(R_SMSD, SMSD_profile, c='b')
    ax2.set_ylabel(r'$\Sigma_*$ ($M_\odot pc^{-2}$)', size=16, color='b')
    ax2.tick_params('y', colors='b')
    ax2.set_xlim([0, rbins2[n_zeromask].max()])
    ax2.set_yticklabels(ax2.get_xticks(), fontsize=12)
    ax2.set_title(titles[2])
    fig.tight_layout()
    with PdfPages('output/_3X1TempRP.pdf') as pp:
        pp.savefig(fig)
    del R_SFR, SFR_profile, R_SMSD, SMSD_profile, fig, ax, ax2
    #
    # 1X2: Fitted vs. predicted temperature: residual map and radial profile
    #
    print('1X2: Fitted vs. predicted temperature: residual map and radial',
          'profile')
    titles = ['residual map',
              'Radial profile']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs)
    cax = ax.imshow(T_FBT - T_FB, origin='lower', cmap=cmap0)
    ax.set_title(titles[0], size=20)
    ax.set_xlabel('RA', size=16)
    ax.set_xticklabels(ax.get_xticks(), fontsize=16)
    ax.set_ylabel('Dec', size=16)
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)
    ax.coords[0].set_major_formatter('hh:mm')
    ax.coords[1].set_major_formatter('dd:mm')
    fig.colorbar(cax, ax=ax)
    with PdfPages('output/_1X1Temp_FB.FBT.res.pdf') as pp:
        pp.savefig(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    R_Tfb, Tfb_profile = simple_profile(T_FB, Radius, 100, SigmaGas)
    R_TFBT, TFBT_profile = simple_profile(T_FBT, Radius, 100, SigmaGas)
    ax.plot(R_Tfb, Tfb_profile, label='FB')
    ax.plot(R_TFBT, TFBT_profile, label='FBT')
    ax.set_xlabel(r'Radius ($R25$)', size=16)
    ax.set_ylabel(r'Temperature ($K$)', size=16)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    ax.legend(fontsize=14)
    fig.tight_layout()
    with PdfPages('output/_1X1Temp_FB.FBT.pdf') as pp:
        pp.savefig(fig)
    del titles, cax, fig, ax, rows, columns, R_Tfb, Tfb_profile, R_TFBT, \
        TFBT_profile
    #
"""
