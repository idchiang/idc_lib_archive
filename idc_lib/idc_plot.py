import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import multiprocessing as mp
# import seaborn as sns
# from matplotlib.colors import LogNorm
# from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib import ticker
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from corner import corner
from astropy.io import fits
# from astropy.wcs import WCS
# from .gal_data import gal_data
from .idc_functions import SEMBB, BEMBB, WD, PowerLaw
from .idc_fitting import parameters, band_wl, band_cap, v_prop
from .z0mg_RSRF import z0mg_RSRF

plt.ioff()

# Some definitions
solar_oxygen_bundance = 8.69  # (O/H)_\odot, ZB12

# Should change these to more meaningful names
cmap_map = 'inferno'
cmap_grey = 'Greys'
cmap_div = 'coolwarm'

# Default binning size
rbin = 21
dbin = 1000
tbin = 90


class Dust_Plots(object):
    def __init__(self, TRUE_FOR_PNG=True):
        self.dict = {}
        self.galaxy_data = {}
        self.kappa160 = {}
        self.TRUE_FOR_PNG = TRUE_FOR_PNG  # False for PDF

    def New_Dataset(self, nickname, object_='ngc5457', model='BE', beta_f=2.0,
                    lambdac_f=300, project_name='', filepath='', datapath='',
                    bands=[]):
        # Pre-prosessing
        beta_f = round(beta_f, 2)
        if type(bands) != np.ndarray:
            bands = np.array(bands)
        wl = np.array([band_wl[b] for b in bands])
        bands_cap = np.array([band_cap[b] for b in bands])
        # Load kappa160 value
        if model not in self.kappa160.keys():
            self.kappa160[model] = {}
        if beta_f not in self.kappa160[model].keys():
            fn = 'hdf5_MBBDust/Calibration_' + str(beta_f) + '.h5'
            try:
                with h5py.File(fn, 'r') as hf:
                    grp = hf[model]
                    self.kappa160[model][beta_f] = grp['kappa160'].value
            except KeyError:
                print("Calibration file for beta=" + str(beta_f) +
                      " doesn't exist!!")
        # Load galaxy properties
        if object_ not in self.galaxy_data.keys():
            self.galaxy_data[object_] = {}
        # Directory to save plots
        plotspath = filepath + '/plots'
        if not os.path.isdir(plotspath):
            os.mkdir(plotspath)
        # fitsfn
        filelist = os.listdir(filepath)
        for fn in filelist:
            temp = fn.split('_')
            if len(temp) > 1:
                assert temp[0] == object_
                if temp[1] == 'dust.surface.density':
                    fitsfnend = '_' + '_'.join(temp[2:])
                    break
        # Save file path and so on with a nick name
        self.dict[nickname] = {'object_': object_,
                               'project_name': project_name,
                               'model': model,
                               'beta_f': beta_f,
                               'lambdac_f': lambdac_f,
                               'kappa160': self.kappa160[model][beta_f],
                               'datapath': datapath,
                               'fnhead': filepath + '/' + object_ + '_',
                               'fnend': fitsfnend,
                               'savefnhead': plotspath + '/' + object_ + '_',
                               'savefnend': fitsfnend.strip('.fits.gz'),
                               'wl': wl,
                               'bands': bands,
                               'bands_cap': bands_cap}

    def corner_plots(self, nickname, paras=[], plot_chi2=False,
                     displaylog=True):
        print('Plotting corner plot for', nickname)
        print(' -- Loading data.')
        if paras == []:
            paras = parameters[self.dict[nickname]['model']]
        if plot_chi2 and ('chi2' not in paras):
            paras += ['chi2']
        samples, labels = [], []
        fnhead = self.dict[nickname]['fnhead']
        fnend = self.dict[nickname]['fnend']
        for para in paras:
            fn = fnhead + para + fnend
            data = fits.getdata(fn, header=False)
            if data.ndim == 3:
                data = data[0]
            if displaylog and v_prop[para][0]:
                data = np.log10(data)
                labels.append(v_prop[para][2][1])
            else:
                labels.append(v_prop[para][2][0])
            assert data.ndim == 2
            samples.append(data.flatten())
        mask = np.all(np.isfinite(samples), axis=0)
        samples = np.array(samples).T[mask]
        print(' -- Data loaded. Start plotting.')
        #
        fig = corner(samples, labels=labels, quantities=(0.16, 0.84),
                     show_titles=True, title_kwargs={"fontsize": 16},
                     label_kwargs={"fontsize": 16})
        #
        savefnhead = self.dict[nickname]['savefnhead']
        savefnend = self.dict[nickname]['savefnend']
        fn = savefnhead + 'corner.plot' + savefnend
        if self.TRUE_FOR_PNG:
            fig.savefig(fn + '.png', bbox_inches='tight')
        else:
            with PdfPages(fn + '.pdf') as pp:
                pp.savefig(fig, bbox_inches='tight')
        print(' -- Plots saved. All figures closing.')
        plt.close('all')

    def example_model(self, nickname, num=1, displaylog=True,
                      bright_onethird=True):
        print('Plotting', num, 'example models for', nickname)
        print(' -- Loading raw data.')
        sed, diskmask = self.sed_and_diskmask(nickname)
        fnhead = self.dict[nickname]['fnhead']
        fnend = self.dict[nickname]['fnend']
        beta_f = round(self.dict[nickname]['beta_f'], 1)
        model = self.dict[nickname]['model']
        assert model != 'PL'
        kappa160 = self.dict[nickname]['kappa160']
        wl = self.dict[nickname]['wl']
        nwl = len(wl)
        bands = self.dict[nickname]['bands']
        fitted_paras = []
        temp_paras = parameters[self.dict[nickname]['model']]
        if 'chi2' not in temp_paras:
            temp_paras.append('chi2')
        for para in temp_paras:
            data = fits.getdata(fnhead + para + fnend, header=False)
            if data.ndim == 3:
                data = data[0]
            if displaylog and v_prop[para][0]:
                data = np.log10(data)
            assert data.ndim == 2
            fitted_paras.append(data)
        fn = fnhead + temp_paras[0] + '.pdf' + fnend
        logSigmad_PDF = fits.getdata(fn)
        logSigmads_1d = v_prop['dust.surface.density'][1]
        fn = fnhead + temp_paras[1] + '.pdf' + fnend
        Td_PDF = fits.getdata(fn)
        Tds_1d = v_prop['dust.temperature'][1]
        #
        print(' -- Building random pool.')
        spire250 = sed[bands == 'spire250'].flatten()
        mask = diskmask.flatten().astype(bool) * np.isfinite(spire250)
        if bright_onethird:
            spire250_ms = np.sort(spire250[mask])
            onethird = np.interp(2 / 3, np.arange(len(spire250_ms)) /
                                 (len(spire250_ms) - 1),
                                 spire250_ms)
            mask[mask] = spire250_ms > onethird
        pool = np.arange(diskmask.size)[mask]
        try:
            ks = np.random.choice(pool, num, replace=False)
        except ValueError:
            print('Total sample size:', len(pool))
            temp_num = 10 * int(len(pool) // 10)
            print('Change sample size to:', temp_num)
            ks = np.random.choice(pool, temp_num, replace=False)
        #
        print(' -- Importing models.')
        bkgcov = fits.getdata(fnhead + 'bkgcov' + fnend, header=False)
        cali_mat2 = fits.getdata(fnhead + 'cali.mat2' + fnend, header=False)

        models = []
        for b in bands:
            fn = 'models/' + b + '_' + model + '.beta=' + \
                str(round(beta_f, 1)) + '.fits.gz'
            models.append(fits.getdata(fn))
        models = np.array(models)
        models = np.moveaxis(models, 0, -1)
        #
        betas = beta_f
        lambdacs = self.dict[nickname]['lambdac_f']
        Tds, beta2s, wdfracs, alphas, loggammas, logUmins = 0, 0, 0, 0, 0, 0
        if model == 'PL':
            logSigmads, alphas, loggammas, logUmins = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['alpha'][1],
                            v_prop['gamma'][1],
                            v_prop['logUmin'][1])
        if model == 'SE':
            logSigmads, Tds, betas = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['dust.temperature'][1],
                            v_prop['beta'][1])
        elif model == 'FB':
            logSigmads, Tds = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['dust.temperature'][1])
        elif model == 'BE':
            logSigmads, Tds, beta2s = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['dust.temperature'][1],
                            v_prop['beta2'][1])
        elif model == 'WD':
            logSigmads, Tds, wdfracs = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['dust.temperature'][1],
                            v_prop['warm.dust.fraction'][1])
        #
        print(' -- Building plotting models.')
        wl_plot = np.linspace(51, 549, 50)
        models_plot = np.empty(list(models.shape)[:-1] + [len(wl_plot)])
        for w in range(len(wl_plot)):
            if model in ['SE', 'FB']:
                models_plot[..., w] = \
                    SEMBB(wl_plot[w], 10**logSigmads, Tds, betas,
                          kappa160=kappa160)
            elif model in ['BEMFB', 'BE']:
                models_plot[..., w] = \
                    BEMBB(wl_plot[w], 10**logSigmads, Tds, betas, lambdacs,
                          beta2s, kappa160=kappa160)
            elif model in ['WD']:
                models_plot[..., w] = \
                    WD(wl_plot[w], 10**logSigmads, Tds, betas, wdfracs,
                       kappa160=kappa160)
            elif model in ['PL']:
                models_plot[..., w] = \
                    PowerLaw(wl_plot[w], 10**logSigmads, alphas, 10**loggammas,
                             logUmins, kappa160=kappa160)
        #
        print(' -- Begin plotting.')
        savefnhead = self.dict[nickname]['savefnhead']
        savefnend = self.dict[nickname]['savefnend']
        plotname = 'example.model.rand='
        os.system('find -name "' + savefnhead + plotname + '*" -delete')
        for k in ks:
            i, j = np.unravel_index(k, diskmask.shape)
            #
            # chi^2
            #
            sed_vec = sed[:, i, j].reshape(1, nwl)
            calcov = sed_vec.T * cali_mat2 * sed_vec
            cov_n1 = np.linalg.inv(bkgcov + calcov)
            unc = np.sqrt(np.linalg.inv(cov_n1).diagonal())
            diff = models - sed[:, i, j]
            shape0 = list(diff.shape)[:-1]
            shape1 = shape0 + [1, nwl]
            shape2 = shape0 + [nwl, 1]
            chi2 = np.matmul(np.matmul(diff.reshape(shape1), cov_n1),
                             diff.reshape(shape2)).reshape(shape0)
            #
            # Selecting samples for plotting
            #
            chi2_threshold = 5.0
            chi2 -= np.nanmin(chi2)
            transp = np.exp(-0.5 * chi2) * 0.2
            chi2_mask = chi2 < chi2_threshold
            plot_pool = np.arange(chi2_mask.size)[chi2_mask.flatten()]
            try:
                ps = np.random.choice(plot_pool, 40, replace=False)
            except ValueError:
                print('Total sample size:', len(plot_pool))
                temp_num = 10 * int(len(plot_pool) // 10)
                print('Change sample size to:', temp_num)
                ps = np.random.choice(plot_pool, temp_num, replace=False)
            #
            # MAX parameters:
            #
            am = np.unravel_index(np.argmin(chi2), chi2.shape)
            #
            # Best fit model
            if model in ['SE']:
                EXP_sed = \
                    SEMBB(wl_plot, 10**fitted_paras[0][i, j],
                          fitted_paras[1][i, j], fitted_paras[2][i, j],
                          kappa160=kappa160)
                MAX_paras = [logSigmads[am], Tds[am], betas[am]]
                MAX_sed = \
                    SEMBB(wl_plot, 10**MAX_paras[0],
                          MAX_paras[1], MAX_paras[2],
                          kappa160=kappa160)
            elif model in ['FB']:
                EXP_sed = \
                    SEMBB(wl_plot, 10**fitted_paras[0][i, j],
                          fitted_paras[1][i, j], betas,
                          kappa160=kappa160)
                MAX_paras = [logSigmads[am], Tds[am]]
                MAX_sed = \
                    SEMBB(wl_plot, 10**MAX_paras[0],
                          MAX_paras[1], betas,
                          kappa160=kappa160)
            elif model in ['BEMFB', 'BE']:
                EXP_sed = \
                    BEMBB(wl_plot, 10**fitted_paras[0][i, j],
                          fitted_paras[1][i, j], betas, lambdacs,
                          fitted_paras[2][i, j], kappa160=kappa160)
                MAX_paras = [logSigmads[am], Tds[am], beta2s[am]]
                MAX_sed = \
                    BEMBB(wl_plot, 10**MAX_paras[0],
                          MAX_paras[1], betas, lambdacs,
                          MAX_paras[2], kappa160=kappa160)
            elif model in ['WD']:
                EXP_sed = \
                    WD(wl_plot, 10**fitted_paras[0][i, j],
                       fitted_paras[1][i, j], betas, fitted_paras[2][i, j],
                       kappa160=kappa160)
                MAX_paras = [logSigmads[am], Tds[am], wdfracs[am]]
                MAX_sed = \
                    WD(wl_plot, 10**MAX_paras[0],
                       MAX_paras[1], betas, MAX_paras[2],
                       kappa160=kappa160)
            elif model in ['PL']:
                EXP_sed = \
                    PowerLaw(wl_plot, 10**fitted_paras[0][i, j],
                             fitted_paras[1][i, j], 10**fitted_paras[2][i, j],
                             fitted_paras[3][i, j], kappa160=kappa160)
                MAX_paras = [logSigmads[am], alphas[am], loggammas[am],
                             logUmins[am]]
                MAX_sed = \
                    PowerLaw(wl_plot, 10**MAX_paras[0],
                             MAX_paras[1], 10**MAX_paras[2],
                             MAX_paras[3], kappa160=kappa160)
            #
            # Plotting
            #
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            # Top-left: Temperature map
            cax = ax[0, 0].imshow(fitted_paras[1], origin='lower',
                                  cmap=cmap_map)
            ax[0, 0].scatter(j, i, s=100, c='c', marker='*')
            ax[0, 0].set_xticklabels([])
            ax[0, 0].xaxis.set_ticks_position('none')
            ax[0, 0].set_yticklabels([])
            ax[0, 0].yaxis.set_ticks_position('none')
            plt.colorbar(cax, ax=ax[0, 0])
            ax[0, 0].set_title(r'$T_d$ [$K$]', x=0.05, y=0.9, ha='left')
            # Top-right: Example model
            # Plot selected models
            for p in ps:
                coord = np.unravel_index(p, chi2.shape)
                ax[0, 1].plot(wl_plot, models_plot[coord], alpha=transp[coord],
                              color='k')
            # Plot EXP and MAX
            ax[0, 1].plot(wl_plot, EXP_sed, linewidth=3, label='EXP',
                          color='orange')
            ax[0, 1].plot(wl_plot, MAX_sed, linewidth=3, label='MAX',
                          color='g')
            # Plot observation
            ax[0, 1].errorbar(wl, sed[:, i, j], yerr=unc, fmt='o', color='red',
                              capsize=10, label='Herschel data')
            ax[0, 1].legend()
            ax[0, 1].set_xlabel(r'Wavelength [$\mu m$]')
            ax[0, 1].set_ylabel(r'SED [$MJy$ $sr^{-1}$]')
            ax[0, 1].set_title('Models and Observation')
            # Bottom-left: Surface density PDF
            ax[1, 0].plot(logSigmads_1d, logSigmad_PDF[:, i, j])
            ax[1, 0].grid()
            ax[1, 0].plot([fitted_paras[0][i, j]] * 2,
                          [0, np.max(logSigmad_PDF[:, i, j])], label='EXP',
                          color='orange')
            ax[1, 0].plot([MAX_paras[0]] * 2,
                          [0, np.max(logSigmad_PDF[:, i, j])], label='MAX',
                          color='g')
            ax[1, 0].set_yticklabels([])
            ax[1, 0].set_xlabel(v_prop['dust.surface.density'][2][1])
            ax[1, 0].yaxis.set_ticks_position('none')
            ax[1, 0].set_title(r'$log(\Sigma_d)$ PDF', x=0.05, y=0.9,
                               ha='left')
            # Bottom-right: Temperature PDF
            ax[1, 1].plot(Tds_1d, Td_PDF[:, i, j])
            ax[1, 1].grid()
            ax[1, 1].plot([fitted_paras[1][i, j]] * 2,
                          [0, np.max(Td_PDF[:, i, j])], label='EXP',
                          color='orange')
            ax[1, 1].plot([MAX_paras[1]] * 2,
                          [0, np.max(Td_PDF[:, i, j])], label='MAX',
                          color='g')
            ax[1, 1].set_xlabel(v_prop['dust.temperature'][2][0])
            ax[1, 1].set_yticklabels([])
            ax[1, 1].yaxis.set_ticks_position('none')
            ax[1, 1].set_title(r'$T_d$ PDF', x=0.05, y=0.9, ha='left')
            # Set title
            title = ''
            for p in range(len(temp_paras)):
                para = temp_paras[p]
                if displaylog and v_prop[para][0]:
                    title += v_prop[para][2][1]
                else:
                    title += v_prop[para][2][0]
                title += '=' + str(round(fitted_paras[p][i, j], 2)) + '; '
            fig.suptitle(title)
            fn = savefnhead + plotname + '[' + "{:03}".format(i) + ',' + \
                "{:03}".format(j) + ']' + savefnend
            if self.TRUE_FOR_PNG:
                fig.savefig(fn + '.png', bbox_inches='tight')
            else:
                with PdfPages(fn + '.pdf') as pp:
                    pp.savefig(fig, bbox_inches='tight')
            plt.close('all')
        print(' -- Plots saved. All figures closing.')

    def color_vs_temperature(self, nicknames=[]):
        print('Plotting color vs. temperature for', nicknames)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        for nickname in nicknames:
            print(' -- Loading raw data for', nickname)
            sed, diskmask = self.sed_and_diskmask(nickname)
            non_nanmask = np.all(np.isfinite(sed), axis=0)
            diskmask = diskmask.astype(bool)
            bkgmask = (~diskmask) * non_nanmask
            diskmask = diskmask * non_nanmask
            bkgcov = np.cov(sed[:, bkgmask])
            unc = np.sqrt(bkgcov.diagonal())
            del bkgmask, diskmask
            Tstr = 'dust.temperature'
            bands = self.dict[nickname]['bands']
            fnhead = self.dict[nickname]['fnhead']
            fnend = self.dict[nickname]['fnend']
            savefnhead = self.dict[nickname]['savefnhead']
            savefnend = self.dict[nickname]['savefnend']
            Td = fits.getdata(fnhead + Tstr + fnend, header=False)
            Td = Td[0]
            with np.errstate(invalid='ignore'):
                p160 = bands == 'pacs160'
                pacs100_pacs160 = sed[bands == 'pacs100'][0] / sed[p160][0]
                pacs160_spire350 = sed[p160][0] / sed[bands == 'spire350'][0]
                three_sigma_mask = sed[p160][0] > 3 * unc[p160][0]
            mask = np.isfinite(Td + pacs100_pacs160 + pacs160_spire350) * \
                three_sigma_mask
            Td, pacs100_pacs160, pacs160_spire350 = \
                Td[mask], pacs100_pacs160[mask], pacs160_spire350[mask]
            print(' -- Start plotting.')
            ax[0].scatter(Td, pacs100_pacs160, s=1, label=nickname, alpha=0.7)
            ax[1].scatter(Td, pacs160_spire350, s=1, label=nickname, alpha=0.7)
        ax[0].set_ylabel('PACS100 / PACS160')
        ax[1].set_ylabel('PACS160 / SPIRE350')
        for i in range(2):
            ax[i].grid()
            ax[i].legend()
            ax[i].set_xlabel(v_prop[Tstr][2][0])
        fn = savefnhead + 'color.vs.temperature' + savefnend
        if self.TRUE_FOR_PNG:
            fig.savefig(fn + '.png', bbox_inches='tight')
        else:
            with PdfPages(fn + '.pdf') as pp:
                pp.savefig(fig, bbox_inches='tight')
        print(' -- Plots saved. All figures closing.')
        plt.close('all')

    def diagnostic_interactive(self, nickname, displaylog=True):
        print('Generating interactive diagnostic plot for', nickname)
        print(' -- Loading raw data.')
        sed, diskmask = self.sed_and_diskmask(nickname)
        fnhead = self.dict[nickname]['fnhead']
        fnend = self.dict[nickname]['fnend']
        beta_f = round(self.dict[nickname]['beta_f'], 1)
        model = self.dict[nickname]['model']
        assert model != 'PL'
        kappa160 = self.dict[nickname]['kappa160']
        wl = self.dict[nickname]['wl']
        nwl = len(wl)
        bands = self.dict[nickname]['bands']
        fitted_paras = []
        temp_paras = parameters[self.dict[nickname]['model']]
        if 'chi2' not in temp_paras:
            temp_paras.append('chi2')
        for para in temp_paras:
            data = fits.getdata(fnhead + para + fnend, header=False)
            if data.ndim == 3:
                data = data[0]
            if displaylog and v_prop[para][0]:
                data = np.log10(data)
            assert data.ndim == 2
            fitted_paras.append(data)
        fn = fnhead + temp_paras[0] + '.pdf' + fnend
        logSigmad_PDF = fits.getdata(fn)
        logSigmads_1d = v_prop['dust.surface.density'][1]
        fn = fnhead + temp_paras[1] + '.pdf' + fnend
        Td_PDF = fits.getdata(fn)
        Tds_1d = v_prop['dust.temperature'][1]
        #
        print(' -- Importing models.')
        bkgcov = fits.getdata(fnhead + 'bkgcov' + fnend, header=False)
        cali_mat2 = fits.getdata(fnhead + 'cali.mat2' + fnend, header=False)

        models = []
        for b in bands:
            fn = 'models/' + b + '_' + model + '.beta=' + \
                str(round(beta_f, 1)) + '.fits.gz'
            models.append(fits.getdata(fn))
        models = np.array(models)
        models = np.moveaxis(models, 0, -1)
        #
        betas = beta_f
        lambdacs = self.dict[nickname]['lambdac_f']
        Tds, beta2s, wdfracs, alphas, loggammas, logUmins = 0, 0, 0, 0, 0, 0
        if model == 'PL':
            logSigmads, alphas, loggammas, logUmins = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['alpha'][1],
                            v_prop['gamma'][1],
                            v_prop['logUmin'][1])
        if model == 'SE':
            logSigmads, Tds, betas = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['dust.temperature'][1],
                            v_prop['beta'][1])
        elif model == 'FB':
            logSigmads, Tds = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['dust.temperature'][1])
        elif model == 'BE':
            logSigmads, Tds, beta2s = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['dust.temperature'][1],
                            v_prop['beta2'][1])
        elif model == 'WD':
            logSigmads, Tds, wdfracs = \
                np.meshgrid(v_prop['dust.surface.density'][1],
                            v_prop['dust.temperature'][1],
                            v_prop['warm.dust.fraction'][1])
        #
        print(' -- Building plotting models.')
        wl_plot = np.linspace(51, 549, 50)
        models_plot = np.empty(list(models.shape)[:-1] + [len(wl_plot)])
        for w in range(len(wl_plot)):
            if model in ['SE', 'FB']:
                models_plot[..., w] = \
                    SEMBB(wl_plot[w], 10**logSigmads, Tds, betas,
                          kappa160=kappa160)
            elif model in ['BEMFB', 'BE']:
                models_plot[..., w] = \
                    BEMBB(wl_plot[w], 10**logSigmads, Tds, betas, lambdacs,
                          beta2s, kappa160=kappa160)
            elif model in ['WD']:
                models_plot[..., w] = \
                    WD(wl_plot[w], 10**logSigmads, Tds, betas, wdfracs,
                       kappa160=kappa160)
            elif model in ['PL']:
                models_plot[..., w] = \
                    PowerLaw(wl_plot[w], 10**logSigmads, alphas, 10**loggammas,
                             logUmins, kappa160=kappa160)
        #
        print(' -- Plotting first model.')

        class Diagnostic_Int:
            def __init__(self):
                self.fig, self.ax = plt.subplots(nrows=2, ncols=2,
                                                 figsize=(10, 10))
                cax = self.ax[0, 0].imshow(fitted_paras[1], origin='lower',
                                           cmap=cmap_map)
                self.point, = self.ax[0, 0].plot(0, 0, 'c*')
                self.ax[0, 0].set_xticklabels([])
                self.ax[0, 0].xaxis.set_ticks_position('none')
                self.ax[0, 0].set_yticklabels([])
                self.ax[0, 0].yaxis.set_ticks_position('none')
                plt.colorbar(cax, ax=self.ax[0, 0])
                self.ax[0, 0].set_title(r'$T_d$ [$K$]', x=0.05, y=0.9,
                                        ha='left')
                plt.show()
                self.cid = \
                    self.point.figure.canvas.mpl_connect('button_press_event',
                                                         self)

            def __call__(self, event):
                # print('click', event)
                if event.inaxes != self.point.axes:
                    return
                j, i = int(round(event.xdata)), int(round(event.ydata))
                self.point.set_data(j, i)
                self.point.figure.canvas.draw()
                #
                self.ax[0, 1].clear()
                self.ax[1, 0].clear()
                self.ax[1, 1].clear()
                plt.draw()
                if not np.isfinite(fitted_paras[1][i, j]):
                    return
                #
                sed_vec = sed[:, i, j].reshape(1, nwl)
                calcov = sed_vec.T * cali_mat2 * sed_vec
                cov_n1 = np.linalg.inv(bkgcov + calcov)
                unc = np.sqrt(np.linalg.inv(cov_n1).diagonal())
                diff = models - sed[:, i, j]
                shape0 = list(diff.shape)[:-1]
                shape1 = shape0 + [1, nwl]
                shape2 = shape0 + [nwl, 1]
                chi2 = np.matmul(np.matmul(diff.reshape(shape1), cov_n1),
                                 diff.reshape(shape2)).reshape(shape0)
                #
                # Selecting samples for plotting
                #
                chi2_threshold = 5.0
                chi2 -= np.nanmin(chi2)
                transp = np.exp(-0.5 * chi2) * 0.2
                chi2_mask = chi2 < chi2_threshold
                plot_pool = np.arange(chi2_mask.size)[chi2_mask.flatten()]
                try:
                    ps = np.random.choice(plot_pool, 40, replace=False)
                except ValueError:
                    print('Total sample size:', len(plot_pool))
                    temp_num = 10 * int(len(plot_pool) // 10)
                    print('Change sample size to:', temp_num)
                    ps = np.random.choice(plot_pool, temp_num, replace=False)
                #
                # MAX parameters:
                #
                am = np.unravel_index(np.argmin(chi2), chi2.shape)
                #
                # Best fit model
                if model in ['SE']:
                    EXP_sed = \
                        SEMBB(wl_plot, 10**fitted_paras[0][i, j],
                              fitted_paras[1][i, j], fitted_paras[2][i, j],
                              kappa160=kappa160)
                    MAX_paras = [logSigmads[am], Tds[am], betas[am]]
                    MAX_sed = \
                        SEMBB(wl_plot, 10**MAX_paras[0],
                              MAX_paras[1], MAX_paras[2],
                              kappa160=kappa160)
                elif model in ['FB']:
                    EXP_sed = \
                        SEMBB(wl_plot, 10**fitted_paras[0][i, j],
                              fitted_paras[1][i, j], betas,
                              kappa160=kappa160)
                    MAX_paras = [logSigmads[am], Tds[am]]
                    MAX_sed = \
                        SEMBB(wl_plot, 10**MAX_paras[0],
                              MAX_paras[1], betas,
                              kappa160=kappa160)
                elif model in ['BEMFB', 'BE']:
                    EXP_sed = \
                        BEMBB(wl_plot, 10**fitted_paras[0][i, j],
                              fitted_paras[1][i, j], betas, lambdacs,
                              fitted_paras[2][i, j], kappa160=kappa160)
                    MAX_paras = [logSigmads[am], Tds[am], beta2s[am]]
                    MAX_sed = \
                        BEMBB(wl_plot, 10**MAX_paras[0],
                              MAX_paras[1], betas, lambdacs,
                              MAX_paras[2], kappa160=kappa160)
                elif model in ['WD']:
                    EXP_sed = \
                        WD(wl_plot, 10**fitted_paras[0][i, j],
                           fitted_paras[1][i, j], betas, fitted_paras[2][i, j],
                           kappa160=kappa160)
                    MAX_paras = [logSigmads[am], Tds[am], wdfracs[am]]
                    MAX_sed = \
                        WD(wl_plot, 10**MAX_paras[0],
                           MAX_paras[1], betas, MAX_paras[2],
                           kappa160=kappa160)
                elif model in ['PL']:
                    EXP_sed = \
                        PowerLaw(wl_plot, 10**fitted_paras[0][i, j],
                                 fitted_paras[1][i, j],
                                 10**fitted_paras[2][i, j],
                                 fitted_paras[3][i, j], kappa160=kappa160)
                    MAX_paras = [logSigmads[am], alphas[am], loggammas[am],
                                 logUmins[am]]
                    MAX_sed = \
                        PowerLaw(wl_plot, 10**MAX_paras[0],
                                 MAX_paras[1], 10**MAX_paras[2],
                                 MAX_paras[3], kappa160=kappa160)
                #
                # Top-right: Example model
                # Plot selected models
                for p in ps:
                    coord = np.unravel_index(p, chi2.shape)
                    self.ax[0, 1].plot(wl_plot, models_plot[coord],
                                       alpha=transp[coord], color='k')
                # Plot EXP and MAX
                self.ax[0, 1].plot(wl_plot, EXP_sed, linewidth=3, label='EXP',
                                   color='orange')
                self.ax[0, 1].plot(wl_plot, MAX_sed, linewidth=3, label='MAX',
                                   color='g')
                # Plot observation
                self.ax[0, 1].errorbar(wl, sed[:, i, j], yerr=unc, fmt='o',
                                       color='red', capsize=10,
                                       label='Herschel data')
                self.ax[0, 1].legend()
                self.ax[0, 1].set_xlabel(r'Wavelength [$\mu m$]')
                self.ax[0, 1].set_ylabel(r'SED [$MJy$ $sr^{-1}$]')
                self.ax[0, 1].set_title('Models and Observation')
                # Bottom-left: Surface density PDF
                self.ax[1, 0].plot(logSigmads_1d, logSigmad_PDF[:, i, j])
                self.ax[1, 0].grid()
                self.ax[1, 0].plot([fitted_paras[0][i, j]] * 2,
                                   [0, np.max(logSigmad_PDF[:, i, j])],
                                   label='EXP',
                                   color='orange')
                self.ax[1, 0].plot([MAX_paras[0]] * 2,
                                   [0, np.max(logSigmad_PDF[:, i, j])],
                                   label='MAX',
                                   color='g')
                self.ax[1, 0].set_yticklabels([])
                self.ax[1, 0].set_xlabel(v_prop['dust.surface.density'][2][1])
                self.ax[1, 0].yaxis.set_ticks_position('none')
                self.ax[1, 0].set_title(r'$log(\Sigma_d)$ PDF', x=0.05, y=0.9,
                                        ha='left')
                # Bottom-right: Temperature PDF
                self.ax[1, 1].plot(Tds_1d, Td_PDF[:, i, j])
                self.ax[1, 1].grid()
                self.ax[1, 1].plot([fitted_paras[1][i, j]] * 2,
                                   [0, np.max(Td_PDF[:, i, j])], label='EXP',
                                   color='orange')
                self.ax[1, 1].plot([MAX_paras[1]] * 2,
                                   [0, np.max(Td_PDF[:, i, j])], label='MAX',
                                   color='g')
                self.ax[1, 1].set_xlabel(v_prop['dust.temperature'][2][0])
                self.ax[1, 1].set_yticklabels([])
                self.ax[1, 1].yaxis.set_ticks_position('none')
                self.ax[1, 1].set_title(r'$T_d$ PDF', x=0.05, y=0.9, ha='left')
                plt.draw()
                # Set title
                title = ''
                for p in range(len(temp_paras)):
                    para = temp_paras[p]
                    if displaylog and v_prop[para][0]:
                        title += v_prop[para][2][1]
                    else:
                        title += v_prop[para][2][0]
                    title += '=' + str(round(fitted_paras[p][i, j], 2)) + '; '
                self.fig.suptitle(title)

        di = Diagnostic_Int()
        del di

    def sed_and_diskmask(self, nickname):
        datapath = self.dict[nickname]['datapath']
        bands = self.dict[nickname]['bands']
        nwl = len(bands)
        if self.dict[nickname]['project_name'] == 'UTOMO18':
            filelist = os.listdir(datapath)
            for fn in filelist:
                temp = fn.split('_')
                if len(temp) > 4:
                    if (temp[1] == 'pacs100') and (temp[-1] == 'mask.fits'):
                        diskmask = fits.getdata(datapath + '/' + fn,
                                                header=False)
                        break
            sed = np.full([nwl] + list(diskmask.shape), np.nan)
            for i in range(nwl):
                for fn in filelist:
                    temp = fn.split('_')
                    if len(temp) > 4:
                        if (temp[1] == bands[i]) and \
                                (temp[-1] != 'mask.fits'):
                            temp = fits.getdata(datapath + '/' + fn,
                                                header=False)
                            sed[i] = temp
                            break
        return sed, diskmask

    def integrated_model_FB(self, project_path='Projects/UTOMO18/'):
        plt.close('all')
        method_abbr = 'FB'
        beta_f = str(1.8)
        bands = ['pacs100', 'pacs160', 'spire250', 'spire350', 'spire500']
        wl = np.array([100, 160, 250, 350, 500])
        nwl = len(bands)
        objects = ['smc', 'lmc', 'm31', 'm33']
        wl_complete = np.linspace(1, 800, 1000)
        #
        if beta_f == '2.0':
            fn = 'hdf5_MBBDust/Calibration.h5'
        else:
            fn = 'hdf5_MBBDust/Calibration_' + beta_f + '.h5'
        with h5py.File(fn, 'r') as hf:
            grp = hf[method_abbr]
            kappa160 = grp['kappa160'].value
        #
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        for ai in range(4):
            name = objects[ai]
            if name in ['lmc', 'smc']:
                int_path = project_path + name + '/integrated_res_13pc/'
            else:
                int_path = project_path + name + '/integrated_res_167pc/'
            i, j = ai // 2, ai % 2
            # Read .csv file for best fit parameters, sed
            fn = int_path + name + '_integrated.csv'
            df = pd.read_csv(fn)
            Sigmad_exp = df['dust.surface.density'][0]
            Td_exp = df['dust.temperature'][0]
            sed = np.empty(nwl)
            sed_vec = sed.reshape(1, nwl)
            for bi in range(nwl):
                sed[bi] = df[bands[bi]][0]
            # Read bkgcov and cali_mat2
            fns = os.listdir(int_path)
            for fn in fns:
                temp = fn.split('_')
                if len(temp) > 1:
                    if temp[1] == 'dust.surface.density.rlcube':
                        fn_Sigmad = int_path + fn
                    elif temp[1] == 'dust.temperature.rlcube':
                        fn_Td = int_path + fn
            #
            all_instr = ['pacs', 'spire', 'mips']
            band_instr = {'pacs70': 'pacs', 'pacs100': 'pacs',
                          'pacs160': 'pacs', 'spire250': 'spire',
                          'spire350': 'spire', 'spire500': 'spire',
                          'mips24': 'mips', 'mips70': 'mips',
                          'mips160': 'mips'}
            cau = {'pacs': 10.0 / 100.0, 'spire': 8.0 / 100.0,
                   'mips': 2.0 / 100.0}
            cru = {'pacs70': 2.0 / 100, 'pacs100': 2.0 / 100,
                   'pacs160': 2.0 / 100, 'spire250': 1.5 / 100,
                   'spire350': 1.5 / 100, 'spire500': 1.5 / 100,
                   'mips24': 4.0 / 100, 'mips70': 7.0 / 100,
                   'mips160': 12.0 / 100}
            cali_mat2 = np.zeros([nwl, nwl])
            for instr in all_instr:
                instr_bands = [bi for bi in range(nwl) if
                               band_instr[bands[bi]] == instr]
                for bi in instr_bands:
                    cali_mat2[bi, bi] += cru[bands[bi]]
                    for bj in instr_bands:
                        cali_mat2[bi, bj] += cau[instr]
            cali_mat2 = cali_mat2**2
            #
            calcov = sed_vec.T * cali_mat2 * sed_vec
            cov_n1 = np.linalg.inv(calcov)
            unc = np.sqrt(np.linalg.inv(cov_n1).diagonal())
            print(name, unc)
            # Read random selection
            Sigmads = fits.getdata(fn_Sigmad).flatten()
            Tds = fits.getdata(fn_Td).flatten()
            # Build (and save) complete models, RSRF, calculate chi^2
            mn = len(Sigmads)
            pr = np.empty(mn)
            models_complete = []
            for mi in range(mn):
                temp = SEMBB(wl_complete, Sigmads[mi], Tds[mi],
                             float(beta_f), kappa160=kappa160)
                models_complete.append([t for t in temp])
                model_short = z0mg_RSRF(wl_complete, temp,
                                        ['PACS_100', 'PACS_160', 'SPIRE_250',
                                         'SPIRE_350', 'SPIRE_500'])
                diff = model_short - sed
                shape0 = list(diff.shape)[:-1]
                shape1 = shape0 + [1, nwl]
                shape2 = shape0 + [nwl, 1]
                chi2 = np.matmul(np.matmul(diff.reshape(shape1), cov_n1),
                                 diff.reshape(shape2)).reshape(shape0)
                pr[mi] = np.exp(-0.5 * chi2)
            pr = pr / np.max(pr)
            # Best fit model
            model_exp = SEMBB(wl_complete, Sigmad_exp, Td_exp, float(beta_f),
                              kappa160=kappa160)
            raw_exp = SEMBB(wl, Sigmad_exp, Td_exp, float(beta_f),
                            kappa160=kappa160)
            rsrf_exp = z0mg_RSRF(wl_complete, model_exp,
                                 ['PACS_100', 'PACS_160', 'SPIRE_250',
                                  'SPIRE_350', 'SPIRE_500'])
            print(name, 'ccf:', rsrf_exp / raw_exp)
            model_exp = SEMBB(wl_complete, Sigmad_exp, Td_exp, float(beta_f),
                              kappa160=kappa160)
            #
            # Plotting
            #
            for mi in range(mn):
                ax[i, j].plot(wl_complete, models_complete[mi],
                              alpha=pr[mi], color='gray')
            # Plot EXP and MAX
            ax[i, j].plot(wl_complete, model_exp, linewidth=3, label='EXP',
                          color='orange')
            ax[i, j].plot(wl_complete, models_complete[np.argmax(pr)],
                          linewidth=3, label='MAX', color='g')
            # Plot observation
            ax[i, j].errorbar(wl, sed, yerr=unc, fmt='o', color='red',
                              capsize=10, label='Herschel data')
            ax[i, j].legend()
            ax[i, j].set_xlabel(r'$\lambda$ [$\mu m$]')
            ax[i, j].set_ylabel(r'$F_\nu$ [$MJy$ $sr^{-1}$]')
            ax[i, j].set_title(name, x=0.05, y=0.9, ha='left')
            #
        fn = project_path + 'integrated/models'
        if self.TRUE_FOR_PNG:
            fig.savefig(fn + '.png', bbox_inches='tight')
        else:
            with PdfPages(fn + '.pdf') as pp:
                pp.savefig(fig, bbox_inches='tight')
        plt.close('all')
        print(' -- Plots saved. All figures closing.')

    def integrated_model_FBPL(self, project_path='Projects/UTOMO18/'):
        plt.close('all')
        beta_f = '1.8'
        bands = ['pacs100', 'pacs160', 'spire250', 'spire350', 'spire500']
        wl = np.array([100, 160, 250, 350, 500])
        nwl = len(bands)
        objects = ['smc', 'lmc', 'm31', 'm33']
        objects_c = ['SMC', 'LMC', 'M31', 'M33']
        wl_complete = np.linspace(1, 800, 1000)
        #
        if beta_f == '2.0':
            fn = 'hdf5_MBBDust/Calibration.h5'
        else:
            fn = 'hdf5_MBBDust/Calibration_' + beta_f + '.h5'
        with h5py.File(fn, 'r') as hf:
            grp = hf['FB']
            kappa160 = grp['kappa160'].value
            grp = hf['PL']
            kappa160PL = grp['kappa160'].value
        #
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        for ai in range(4):
            name = objects[ai]
            name_c = objects_c[ai]
            if name in ['lmc', 'smc']:
                int_path = project_path + name + '/integrated_res_13pc/'
                int_pathPL = project_path + name + '_PL/integrated_res_13pc/'
            else:
                int_path = project_path + name + '/integrated_res_167pc/'
                int_pathPL = project_path + name + '_PL/integrated_res_167pc/'
            i, j = ai // 2, ai % 2
            # Read .csv file for best fit parameters, sed
            fn = int_path + name + '_integrated.csv'
            df = pd.read_csv(fn)
            sed = np.empty(nwl)
            sed_vec = sed.reshape(1, nwl)
            for bi in range(nwl):
                sed[bi] = df[bands[bi]][0]
            Sigmad_exp = df['dust.surface.density'][0]
            Td_exp = df['dust.temperature'][0]
            # PL
            fn = int_pathPL + name + '_integrated.csv'
            df = pd.read_csv(fn)
            Sigmad_expPL = df['dust.surface.density'][0]
            alpha_expPL = df['alpha'][0]
            gamma_expPL = df['gamma'][0]
            logUmin_expPL = df['logUmin'][0]
            # Read realize cube
            fns = os.listdir(int_path)  # FB
            for fn in fns:
                temp = fn.split('_')
                if len(temp) > 1:
                    if temp[1] == 'dust.surface.density.rlcube':
                        fn_Sigmad = int_path + fn
                    elif temp[1] == 'dust.temperature.rlcube':
                        fn_Td = int_path + fn
            fns = os.listdir(int_pathPL)  # PL
            for fn in fns:
                temp = fn.split('_')
                if len(temp) > 1:
                    if temp[1] == 'dust.surface.density.rlcube':
                        fn_SigmadPL = int_pathPL + fn
                    elif temp[1] == 'alpha.rlcube':
                        fn_alphaPL = int_pathPL + fn
                    elif temp[1] == 'gamma.rlcube':
                        fn_gammaPL = int_pathPL + fn
                    elif temp[1] == 'logUmin.rlcube':
                        fn_logUminPL = int_pathPL + fn
            #
            all_instr = ['pacs', 'spire', 'mips']
            band_instr = {'pacs70': 'pacs', 'pacs100': 'pacs',
                          'pacs160': 'pacs', 'spire250': 'spire',
                          'spire350': 'spire', 'spire500': 'spire',
                          'mips24': 'mips', 'mips70': 'mips',
                          'mips160': 'mips'}
            cau = {'pacs': 10.0 / 100.0, 'spire': 8.0 / 100.0,
                   'mips': 2.0 / 100.0}
            cru = {'pacs70': 2.0 / 100, 'pacs100': 2.0 / 100,
                   'pacs160': 2.0 / 100, 'spire250': 1.5 / 100,
                   'spire350': 1.5 / 100, 'spire500': 1.5 / 100,
                   'mips24': 4.0 / 100, 'mips70': 7.0 / 100,
                   'mips160': 12.0 / 100}
            cali_mat2 = np.zeros([nwl, nwl])
            for instr in all_instr:
                instr_bands = [bi for bi in range(nwl) if
                               band_instr[bands[bi]] == instr]
                for bi in instr_bands:
                    cali_mat2[bi, bi] += cru[bands[bi]]
                    for bj in instr_bands:
                        cali_mat2[bi, bj] += cau[instr]
            cali_mat2 = cali_mat2**2
            #
            calcov = sed_vec.T * cali_mat2 * sed_vec
            cov_n1 = np.linalg.inv(calcov)
            unc = np.sqrt(np.linalg.inv(cov_n1).diagonal())
            print(name, unc)
            #
            # FB
            #
            # Read random selection
            num = 100
            Sigmads = fits.getdata(fn_Sigmad).flatten()[:num]
            Tds = fits.getdata(fn_Td).flatten()[:num]
            # Build (and save) complete models, RSRF, calculate chi^2
            mn = len(Sigmads)
            pr = np.empty(mn)
            models_complete = []
            for mi in range(mn):
                temp = SEMBB(wl_complete, Sigmads[mi], Tds[mi],
                             float(beta_f), kappa160=kappa160)
                models_complete.append([t for t in temp])
                model_short = z0mg_RSRF(wl_complete, temp,
                                        ['PACS_100', 'PACS_160', 'SPIRE_250',
                                         'SPIRE_350', 'SPIRE_500'])
                diff = model_short - sed
                shape0 = list(diff.shape)[:-1]
                shape1 = shape0 + [1, nwl]
                shape2 = shape0 + [nwl, 1]
                chi2 = np.matmul(np.matmul(diff.reshape(shape1), cov_n1),
                                 diff.reshape(shape2)).reshape(shape0)
                pr[mi] = np.exp(-0.5 * chi2)
            pr = pr / np.max(pr) * 0.15
            #
            # PL
            #
            # Read random selection
            Sigmads = fits.getdata(fn_SigmadPL).flatten()[:num]
            alphas = fits.getdata(fn_alphaPL).flatten()[:num]
            gammas = fits.getdata(fn_gammaPL).flatten()[:num]
            logUmins = fits.getdata(fn_logUminPL).flatten()[:num]
            # Build (and save) complete models, RSRF, calculate chi^2
            mn = len(Sigmads)
            prPL = np.empty(mn)
            models_completePL = []
            for mi in range(mn):
                temp = PowerLaw(wl_complete, Sigmads[mi], alphas[mi],
                                gammas[mi], logUmins[mi], kappa160=kappa160PL)
                models_completePL.append([t for t in temp])
                model_short = z0mg_RSRF(wl_complete, temp,
                                        ['PACS_100', 'PACS_160', 'SPIRE_250',
                                         'SPIRE_350', 'SPIRE_500'])
                diff = model_short - sed
                shape0 = list(diff.shape)[:-1]
                shape1 = shape0 + [1, nwl]
                shape2 = shape0 + [nwl, 1]
                chi2 = np.matmul(np.matmul(diff.reshape(shape1), cov_n1),
                                 diff.reshape(shape2)).reshape(shape0)
                prPL[mi] = np.exp(-0.5 * chi2)
            prPL = prPL / np.max(prPL) * 0.15
            #
            # Plotting FB + PL
            #
            models_complete = np.array(models_complete)
            models_completePL = np.array(models_completePL)
            """
            for mi in range(mn):
                ax[i, j].plot(wl_complete, models_complete[mi],
                              alpha=0.05, color='green')
                ax[i, j].plot(wl_complete, models_completePL[mi],
                              alpha=0.05, color='blue')
            """
            ax[i, j].fill_between(wl_complete,
                                  np.max(models_complete, axis=0),
                                  np.min(models_complete, axis=0),
                                  color='#1b9e77', alpha=0.3)
            ax[i, j].fill_between(wl_complete,
                                  np.max(models_completePL, axis=0),
                                  np.min(models_completePL, axis=0),
                                  color='#7570b3', alpha=0.3)
            """
            # Plot max and legend
            ax[i, j].plot(wl_complete, models_complete[np.argmax(pr)],
                          color='green', label='FB')
            ax[i, j].plot(wl_complete, models_completePL[np.argmax(pr)],
                          color='blue', label='PL')
            """
            # Plot exp and legend
            model_exp = SEMBB(wl_complete, Sigmad_exp, Td_exp, float(beta_f),
                              kappa160=kappa160)
            ax[i, j].plot(wl_complete, model_exp,
                          color='#1b9e77', label='Single-temperature')
            model_expPL = PowerLaw(wl_complete, Sigmad_expPL, alpha_expPL,
                                   gamma_expPL, logUmin_expPL,
                                   kappa160=kappa160PL)
            ax[i, j].plot(wl_complete, model_expPL, linestyle='--',
                          color='#7570b3', label='Multi-temperature')
            # Plot observation
            ax[i, j].errorbar(wl, sed, yerr=unc, fmt='o', color='#d95f02',
                              capsize=10, label='Herschel data')
            ax[i, j].legend()
            ax[i, j].set_xlabel(r'$\lambda$ [$\mu m$]')
            ax[i, j].set_ylabel(r'$I_\nu$ [$MJy$ $sr^{-1}$]')
            ax[i, j].set_title(name_c, x=0.05, y=0.9, ha='left')
            ax[i, j].grid(linestyle='--')
            #
        fn = project_path + 'integrated/models'
        if self.TRUE_FOR_PNG:
            fig.savefig(fn + '.png', bbox_inches='tight')
        else:
            with PdfPages(fn + '.pdf') as pp:
                pp.savefig(fig, bbox_inches='tight')
        plt.close('all')
        print(' -- Plots saved. All figures closing.')
