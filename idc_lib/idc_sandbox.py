from h5py import File
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
# from time import clock
import astropy.units as u
from astropy.constants import c, h, k_B
# from .gal_data import gal_data


wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
# (O/H)_\odot, ZB12
solar_oxygen_bundance = 8.69


def simple_profile(data, radius, bins, gas=None, gas_weighted=True):
    gas = np.ones(data.shape) if not (gas_weighted or gas) else gas
    n_nanmask = ~np.isnan(gas + radius + data)
    gas, radius, data = gas[n_nanmask], radius[n_nanmask], data[n_nanmask]

    r, profile = [], []
    rbins = np.linspace(radius.min(), radius.max(), bins)
    for i in range(bins - 1):
        mask = (rbins[i] <= radius) * (radius < rbins[i + 1])
        r.append((rbins[i] + rbins[i + 1]) / 2)
        with np.errstate(invalid='ignore'):
            profile.append(np.sum(data[mask] * gas[mask]) / np.sum(gas[mask]))
    return np.array(r), np.array(profile)


def map2bin(data, binlist, binmap):
    nanmask = np.isnan(data + binmap)
    binmap[nanmask] = binlist.max() + 1
    for bin_ in binlist:
        mask = binmap == bin_
        data[mask] = data[mask].mean()
    return data


def list2bin(listData, binlist, binmap):
    assert len(listData) == len(binlist)
    data = np.full_like(binmap, np.nan, dtype=float)
    for i in range(len(binlist)):
        data[binmap == binlist[i]] = listData[i]
    return data


def B(T, freq=nu):
    """Return blackbody SED of temperature T(with unit) in MJy"""
    with np.errstate(over='ignore'):
        return (2 * h * freq**3 / c**2 / (np.exp(h * freq / k_B / T) - 1)
                ).to(u.Jy).value * 1E-6


def model(wl, sigma, T, beta, freq=nu):
    """Return fitted SED in MJy"""
    const = 2.0891E-4
    kappa160 = 9.6 * np.pi
    return const * kappa160 * (160.0 / wl)**beta * \
        sigma * B(T * u.K, freq)


def Residue_maps(name='NGC5457', rbin=51, dbin=100, tbin=90, SigmaDoff=2.,
                 Toff=20, dr25=0.025,
                 cmap0='gist_heat', cmap1='Greys', cmap2='seismic',
                 cmap3='Reds'):
    plt.close('all')
    plt.ioff()
    print('Loading data...\n')
    with File('output/Voronoi_data.h5', 'r') as hf:
        grp = hf[name]
        binlist = np.array(grp['BINLIST'])
        binmap = np.array(grp['BINMAP'])
        # aGas = np.array(grp['GAS_AVG'])
        # SigmaGas = list2bin(aGas, binlist, binmap)
        aRadius = np.array(grp['Radius_avg'])
        aSED = np.array(grp['Herschel_SED'])
        # acov_n1 = np.array(grp['Herschel_covariance_matrix'])
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        R25 = float(np.array(grp['R25_KPC']))
        aRadius /= R25
        ubRadius = np.array(grp['RADIUS_KPC']) / R25
        # Radius = list2bin(aRadius, binlist, binmap)
        # H2 = map2bin(np.array(grp['HERACLES']), binlist, binmap)
        # SFR = map2bin(np.array(grp['SFR']), binlist, binmap)
        # SMSD = map2bin(np.array(grp['SMSD']), binlist, binmap)
        # bkgcov = np.array(grp['HERSCHEL_011111_BKGCOV'])
    # Calculating Image scale
    lbl = len(binlist)
    extent = np.array([-ubRadius[:, 0].max(), ubRadius[:, -1].max(),
                       -ubRadius[0, :].max(), ubRadius[-1, :].max()])
    # uncs = np.array([np.sqrt((np.linalg.inv(acov_n1[i]) /
    #                           np.sum(binmap == binlist[i])).diagonal())
    #                  for i in range(lbl)])

    fn_AF = 'output/Dust_data_AF_' + name + '.h5'
    with File(fn_AF, 'r') as hf:
        aSigmaD_AF = 10**np.array(hf['Dust_surface_density_log'])
        # SigmaD = list2bin(10**np.array(hf['Dust_surface_density_log']),
        #                   binlist, binmap)
        # with np.errstate(invalid='ignore'):
        #     DGR = SigmaD / SigmaGas
        # SigmaD_dexerr = \
        #     list2bin(np.array(hf['Dust_surface_density_err_dex']), binlist,
        #              binmap)
        aT_AF = np.array(hf['Dust_temperature'])
        # T = list2bin(np.array(hf['Dust_temperature']), binlist, binmap)
        # T_err = list2bin(np.array(hf['Dust_temperature_err']), binlist,
        #                  binmap)
        aBeta_AF = np.array(hf['beta'])
        # Beta = list2bin(np.array(hf['beta']), binlist, binmap)
        # Beta_err = list2bin(np.array(hf['beta_err']), binlist, binmap)
        # SigmaDs = 10**np.array(hf['logsigmas'])
        # aPDFs = np.array(hf['PDF'])
        # Ts = np.array(hf['Ts'])
        # a_T_PDFs = np.array(hf['PDF_T'])
        # aModel = np.array(hf['Best_fit_model'])
        pass

    fn_FB = 'output/Dust_data_FB_' + name + '.h5'
    with File(fn_FB, 'r') as hf:
        aSigmaD_FB = 10**np.array(hf['Dust_surface_density_log'])
        # SigmaD_fb = list2bin(aSigmaD_fb, binlist, binmap)
        # with np.errstate(invalid='ignore'):
        #     DGR_fb = SigmaD_fb / SigmaGas
        # SigmaD_dexerr_fb = \
        #     list2bin(np.array(hf['Dust_surface_density_err_dex']), binlist,
        #              binmap)
        aT_FB = np.array(hf['Dust_temperature'])
        # T_fb = list2bin(aT_fb, binlist, binmap)
        # T_err_fb = list2bin(np.array(hf['Dust_temperature_err']), binlist,
        #                     binmap)
        aBeta_FB = np.array(hf['beta'])
        # Beta_fb = list2bin(aBeta_fb, binlist, binmap)
        # SigmaDs_fb = 10**np.array(hf['logsigmas'])
        # aPDFs_fb = np.array(hf['PDF'])
        # Ts_fb = np.array(hf['Ts'])
        # a_T_PDFs_fb = np.array(hf['PDF_T'])
        # aModel = np.array(hf['Best_fit_model'])

    fn_FBT = 'output/Dust_data_FBT_' + name + '.h5'
    with File(fn_FBT, 'r') as hf:
        aSigmaD_FBT = 10**np.array(hf['Dust_surface_density_log'])
        # SigmaD_FBT = list2bin(aSigmaD_FBT, binlist, binmap)
        # with np.errstate(invalid='ignore'):
        #     DGR_FBT = SigmaD_FBT / SigmaGas
        # SigmaD_dexerr_FBT = \
        #     list2bin(np.array(hf['Dust_surface_density_err_dex']), binlist,
        #              binmap)
        aT_FBT = np.array(hf['Dust_temperature'])
        # T_FBT = list2bin(aT_FBT, binlist, binmap)
        # T_err_FBT = list2bin(np.array(hf['Dust_temperature_err']), binlist,
        #                       binmap)
        aBeta_FBT = np.array(hf['beta'])
        # Beta_FBT = list2bin(np.array(hf['beta']), binlist, binmap)
        # SigmaDs_FBT = 10**np.array(hf['logsigmas'])
        # aPDFs_FBT = np.array(hf['PDF'])
        pass

    aModel_exp5_AF = np.zeros_like(aSED)
    aModel_exp5_FB = np.zeros_like(aSED)
    aModel_exp5_FBT = np.zeros_like(aSED)
    """
    Calculate color correction factors
    """
    pacs_rsrf = pd.read_csv("data/RSRF/PACS_RSRF.csv")
    pacs_wl = pacs_rsrf['Wavelength'].values
    pacs_nu = (c / pacs_wl / u.um).to(u.Hz)
    pacs100dnu = \
        pacs_rsrf['PACS_100'].values * pacs_rsrf['dnu'].values[0]
    pacs160dnu = \
        pacs_rsrf['PACS_160'].values * pacs_rsrf['dnu'].values[0]
    del pacs_rsrf
    #
    p_models = np.zeros([lbl, len(pacs_wl)])
    for j in range(lbl):
        p_models[j] = model(pacs_wl, aSigmaD_AF[j], aT_AF[j],
                            aBeta_AF[j], pacs_nu)
    aModel_exp5_AF[:, 0] = np.sum(p_models * pacs100dnu, axis=1) / \
        np.sum(pacs100dnu * pacs_wl / wl[0])
    aModel_exp5_AF[:, 1] = np.sum(p_models * pacs160dnu, axis=1) / \
        np.sum(pacs160dnu * pacs_wl / wl[1])
    for j in range(lbl):
        p_models[j] = model(pacs_wl, aSigmaD_FB[j], aT_FB[j],
                            aBeta_FB[j], pacs_nu)
    aModel_exp5_FB[:, 0] = np.sum(p_models * pacs100dnu, axis=1) / \
        np.sum(pacs100dnu * pacs_wl / wl[0])
    aModel_exp5_FB[:, 1] = np.sum(p_models * pacs160dnu, axis=1) / \
        np.sum(pacs160dnu * pacs_wl / wl[1])
    for j in range(lbl):
        p_models[j] = model(pacs_wl, aSigmaD_FBT[j], aT_FBT[j],
                            aBeta_FBT[j], pacs_nu)
    aModel_exp5_FBT[:, 0] = np.sum(p_models * pacs100dnu, axis=1) / \
        np.sum(pacs100dnu * pacs_wl / wl[0])
    aModel_exp5_FBT[:, 1] = np.sum(p_models * pacs160dnu, axis=1) / \
        np.sum(pacs160dnu * pacs_wl / wl[1])
    #
    del pacs_wl, pacs100dnu, pacs160dnu, p_models
    ##
    spire_rsrf = pd.read_csv("data/RSRF/SPIRE_RSRF.csv")
    spire_wl = spire_rsrf['Wavelength'].values
    spire_nu = (c / spire_wl / u.um).to(u.Hz)
    spire250dnu = \
        spire_rsrf['SPIRE_250'].values * spire_rsrf['dnu'].values[0]
    spire350dnu = \
        spire_rsrf['SPIRE_350'].values * spire_rsrf['dnu'].values[0]
    spire500dnu = \
        spire_rsrf['SPIRE_500'].values * spire_rsrf['dnu'].values[0]
    del spire_rsrf
    #
    s_models = np.zeros([lbl, len(spire_wl)])
    for j in range(lbl):
        s_models[j] = model(spire_wl, aSigmaD_AF[j], aT_AF[j],
                            aBeta_AF[j], spire_nu)
    aModel_exp5_AF[:, 2] = np.sum(s_models * spire250dnu, axis=1) / \
        np.sum(spire250dnu * spire_wl / wl[2])
    aModel_exp5_AF[:, 3] = np.sum(s_models * spire350dnu, axis=1) / \
        np.sum(spire350dnu * spire_wl / wl[3])
    aModel_exp5_AF[:, 4] = np.sum(s_models * spire500dnu, axis=1) / \
        np.sum(spire500dnu * spire_wl / wl[4])
    for j in range(lbl):
        s_models[j] = model(spire_wl, aSigmaD_FB[j], aT_FB[j],
                            aBeta_FB[j], spire_nu)
    aModel_exp5_FB[:, 2] = np.sum(s_models * spire250dnu, axis=1) / \
        np.sum(spire250dnu * spire_wl / wl[2])
    aModel_exp5_FB[:, 3] = np.sum(s_models * spire350dnu, axis=1) / \
        np.sum(spire350dnu * spire_wl / wl[3])
    aModel_exp5_FB[:, 4] = np.sum(s_models * spire500dnu, axis=1) / \
        np.sum(spire500dnu * spire_wl / wl[4])
    for j in range(lbl):
        s_models[j] = model(spire_wl, aSigmaD_FBT[j], aT_FBT[j],
                            aBeta_FBT[j], spire_nu)
    aModel_exp5_FBT[:, 2] = np.sum(s_models * spire250dnu, axis=1) / \
        np.sum(spire250dnu * spire_wl / wl[2])
    aModel_exp5_FBT[:, 3] = np.sum(s_models * spire350dnu, axis=1) / \
        np.sum(spire350dnu * spire_wl / wl[3])
    aModel_exp5_FBT[:, 4] = np.sum(s_models * spire500dnu, axis=1) / \
        np.sum(spire500dnu * spire_wl / wl[4])
    #
    del spire_nu, s_models
    #
    del spire_wl, spire250dnu, spire350dnu, spire500dnu
    """
    2X2: Residue Maps (AF)
    """
    rmax, rmin = 0.6, -0.6
    print('2X2: Residue Maps (AF)')
    titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']
    # maxs = [np.nanmax(np.append(temp[0], temp[2])),
    #         np.nanmax(np.append(temp[1], temp[3]))]
    # mins = [np.nanmin(np.append(temp[0], temp[2])),
    #         np.nanmin(np.append(temp[1], temp[3]))]
    for i in range(5):
        sed_map = list2bin(aSED[:, i], binlist, binmap)
        model_map = list2bin(aModel_exp5_AF[:, i], binlist, binmap)
        print(titles[i] + ':', sed_map[55, 55], model_map[55, 55])
        residue_map = sed_map - model_map
        fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
        p, q = 0, 0
        cax = ax[p, q].imshow(sed_map, norm=LogNorm(),
                              origin='lower', cmap=cmap0, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Observed')
        p, q = 0, 1
        cax = ax[p, q].imshow(model_map, norm=LogNorm(),
                              origin='lower', cmap=cmap0, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Model')
        p, q = 1, 0
        cax = ax[p, q].imshow(residue_map,
                              origin='lower', cmap=cmap0, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Residue')
        p, q = 1, 1
        cax = ax[p, q].imshow(residue_map / sed_map, vmin=rmin, vmax=rmax,
                              origin='lower', cmap=cmap2, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Residue / Observed')
        fig.suptitle(titles[i] + ' (AF)')
        fig.tight_layout()
        fig.savefig('output/_2X2Residue_Map_AF_' + titles[i] + '.png')
        plt.close(fig)
    """
    2X2: Residue Maps (FB)
    """
    print('2X2: Residue Maps (FB)')
    titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']
    # maxs = [np.nanmax(np.append(temp[0], temp[2])),
    #         np.nanmax(np.append(temp[1], temp[3]))]
    # mins = [np.nanmin(np.append(temp[0], temp[2])),
    #         np.nanmin(np.append(temp[1], temp[3]))]
    for i in range(5):
        sed_map = list2bin(aSED[:, i], binlist, binmap)
        model_map = list2bin(aModel_exp5_FB[:, i], binlist, binmap)
        print(titles[i] + ':', sed_map[55, 55], model_map[55, 55])
        residue_map = sed_map - model_map
        fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
        p, q = 0, 0
        cax = ax[p, q].imshow(sed_map, norm=LogNorm(),
                              origin='lower', cmap=cmap0, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Observed')
        p, q = 0, 1
        cax = ax[p, q].imshow(model_map, norm=LogNorm(),
                              origin='lower', cmap=cmap0, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Model')
        p, q = 1, 0
        cax = ax[p, q].imshow(residue_map,
                              origin='lower', cmap=cmap0, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Residue')
        p, q = 1, 1
        cax = ax[p, q].imshow(residue_map / sed_map, vmin=rmin, vmax=rmax,
                              origin='lower', cmap=cmap2, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Residue / Observed')
        fig.suptitle(titles[i] + ' (FB)')
        fig.tight_layout()
        fig.savefig('output/_2X2Residue_Map_FB_' + titles[i] + '.png')
        plt.close(fig)
    """
    2X2: Residue Maps (FBT)
    """
    print('2X2: Residue Maps (FBT)')
    titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']
    # maxs = [np.nanmax(np.append(temp[0], temp[2])),
    #         np.nanmax(np.append(temp[1], temp[3]))]
    # mins = [np.nanmin(np.append(temp[0], temp[2])),
    #         np.nanmin(np.append(temp[1], temp[3]))]
    for i in range(5):
        sed_map = list2bin(aSED[:, i], binlist, binmap)
        model_map = list2bin(aModel_exp5_FBT[:, i], binlist, binmap)
        print(titles[i] + ':', sed_map[55, 55], model_map[55, 55])
        residue_map = sed_map - model_map
        fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
        p, q = 0, 0
        cax = ax[p, q].imshow(sed_map, norm=LogNorm(),
                              origin='lower', cmap=cmap0, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Observed')
        p, q = 0, 1
        cax = ax[p, q].imshow(model_map, norm=LogNorm(),
                              origin='lower', cmap=cmap0, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Model')
        p, q = 1, 0
        cax = ax[p, q].imshow(residue_map,
                              origin='lower', cmap=cmap0, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Residue')
        p, q = 1, 1
        cax = ax[p, q].imshow(residue_map / sed_map, vmin=rmin, vmax=rmax,
                              origin='lower', cmap=cmap2, extent=extent)
        fig.colorbar(cax, ax=ax[p, q])
        ax[p, q].set_title('Residue / Observed')
        fig.suptitle(titles[i] + ' (FBT)')
        fig.tight_layout()
        fig.savefig('output/_2X2Residue_Map_FBT_' + titles[i] + '.png')
        plt.close(fig)
    """
    End plotting
    """
    plt.close('all')
