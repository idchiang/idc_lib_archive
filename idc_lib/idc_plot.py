from h5py import File
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from time import clock


wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
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


def read_dust_file(name='NGC5457', rbin=51, dbin=250, tbin=30, off=45.,
                   dr25=0.025, fixed_beta=True,
                   cmap0='gist_heat', cmap1='Reds', cmap2='seismic'):
    plt.close('all')
    plt.ioff()

    print('Loading data...\n')
    if fixed_beta:
        fn = 'output/' + name + '_dust_data_fb.h5'
    else:
        fn = 'output/' + name + '_dust_data.h5'

    with File(fn, 'r') as hf:
        aSigmaD = 10**np.array(hf['Dust_surface_density_log'])
        aSigmaD_dexerr = np.array(hf['Dust_surface_density_err_dex'])  # in dex
        aT = np.array(hf['Dust_temperature'])
        aT_err = np.array(hf['Dust_temperature_err'])
        aBeta = np.array(hf['beta'])
        aBeta_err = np.array(hf['beta_err'])
        aSED = np.array(hf['Herschel_SED'])
        aCov_n1 = np.array(hf['Herschel_covariance_matrix'])
        aGas = np.array(hf['Total_gas'])
        binmap = np.array(hf['Binmap'])
        binlist = np.array(hf['Binlist'])
        aRadius = np.array(hf['Radius_avg'])  # kpc
        SigmaDs = 10**np.array(hf['logsigmas'])
        if len(SigmaDs.shape) == 2:
            SigmaDs = SigmaDs[:, 0]
        Ts = np.array(hf['Ts'])
        aPDFs = np.array(hf['PDF'])
        a_T_PDFs = np.array(hf['PDF_T'])
        aModel = np.array(hf['Best_fit_model'])

    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        H2 = map2bin(np.array(grp['HERACLES']), binlist, binmap)
        ubRadius = np.array(grp['RADIUS_KPC'])
        SFR = map2bin(np.array(grp['SFR']), binlist, binmap)
        SMSD = map2bin(np.array(grp['SMSD']), binlist, binmap)
        R25 = float(np.array(grp['R25_KPC']))

    # Filtering bad fits
    diffs = (aSED - aModel).reshape(-1, 5)
    aChi2 = np.array([np.dot(np.dot(diffs[i].T, aCov_n1[i]), diffs[i])
                      for i in range(len(binlist))])
    # Converting Radius
    aRadius /= R25
    ubRadius /= R25
    # Calculating Image scale
    extent = np.array([-ubRadius[:, 0].max(), ubRadius[:, -1].max(),
                       -ubRadius[0, :].max(), ubRadius[-1, :].max()])

    # Constructing maps
    SigmaD = list2bin(aSigmaD, binlist, binmap)
    SigmaD_dexerr = list2bin(aSigmaD_dexerr, binlist, binmap)
    T = list2bin(aT, binlist, binmap)
    T_err = list2bin(aT_err, binlist, binmap)
    Beta = list2bin(aBeta, binlist, binmap)
    Beta_err = list2bin(aBeta_err, binlist, binmap)
    SigmaGas = list2bin(aGas, binlist, binmap)
    Chi2 = list2bin(aChi2, binlist, binmap)
    Radius = list2bin(aRadius, binlist, binmap)
    with np.errstate(invalid='ignore'):
        DGR = SigmaD / SigmaGas
        Model_d_SED = np.array([list2bin(aModel[:, i] / aSED[:, i],
                                         binlist, binmap) for i in range(5)])
    del aSigmaD, aSigmaD_dexerr, aT, aT_err, aBeta, aBeta_err, aSED, aCov_n1, \
        aModel

    """ Croxall metallicity """
    print('Plotting metallicity...\n')
    # Plot points with metallicity measurements on H2 map
    plt.close('all')
    mtl = pd.read_csv('output/' + name + '_metal.csv')
    l = len(mtl)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(H2, origin='lower', cmap=cmap0, norm=LogNorm())
    ax.scatter(mtl.new_c1, mtl.new_c2, s=100, marker='s', facecolors='none',
               edgecolors='c')
    fig.savefig('output/' + name + '_Metallicity_on_H2.png')
    # Grab data for plotting
    DGR_at_ROA = Radius_at_ROA = Rel_Oxygen_Abd = np.array([])
    for i in range(l):
        j, k = int(mtl['new_c2'].iloc[i]), int(mtl['new_c1'].iloc[i])
        if j < 0 or j >= DGR.shape[0] or k < 0 or k >= DGR.shape[1]:
            pass
        else:
            DGR_at_ROA = np.append(DGR_at_ROA, DGR[j, k])
            Radius_at_ROA = np.append(Radius_at_ROA, ubRadius[j, k])
            Rel_Oxygen_Abd = np.append(Rel_Oxygen_Abd,
                                       10**(mtl.iloc[i]['12+log(O/H)'] -
                                            solar_oxygen_bundance))
    DGR_d_91 = DGR_at_ROA / (0.0091 / 1.36)
    l = len(DGR_d_91)
    # Plot metallicity vs. DGR
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.scatter(Rel_Oxygen_Abd, DGR_d_91, c='r', s=15, label='Data points')
    ax.plot(Rel_Oxygen_Abd, Rel_Oxygen_Abd, label='x=y')
    ax.plot([10**(8.0 - solar_oxygen_bundance)] * 2,
            [DGR_d_91.min(), DGR_d_91.max()], '-k', alpha=0.5, label='8.0')
    ax.plot([10**(8.2 - solar_oxygen_bundance)] * 2,
            [DGR_d_91.min(), DGR_d_91.max()], '-k', alpha=0.5, label='8.2')
    ax.set_xlabel(r'$(O/H)/(O/H)_\odot$', size=20)
    ax.set_ylabel(r'$DGR / 0.0067$', size=20)
    ax.legend()
    fig.savefig('output/' + name + '_Metallicity_vs_DGR.png')
    # DGR & (O/H) vs. Radius twin axis
    boundary_ratio = 1.2
    fig, ax1 = plt.subplots(figsize=(10, 7.5))
    max1 = Rel_Oxygen_Abd.max() * 5
    max2, min2 = np.nanmax(DGR_at_ROA), np.nanmin(DGR_at_ROA)
    min1 = max1 * (min2 / max2)
    ax1.set_yscale('log')
    ax1.scatter(Radius_at_ROA, Rel_Oxygen_Abd, c='r', s=15,
                label=r'$(O/H)/(O/H)_\odot$')
    ax1.set_xlabel(r'Radius ($R25$)', size=20)
    ax1.set_ylabel(r'$(O/H)/(O/H)_\odot$', size=20, color='r')
    ax1.tick_params('y', colors='r')
    ax1.set_ylim([min1 / boundary_ratio, max1 * boundary_ratio])
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.scatter(Radius_at_ROA, DGR_at_ROA, c='b', s=15, label=r'$DGR$')
    ax2.set_ylabel(r'$DGR$', size=20, color='b')
    ax2.tick_params('y', colors='b')
    ax2.set_ylim([min2 / boundary_ratio, max2 * boundary_ratio])
    fig.tight_layout()
    fig.savefig('output/' + name + '_Metallicity_DGR_Profile.png')
    del mtl, l, DGR_at_ROA, DGR_d_91, Radius_at_ROA, Rel_Oxygen_Abd, ax1, ax2

    """ Plot fitting results """
    print('Plotting fitting results...\n')
    # Fitting results
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    ax[0, 0].set_title(r'$\Sigma_D$ $(M_\odot pc^{-2})$', size=20)
    cax[0, 0] = ax[0, 0].imshow(SigmaD, origin='lower', cmap=cmap0,
                                extent=extent, norm=LogNorm())
    ax[1, 0].set_title(r'$\Sigma_D$ error (dex)', size=20)
    cax[1, 0] = ax[1, 0].imshow(SigmaD_dexerr, origin='lower', cmap=cmap0,
                                extent=extent)
    ax[0, 1].set_title(r'$T_d$ ($K$)', size=20)
    cax[0, 1] = ax[0, 1].imshow(T, origin='lower', cmap=cmap0,
                                extent=extent)
    ax[1, 1].set_title(r'$T_d$ error ($K$)', size=20)
    cax[1, 1] = ax[1, 1].imshow(T_err, origin='lower', cmap=cmap0,
                                extent=extent)
    ax[0, 2].set_title(r'$\beta$', size=20)
    cax[0, 2] = ax[0, 2].imshow(Beta, origin='lower', cmap=cmap0,
                                extent=extent)
    ax[1, 2].set_title(r'$\beta$ error', size=20)
    cax[1, 2] = ax[1, 2].imshow(Beta_err, origin='lower', cmap=cmap0,
                                extent=extent)
    for i in range(2):
        for j in range(3):
            fig.colorbar(cax[i, j], ax=ax[i, j])
            ax[i, j].set_xlabel('r25')
            ax[i, j].set_ylabel('r25')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_Fitting_result_maps.png')
    fig.clf()
    del T, T_err, Beta, Beta_err
    # Best fit models versus observed SEDs
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    cax = np.empty_like(ax)
    titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']
    for i in range(5):
        p, q = i // 3, i % 3
        cax[p, q] = ax[p, q].imshow(Model_d_SED[i], origin='lower',
                                    cmap=cmap2, vmin=0, vmax=2, extent=extent)
        fig.colorbar(cax[p, q], ax=ax[p, q])
        ax[p, q].set_xlabel('r25')
        ax[p, q].set_ylabel('r25')
        ax[p, q].set_title(titles[i], size=20)
    fig.tight_layout()
    fig.savefig('output/' + name + '_Fitting_Divided_by_Observed.png')
    del titles, Model_d_SED

    """ DGR starts... """
    print('Plotting DGR map...\n')
    # Total gas & DGR
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    cax = np.empty_like(ax)
    fig.suptitle(name, size=28, y=0.995)
    cax[0, 0] = ax[0, 0].imshow(DGR, origin='lower', cmap=cmap0,
                                extent=extent, norm=LogNorm())
    ax[0, 0].set_title('DGR', size=20)
    cax[0, 1] = ax[0, 1].imshow(SigmaGas, origin='lower', cmap=cmap0,
                                extent=extent, norm=LogNorm())
    ax[0, 1].set_title(r'$\Sigma_{Gas}$ $(\log_{10}(M_\odot pc^{-2}))$',
                       size=20)
    cax[0, 2] = ax[0, 2].imshow(H2, origin='lower', cmap=cmap0,
                                extent=extent, norm=LogNorm())
    ax[0, 2].set_title(r'HERACLES (log, not binned)', size=20)
    cax[1, 0] = ax[1, 0].imshow(SigmaD, origin='lower', cmap=cmap0,
                                extent=extent, norm=LogNorm())
    ax[1, 0].set_title(r'$\Sigma_D$ $(\log_{10}(M_\odot pc^{-2}))$', size=20)
    cax[1, 1] = ax[1, 1].imshow(SigmaD_dexerr, origin='lower', cmap=cmap0,
                                extent=extent)
    ax[1, 1].set_title(r'$\Sigma_D$ error (dex)', size=20)
    cax[1, 2] = ax[1, 2].imshow(Chi2, origin='lower', cmap=cmap0,
                                extent=extent)
    ax[1, 2].set_title(r'$\chi^2$', size=20)
    for i in range(2):
        for j in range(3):
            ax[i, j].set_xlabel('r25')
            ax[i, j].set_ylabel('r25')
            fig.colorbar(cax[i, j], ax=ax[i, j])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    fig.savefig('output/' + name + '_DGR_SigmaD_GAS.png')
    del SigmaD, SigmaD_dexerr, Chi2
    # Overlay DGR & H2
    plt.figure()
    plt.imshow(DGR, alpha=0.8, origin='lower', cmap=cmap0,
               extent=extent, norm=LogNorm())
    plt.colorbar()
    plt.imshow(H2, alpha=0.6, origin='lower', cmap='bone',
               extent=extent, norm=LogNorm())
    plt.xlabel('r25')
    plt.ylabel('r25')
    plt.colorbar()
    plt.title(r'DGR overlay with $H_2$ map', size=24)
    plt.savefig('output/' + name + '_DGR_overlay_H2.png')
    del H2

    """ Gas Weighted PDF profiles """
    l = len(binlist)
    print('Generating Gas-weighted SigmaD PDFs...')
    tic = clock()
    r = d = w = np.array([])  # radius, dgr, weight
    for i in range(l):
        temp_G = aGas[i] * np.sum(binmap == binlist[i])
        temp_R = aRadius[i]
        mask = aPDFs[i] > aPDFs[i].max() / 1000
        temp_DGR, temp_P = SigmaDs[mask] / temp_G, aPDFs[i][mask]
        temp_P = temp_P / np.sum(temp_P) * temp_G
        r = np.append(r, [temp_R] * len(temp_P))
        d = np.append(d, temp_DGR)
        w = np.append(w, temp_P)
        if i % (l // 10) == 0:
            print(' --computing bin:', i, 'of', l)
    nanmask = np.isnan(r + d + w)
    r, d, w = r[~nanmask], d[~nanmask], w[~nanmask]
    rbins = np.linspace(np.min(r), np.max(r), rbin)
    dbins = np.logspace(np.min(np.log10(d)), np.max(np.log10(d)), dbin)
    print(' --Counting hist2d...')
    counts, _, _ = np.histogram2d(r, d, bins=(rbins, dbins), weights=w)
    counts2, _, _ = np.histogram2d(r, d, bins=(rbins, dbins))
    del r, d, w
    counts, counts2 = counts.T, counts2.T
    dbins2, rbins2 = (dbins[:-1] + dbins[1:]) / 2, (rbins[:-1] + rbins[1:]) / 2
    DGR_Median = np.array([])
    n_zeromask = np.full(counts.shape[1], True, dtype=bool)
    print(' --Plotting PDFs at each radial bin...')
    for i in range(counts.shape[1]):
        if np.sum(counts2[:, i]) > 0:
            counts[:, i] /= np.sum(counts[:, i])
            counts2[:, i] /= np.sum(counts2[:, i])
            mask = counts[:, i] > (counts[:, i].max() / 1000)
            dmax = counts[mask, i].max()
            dmin = counts[mask, i].min()
            csp = np.cumsum(counts[:, i])[:-1]
            csp = np.append(0, csp / csp[-1])
            ssd = np.interp([0.16, 0.5, 0.84], csp, dbins2)
            DGR_Median = np.append(DGR_Median, ssd[1])
            fig, ax = plt.subplots(2, 1)
            ax[0].semilogx([ssd[0]] * len(counts[:, i]), counts[:, i],
                           label='16')
            ax[0].semilogx([ssd[1]] * len(counts[:, i]), counts[:, i],
                           label='50')
            ax[0].semilogx([ssd[2]] * len(counts[:, i]), counts[:, i],
                           label='84')
            ax[0].semilogx(dbins2, counts[:, i], label='Gas-weighted PDF')
            ax[0].set_xlim([dmin, dmax])
            ax[0].legend()
            ax[1].semilogx(dbins2, counts2[:, i], label='Non-weighted PDF')
            ax[1].set_xlim([dmin, dmax])
            ax[1].legend()
            fig.suptitle(str(round(np.log10(ssd[0]), 2)) + '; ' +
                         str(round(np.log10(ssd[1]), 2)) + '; ' +
                         str(round(np.log10(ssd[2]), 2)))
            fig.savefig('output/' + name + '_Radial_No.' + str(i) + '_R=' +
                        str(round(rbins2[i], 2)) + '.png')
            plt.close('all')
        else:
            n_zeromask[i] = False
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
    #
    c_median = 'c'
    #
    plt.figure(figsize=(10, 7.5))
    plt.pcolormesh(rbins2, dbins2, counts, norm=LogNorm(), cmap=cmap1,
                   vmin=1E-3)
    plt.yscale('log')
    plt.colorbar()
    plt.plot(rbins2[n_zeromask], DGR_Median, c_median, label='Median')
    plt.ylim([1E-6, 1E-1])
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'DGR', size=16)
    plt.title('Gas mass weighted median DGR', size=20)
    plt.savefig('output/' + name + '_DGR_PDF.png')
    # hist2d with metallicity
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.pcolor(rbins2, dbins2 / (0.0091 / 1.36), counts, norm=LogNorm(),
              cmap=cmap1, vmin=1E-3)
    ax.set_ylim([1E-4, 1E1])
    ax.set_yscale('log')
    ax.plot(rbins2[n_zeromask], DGR_Median / (0.0091 / 1.36), 'g',
            label='DGR Median / 0.0067')
    ax.set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax.set_title('Gas mass weighted median DGR vs. Metallicity', size=20)
    ax.set_ylabel('Ratio', size=16)
    ax.plot(rbins2,
            10**(8.715 - 0.027 * rbins2 * R25 - solar_oxygen_bundance), 'b',
            label='$(O/H) / (O/H)_\odot$')
    ax.legend()
    fig.savefig('output/' + name + '_DGR_PDF_and_Metallicity_Gradient.png')
    del counts, counts2, rbins, dbins, rbins2, dbins2, DGR_Median

    print('Generating Gas-weighted Temperature PDFs...')
    tic = clock()
    r = t = w = np.array([])  # radius, dgr, weight
    for i in range(l):
        temp_G = aGas[i] * np.sum(binmap == binlist[i])
        temp_R = aRadius[i]
        mask = a_T_PDFs[i] > a_T_PDFs[i].max() / 1000
        temp_T, temp_P = Ts[mask], a_T_PDFs[i][mask]
        temp_P = temp_P / np.sum(temp_P) * temp_G
        for j in range(len(temp_T)):
            r = np.append(r, temp_R)
            t = np.append(t, temp_T[j])
            w = np.append(w, temp_P[j])
        if i % (l // 10) == 0:
            print(' --computing bin:', i, 'of', l)
    nanmask = np.isnan(r + t + w)
    r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
    rbins = np.linspace(np.min(r), np.max(r), rbin)
    tbins = np.logspace(np.min(np.log10(t)), np.max(np.log10(t)), tbin)
    print(' --Counting hist2d...')
    counts, _, _ = np.histogram2d(r, t, bins=(rbins, tbins), weights=w)
    del r, t, w
    counts = counts.T
    tbins2, rbins2 = (tbins[:-1] + tbins[1:]) / 2, (rbins[:-1] + rbins[1:]) / 2
    T_Median = np.array([])
    n_zeromask = np.full(counts.shape[1], True, dtype=bool)
    print(' --Plotting PDFs at each radial bin...')
    for i in range(counts.shape[1]):
        if np.sum(counts[:, i]) > 0:
            counts[:, i] /= np.sum(counts[:, i])
            mask = counts[:, i] > (counts[:, i].max() / 1000)
            csp = np.cumsum(counts[:, i])[:-1]
            csp = np.append(0, csp / csp[-1])
            sst = np.interp([0.16, 0.5, 0.84], csp, tbins2)
            T_Median = np.append(T_Median, sst[1])
        else:
            n_zeromask[i] = False
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
    #
    plt.figure(figsize=(10, 7.5))
    plt.pcolormesh(rbins2, tbins2, counts, norm=LogNorm(), cmap=cmap1,
                   vmin=1E-3)
    plt.colorbar()
    plt.plot(rbins2[n_zeromask], T_Median, c_median, label='Median')
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'Temperature', size=16)
    plt.title('Gas mass weighted median Temperature', size=20)
    plt.savefig('output/' + name + '_T_PDF.png')
    plt.close('all')
    del counts, rbins, tbins, tbins2
    # Temperature, SFR, and SMSD
    R_SFR, SFR_profile = simple_profile(SFR, Radius, 100, SigmaGas)
    R_SMSD, SMSD_profile = simple_profile(SMSD, Radius, 100, SigmaGas)
    fig, ax = plt.subplots(2, figsize=(12, 12))
    ax[0].plot(rbins2[n_zeromask], T_Median, c_median, label='Temperature')
    ax[0].legend()
    ax[0].set_ylabel(r'Temperature ($K$)')
    ax[0].set_xlim([0, rbins2[n_zeromask].max()])
    ax[1].semilogy(R_SFR, SFR_profile, 'r')
    ax[1].set_xlabel(r'Radius ($R25$)', size=20)
    ax[1].set_ylabel(r'$\Sigma_{SFR}$ ($M_\odot kpc^{-2} yr^{-1}$)',
                     size=20, color='r')
    ax[1].tick_params('y', colors='r')
    ax2 = ax[1].twinx()
    ax2.semilogy(R_SMSD, SMSD_profile, c='b')
    ax2.set_ylabel(r'$\Sigma_*$ ($M_\odot pc^{-2}$)', size=20, color='b')
    ax2.tick_params('y', colors='b')
    ax2.set_xlim([0, rbins2[n_zeromask].max()])
    fig.tight_layout()
    fig.savefig('output/' + name + '_T_and_SFR_and_SMSD.png')
    plt.close('all')


# Might Need updates
def plot_single_pixel(name='NGC5457', cmap0='gist_heat'):
    plt.close('all')
    plt.ioff()
    with File('output/' + name + '_dust_data.h5', 'r') as hf:
        ased = np.array(hf['Herschel_SED'])
        acov_n1 = np.array(hf['Herschel_covariance_matrix'])
        binmap = np.array(hf['Binmap'])
        binlist = np.array(hf['Binlist'])
        logsigmas = np.array(hf['logsigmas'])
        apdfs = np.array(hf['PDF'])
        amodel = np.array(hf['Best_fit_model'])
        Ts_f = np.array(hf['Fixed_temperatures'])
        f_amodel = np.array(hf['Fixed_best_fit_model'])

    if Ts_f.ndim == 2:
        Ts_f = Ts_f[:, 0]

    print("Input x coordinate:")
    x = int(input())
    print("Input y coordinate:")
    y = int(input())
    for i in range(len(binlist)):
        mask = binmap == binlist[i]
        if mask[y, x]:
            pn = name + '_bin-' + str(i) + '_x-' + str(x) + '_y-' + str(y)
            plt.imshow(mask, origin='lower')
            plt.savefig('output/' + pn)
            diff = (ased[i] - amodel[i]).reshape(-1, 5)
            chi2 = round(np.dot(np.dot(diff, acov_n1[i]), diff.T)[0, 0], 3)
            # SED vs. fitting
            fig, ax = plt.subplots()
            ax.plot(wl, ased[i], label='Observed')
            ax.plot(wl, amodel[i], label=r'Model, $\chi^2=$' +
                    str(chi2))
            for j in range(len(Ts_f)):
                diff = (ased[i] - f_amodel[j][i]).reshape(-1, 5)
                chi2 = round(np.dot(np.dot(diff, acov_n1[i]), diff.T)[0, 0], 3)
                ax.plot(wl, f_amodel[j][i], label=r'$T=$' +
                        str(int(Ts_f[j])) + r'Model, $\chi^2=$' + str(chi2))
            ax.legend()
            fig.savefig('output/' + pn + '_model_fitting.png')
            # PDF
            pdf = apdfs[i]
            mask = pdf > pdf.max() / 1000
            pdf, logsigmas = pdf[mask], logsigmas[mask]
            fig, ax = plt.subplots()
            ax.plot(logsigmas, pdf)
            ax.set_xlabel(r'$\Sigma_d$')
            ax.set_ylabel('PDF')
            fig.savefig('output/' + pn + '_PDF.png')
            plt.close("all")
