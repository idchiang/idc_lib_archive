from h5py import File
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from time import clock
import astropy.units as u
from astropy.constants import c, h, k_B
from .gal_data import gal_data


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


def BPL_DGR(x, XCO):
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


def read_dust_file(name='NGC5457', rbin=51, dbin=1000, tbin=30, SigmaDoff=2.,
                   Toff=20, dr25=0.025, fixed_beta=True, fixed_T=False,
                   cmap0='gist_heat', cmap1='Greys', cmap2='seismic'):
    plt.close('all')
    plt.ioff()

    print('Loading data...\n')
    if fixed_beta:
        if fixed_T:
            fn = 'output/Dust_data_fbft_' + name + '.h5'
        else:
            fn = 'output/Dust_data_fb_' + name + '.h5'
    else:
        fn = 'output/Dust_data_' + name + '.h5'

    with File(fn, 'r') as hf:
        aSigmaD = 10**np.array(hf['Dust_surface_density_log'])
        aSigmaD_dexerr = np.array(hf['Dust_surface_density_err_dex'])  # in dex
        aT = np.array(hf['Dust_temperature'])
        aT_err = np.array(hf['Dust_temperature_err'])
        aBeta = np.array(hf['beta'])
        aBeta_err = np.array(hf['beta_err'])
        SigmaDs = 10**np.array(hf['logsigmas'])
        aPDFs = np.array(hf['PDF'])
        if not fixed_T:
            Ts = np.array(hf['Ts'])
            a_T_PDFs = np.array(hf['PDF_T'])
        aModel = np.array(hf['Best_fit_model'])
    with File('output/Voronoi_data.h5', 'r') as hf:
        grp = hf[name]
        binlist = np.array(grp['BINLIST'])
        binmap = np.array(grp['BINMAP'])
        aGas = np.array(grp['GAS_AVG'])
        aSED = np.array(grp['Herschel_SED'])
        aRadius = np.array(grp['Radius_avg'])
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        R25 = float(np.array(grp['R25_KPC']))
        H2 = map2bin(np.array(grp['HERACLES']), binlist, binmap)
        ubRadius = np.array(grp['RADIUS_KPC']) / R25
        aRadius /= R25
        SFR = map2bin(np.array(grp['SFR']), binlist, binmap)
        SMSD = map2bin(np.array(grp['SMSD']), binlist, binmap)

    # Filtering bad fits
    bad = (aSigmaD_dexerr > SigmaDoff) + (aT_err > Toff)
    aSigmaD[bad] = aSigmaD_dexerr[bad] = aT[bad] = aT_err[bad] = aBeta[bad] = \
        aBeta_err[bad] = np.nan
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
    Radius = list2bin(aRadius, binlist, binmap)
    with np.errstate(invalid='ignore'):
        DGR = SigmaD / SigmaGas
        Model_d_SED = np.array([list2bin(aModel[:, i] / aSED[:, i],
                                         binlist, binmap) for i in range(5)])
    del aSigmaD, aSigmaD_dexerr, aT, aT_err, aBeta, aBeta_err, aSED, aModel

    """ Croxall metallicity """
    print('Plotting metallicity...\n')
    # Plot points with metallicity measurements on H2 map
    plt.close('all')
    mtl = pd.read_csv('output/Metal_' + name + '.csv')
    ml = len(mtl)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(DGR, origin='lower', cmap=cmap0, norm=LogNorm())
    ax.scatter(mtl.new_c1, mtl.new_c2, s=100, marker='s', facecolors='none',
               edgecolors='c')
    ax.set_xlabel('Pixel', size=16)
    ax.set_ylabel('Pixel', size=16)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    with PdfPages('output/' + name + '_Metallicity_on_DGR.pdf') as pp:
        pp.savefig(fig)
    # Grab data for plotting
    DGR_at_ROA = Radius_at_ROA = Rel_Oxygen_Abd = np.array([])
    for i in range(ml):
        j, k = int(mtl['new_c2'].iloc[i]), int(mtl['new_c1'].iloc[i])
        if j < 0 or j >= DGR.shape[0] or k < 0 or k >= DGR.shape[1]:
            pass
        else:
            DGR_at_ROA = np.append(DGR_at_ROA, DGR[j, k])
            Radius_at_ROA = np.append(Radius_at_ROA, ubRadius[j, k])
            Rel_Oxygen_Abd = np.append(Rel_Oxygen_Abd,
                                       10**(mtl.iloc[i]['12+log(O/H)'] -
                                            solar_oxygen_bundance))
    DGR_d_91 = DGR_at_ROA * 150
    # Plot DGR vs. metallicity
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.scatter(Rel_Oxygen_Abd, DGR_d_91, c='r', s=15, label='Data points')
    ax.plot(Rel_Oxygen_Abd, Rel_Oxygen_Abd, label='x=y')
    bdr = [min(np.nanmin(DGR_d_91), np.nanmin(Rel_Oxygen_Abd)),
           max(np.nanmax(DGR_d_91), np.nanmax(Rel_Oxygen_Abd))]
    ax.plot([10**(8.0 - solar_oxygen_bundance)] * 2, bdr, '-k', alpha=0.5,
            label='12+log(O/H) = 8.0')
    ax.plot([10**(8.2 - solar_oxygen_bundance)] * 2, bdr, '-g', alpha=0.5,
            label='12+log(O/H) = 8.2')
    ax.set_xlabel(r'$(O/H)/(O/H)_\odot$', size=20)
    ax.set_ylabel(r'$DGR * 150$', size=20)
    ax.legend(fontsize=14)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    with PdfPages('output/' + name + '_DGR_vs_Metallicity.pdf') as pp:
        pp.savefig(fig)
    # vs. Remy-Ruyer
    df = pd.read_csv("data/Tables/Remy-Ruyer_2014.csv")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_yscale('log')
    ax.set_xscale('log')
    x_ = np.linspace(np.min(df['12+log(O/H)'].values - solar_oxygen_bundance),
                     np.max(df['12+log(O/H)'].values - solar_oxygen_bundance),
                     100)
    z1 = np.linspace(1.81 * np.exp(-18 / 19), 1.81 * np.exp(-8 / 19), 50)
    ax.plot(10**(x_), 150 * 10**(1.62 * x_ - 2.21), 'k--', alpha=0.6,
            label='R14 power law')
    ax.plot(10**(x_), 150 * BPL_DGR(x_, 'MW'), 'k:', alpha=0.6,
            label='R14 broken power law')
    ax.plot(10**(x_), 10**(x_), 'k', alpha=0.6, label='D14 power law')
    ax.plot(z1, z1, 'c', label='ZB12 range', linewidth=3.0)
    ax.scatter(Rel_Oxygen_Abd, DGR_d_91, c='r', s=15, label='This work')
    ax.scatter(10**(df['12+log(O/H)'] - solar_oxygen_bundance),
               df['DGR_MW'] * 150, c='b', s=15, label='R14 (MW)')
    ax.set_xlabel(r'$(O/H)/(O/H)_\odot$', size=20)
    ax.set_ylabel(r'$DGR * 150$', size=20)
    ax.legend(fontsize=14)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    with PdfPages('output/' + name + '_vs_Remy-Ruyer_MW.pdf') as pp:
        pp.savefig(fig)
    fig.savefig('output/' + name + '_vs_Remy-Ruyer_MW.png')
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot(10**(x_), 150 * 10**(2.02 * x_ - 2.21), 'k--', alpha=0.6,
            label='R14 power law')
    ax.plot(10**(x_), 150 * BPL_DGR(x_, 'Z'), 'k:', alpha=0.6,
            label='R14 broken power law')
    ax.plot(10**(x_), 10**(x_), 'k', alpha=0.6, label='D14 power law')
    ax.plot(z1, z1, 'c', label='ZB12 range', linewidth=3.0)
    ax.scatter(Rel_Oxygen_Abd, DGR_d_91, c='r', s=15, label='This work')
    ax.scatter(10**(df['12+log(O/H)'] - solar_oxygen_bundance),
               df['DGR_Z'] * 150, c='b', s=15, label='R14 (Z)')
    ax.set_xlabel(r'$(O/H)/(O/H)_\odot$', size=20)
    ax.set_ylabel(r'$DGR * 150$', size=20)
    ax.legend(fontsize=14)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    with PdfPages('output/' + name + '_vs_Remy-Ruyer_Z.pdf') as pp:
        pp.savefig(fig)
    fig.savefig('output/' + name + '_vs_Remy-Ruyer_Z.png')
    # del mtl, l, DGR_at_ROA, DGR_d_91, Radius_at_ROA, Rel_Oxygen_Abd, df

    """ Plot fitting results """
    print('Plotting fitting results...\n')
    # Fitting results
    fig, ax = plt.subplots(2, 3 - fixed_beta,
                           figsize=(6 * (3 - fixed_beta), 11))
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
    if not fixed_beta:
        ax[0, 2].set_title(r'$\beta$', size=20)
        cax[0, 2] = ax[0, 2].imshow(Beta, origin='lower', cmap=cmap0,
                                    extent=extent)
        ax[1, 2].set_title(r'$\beta$ error', size=20)
        cax[1, 2] = ax[1, 2].imshow(Beta_err, origin='lower', cmap=cmap0,
                                    extent=extent)
    for i in range(2):
        for j in range(3 - fixed_beta):
            fig.colorbar(cax[i, j], ax=ax[i, j])
            ax[i, j].set_xlabel('r25', size=16)
            ax[i, j].set_ylabel('r25', size=16)
            ax[i, j].set_xticklabels(ax[i, j].get_xticks(), fontsize=16)
            ax[i, j].set_yticklabels(ax[i, j].get_yticks(), fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    with PdfPages('output/' + name + '_Fitting_result_maps.pdf') as pp:
        pp.savefig(fig)
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
    with PdfPages('output/' + name + '_Fitting_Divided_by_Observed.pdf') as pp:
        pp.savefig(fig)
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
    ax[0, 2].set_title(r'HERACLES', size=20)
    cax[1, 0] = ax[1, 0].imshow(SigmaD, origin='lower', cmap=cmap0,
                                extent=extent, norm=LogNorm())
    ax[1, 0].set_title(r'$\Sigma_D$ $(\log_{10}(M_\odot pc^{-2}))$', size=20)
    cax[1, 1] = ax[1, 1].imshow(SigmaD_dexerr, origin='lower', cmap=cmap0,
                                extent=extent)
    ax[1, 1].set_title(r'$\Sigma_D$ error (dex)', size=20)
    for i in range(5):
        p, q = i // 3, i % 3
        ax[p, q].set_xlabel('r25', size=16)
        ax[p, q].set_ylabel('r25', size=16)
        ax[p, q].set_xticklabels(ax[p, q].get_xticks(), fontsize=16)
        ax[p, q].set_yticklabels(ax[p, q].get_yticks(), fontsize=16)
        fig.colorbar(cax[p, q], ax=ax[p, q])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    with PdfPages('output/' + name + '_DGR_SigmaD_GAS.pdf') as pp:
        pp.savefig(fig)
    del SigmaD, SigmaD_dexerr
    # Overlay DGR & H2
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(DGR, alpha=0.8, origin='lower', cmap=cmap0,
              extent=extent, norm=LogNorm())
    ax.imshow(H2, alpha=0.6, origin='lower', cmap='bone',
              extent=extent, norm=LogNorm())
    ax.set_xlabel('r25', size=16)
    ax.set_ylabel('r25', size=16)
    ax.set_xticklabels(ax.get_xticks(), fontsize=16)
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)
    plt.title(r'DGR (Orange) and $H_2$ (Gray)', size=20)
    with PdfPages('output/' + name + '_DGR_overlay_H2.pdf') as pp:
        pp.savefig(fig)
    del H2

    """ Gas Weighted PDF profiles """
    bl = len(binlist)
    print('Generating Gas-weighted SigmaD PDFs...')
    tic = clock()
    r = d = w = np.array([])  # radius, dgr, weight
    for i in range(bl):
        temp_G = aGas[i]
        temp_R = aRadius[i]
        mask = aPDFs[i] > aPDFs[i].max() / 1000
        temp_DGR, temp_P = SigmaDs[mask] / temp_G, aPDFs[i][mask]
        temp_P = temp_P / np.sum(temp_P) * temp_G * \
            (binmap == binlist[i]).sum()
        r = np.append(r, [temp_R] * len(temp_P))
        d = np.append(d, temp_DGR)
        w = np.append(w, temp_P)
        if i % (bl // 10) == 0:
            print(' --computing bin:', i, 'of', bl)
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
    DGR_Median = DGR_LExp = DGR_Max = np.array([])
    n_zeromask = np.full(counts.shape[1], True, dtype=bool)
    print(' --Plotting PDFs at each radial bin...')
    pp = PdfPages('output/' + name + '_Radial_PDFs.pdf')
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
            fig, ax = plt.subplots(2, 1)
            ax[0].semilogx([ssd[0]] * len(counts[:, i]), counts[:, i],
                           label='16%')
            ax[0].semilogx([ssd[1]] * len(counts[:, i]), counts[:, i],
                           label='Median')
            ax[0].semilogx([ssd[2]] * len(counts[:, i]), counts[:, i],
                           label='84%')
            ax[0].semilogx([DGR_LExp[-1]] * len(counts[:, i]), counts[:, i],
                           label='Expectation')
            ax[0].semilogx([DGR_Max[-1]] * len(counts[:, i]), counts[:, i],
                           label='Max likelihood')
            ax[0].semilogx(dbins2, counts[:, i], label='Gas-weighted PDF')
            # ax[0].set_xlim([dmin, dmax])
            ax[0].legend()
            ax[1].semilogx(dbins2, counts2[:, i], label='Non-weighted PDF')
            # ax[1].set_xlim([dmin, dmax])
            ax[1].legend()
            fig.suptitle('Radial No.' + str(i) + ', R=' +
                         str(round(rbins2[i], 2)))
            pp.savefig(fig)
            plt.close('all')
        else:
            n_zeromask[i] = False
    pp.close()
    plt.close('all')
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
    #
    fig = plt.figure(figsize=(10, 7.5))
    plt.pcolormesh(rbins2, dbins2, counts, norm=LogNorm(), cmap=cmap1,
                   vmin=1E-3)
    plt.yscale('log')
    plt.colorbar()
    plt.plot(rbins2[n_zeromask], DGR_Median, 'r', label='Median')
    plt.plot(rbins2[n_zeromask], DGR_LExp, 'g', label='Log Expectation')
    plt.plot(rbins2[n_zeromask], DGR_Max, 'b', label='Max likelihhod')
    plt.ylim([1E-6, 1E-1])
    plt.xlabel(r'Radius ($R_{25}$)', size=16)
    plt.ylabel(r'DGR', size=16)
    plt.legend(fontsize=16)
    plt.title('Gas mass weighted median DGR', size=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    with PdfPages('output/' + name + '_DGR_PDF.pdf') as pp:
        pp.savefig(fig)
    # hist2d with metallicity
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.pcolor(rbins2, dbins2 * 150, counts, norm=LogNorm(),
              cmap=cmap1, vmin=1E-3)
    ax.set_ylim([1E-4, 1E1])
    ax.set_yscale('log')
    ax.plot(rbins2[n_zeromask], DGR_Median * 150, 'r', label='DGR Median')
    ax.plot(rbins2[n_zeromask], DGR_LExp * 150, 'g',
            label='DGR Log Expectation')
    ax.plot(rbins2[n_zeromask], DGR_Max * 150, 'b', label='DGR Max Likelihood')
    ax.set_xlabel(r'Radius ($R_{25}$)', size=16)
    ax.set_title('DGR * 150 vs. Metallicity', size=20)
    ax.set_ylabel('Ratio', size=16)
    # Fixing the distance difference between gal_data & Croxall
    GD_dist = gal_data(name).field('DIST_MPC')[0]
    ax.plot(rbins2,
            10**(8.715 - 0.027 * rbins2 * R25 * 7.4 / GD_dist -
                 solar_oxygen_bundance),
            'm', label='$(O/H) / (O/H)_\odot$')
    ax.legend(fontsize=16)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    with PdfPages('output/' + name + '_DGR_and_Metallicity_Grad.pdf') as pp:
        pp.savefig(fig)
    # My DGR gradient with Remy-Ruyer data and various models
    fig, ax = plt.subplots(figsize=(10, 7.5))
    xbins2 = (8.715 - 0.027 * rbins2 * R25 * 7.4 / GD_dist)
    ax.pcolor(xbins2, dbins2, counts, norm=LogNorm(),
              cmap='Reds', vmin=1E-3)
    ax.set_ylim([1E-5, 1E0])
    ax.set_yscale('log')
    ax.plot(xbins2[n_zeromask], DGR_LExp, 'g',
            label='This work (Log Expectation)')
    ax.set_xlabel('12 + log(O/H)', size=16)
    ax.set_ylabel('DGR', size=16)
    r_ = (8.715 - df['12+log(O/H)'].values) / 0.027 * GD_dist / 7.4 / R25
    r__ = np.linspace(np.nanmin(r_), np.nanmax(r_), 50)
    x__ = (8.715 - 0.027 * r__ * R25 * 7.4 / GD_dist - solar_oxygen_bundance)
    ax.plot(x__ + solar_oxygen_bundance, 10**(1.62 * x__ - 2.21),
            'k--', alpha=0.6, label='R14 power law')
    ax.plot(x__ + solar_oxygen_bundance, BPL_DGR(x__, 'MW'), 'k:', alpha=0.6,
            label='R14 broken power law')
    ax.plot(x__ + solar_oxygen_bundance, 10**(x__) / 150, 'k', alpha=0.6,
            label='D14 power law')
    ax.scatter(df['12+log(O/H)'], df['DGR_MW'], c='b', s=15,
               label='R14 data (MW)')
    zl = np.log10(1.81 * np.exp(-18 / 19))
    zu = np.log10(1.81 * np.exp(-8 / 19))
    z_ = np.linspace(zl, zu, 50)
    ax.plot(z_ + solar_oxygen_bundance, 10**z_ / 150, 'c', label='ZB12 range',
            linewidth=3.0)
    ax.legend(fontsize=16)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    with PdfPages('output/' + name + '_DGR_and_Metallicity_Models.pdf') as pp:
        pp.savefig(fig)
    del counts, counts2, rbins, dbins, rbins2, dbins2, DGR_Median

    if not fixed_T:
        print('Generating Gas-weighted Temperature PDFs...')
        tic = clock()
        r = t = w = np.array([])  # radius, dgr, weight
        for i in range(bl):
            temp_G = aGas[i]
            temp_R = aRadius[i]
            mask = a_T_PDFs[i] > a_T_PDFs[i].max() / 1000
            temp_T, temp_P = Ts[mask], a_T_PDFs[i][mask]
            temp_P = temp_P / np.sum(temp_P) * temp_G * \
                (binmap == binlist[i]).sum()
            for j in range(len(temp_T)):
                r = np.append(r, temp_R)
                t = np.append(t, temp_T[j])
                w = np.append(w, temp_P[j])
            if i % (bl // 10) == 0:
                print(' --computing bin:', i, 'of', bl)
        nanmask = np.isnan(r + t + w)
        r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
        rbins = np.linspace(np.min(r), np.max(r), rbin)
        tbins = np.logspace(np.min(np.log10(t)), np.max(np.log10(t)), tbin)
        print(' --Counting hist2d...')
        counts, _, _ = np.histogram2d(r, t, bins=(rbins, tbins), weights=w)
        del r, t, w
        counts = counts.T
        tbins2 = (tbins[:-1] + tbins[1:]) / 2
        rbins2 = (rbins[:-1] + rbins[1:]) / 2
        T_Median = T_Exp = T_Max = np.array([])
        n_zeromask = np.full(counts.shape[1], True, dtype=bool)
        print(' --Calculating PDFs at each radial bin...')
        for i in range(counts.shape[1]):
            if np.sum(counts[:, i]) > 0:
                counts[:, i] /= np.sum(counts[:, i])
                mask = counts[:, i] > (counts[:, i].max() / 1000)
                csp = np.cumsum(counts[:, i])[:-1]
                csp = np.append(0, csp / csp[-1])
                sst = np.interp([0.16, 0.5, 0.84], csp, tbins2)
                T_Median = np.append(T_Median, sst[1])
                T_Exp = np.append(T_Exp, np.sum(counts[:, i] * tbins2))
                T_Max = np.append(T_Max, tbins2[np.argmax(counts[:, i])])
            else:
                n_zeromask[i] = False
        print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
        #
        fig, ax = plt.subplots(figsize=(10, 7.5))
        cax = ax.pcolormesh(rbins2, tbins2, counts, norm=LogNorm(), cmap=cmap1,
                            vmin=1E-3)
        plt.colorbar(cax, ax=ax)
        ax.plot(rbins2[n_zeromask], T_Median, 'r', label='Median')
        ax.plot(rbins2[n_zeromask], T_Exp, 'g', label='Expectation')
        ax.plot(rbins2[n_zeromask], T_Max, 'b', label='Max Likelihood')
        ax.set_xlabel(r'Radius ($R_{25}$)', size=16)
        ax.set_ylabel(r'Temperature', size=16)
        ax.set_xticklabels(ax.get_xticks(), fontsize=12)
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)
        ax.legend(fontsize=14)
        with PdfPages('output/' + name + '_T_PDF.pdf') as pp:
            pp.savefig(fig)
        del counts, rbins, tbins, tbins2
        # Temperature, SFR, and SMSD
        R_SFR, SFR_profile = simple_profile(SFR, Radius, 100, SigmaGas)
        R_SMSD, SMSD_profile = simple_profile(SMSD, Radius, 100, SigmaGas)
        fig, ax = plt.subplots(2, figsize=(12, 12))
        ax[0].plot(rbins2[n_zeromask], T_Median, 'r', label='Median')
        ax[0].plot(rbins2[n_zeromask], T_Exp, 'g', label='Expectation')
        ax[0].plot(rbins2[n_zeromask], T_Max, 'b', label='Max Likelihood')
        ax[0].legend(fontsize=14)
        ax[0].set_ylabel(r'Temperature ($K$)', size=20)
        ax[0].set_xlim([0, rbins2[n_zeromask].max()])
        ax[0].set_xticklabels(ax[0].get_xticks(), fontsize=12)
        ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=12)
        ax[1].semilogy(R_SFR, SFR_profile, 'r')
        ax[1].set_xlabel(r'Radius ($R25$)', size=20)
        ax[1].set_ylabel(r'$\Sigma_{SFR}$ ($M_\odot kpc^{-2} yr^{-1}$)',
                         size=20, color='r')
        ax[1].tick_params('y', colors='r')
        ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=12)
        ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=12)
        ax2 = ax[1].twinx()
        ax2.semilogy(R_SMSD, SMSD_profile, c='b')
        ax2.set_ylabel(r'$\Sigma_*$ ($M_\odot pc^{-2}$)', size=20, color='b')
        ax2.tick_params('y', colors='b')
        ax2.set_xlim([0, rbins2[n_zeromask].max()])
        ax2.set_yticklabels(ax2.get_xticks(), fontsize=12)
        fig.tight_layout()
        with PdfPages('output/' + name + '_T_and_SFR_and_SMSD.pdf') as pp:
            pp.savefig(fig)
        plt.close('all')


def plots_for_paper(name='NGC5457', rbin=51, dbin=100, tbin=90, SigmaDoff=2.,
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
        aGas = np.array(grp['GAS_AVG'])
        SigmaGas = list2bin(aGas, binlist, binmap)
        aRadius = np.array(grp['Radius_avg'])
        aSED = np.array(grp['Herschel_SED'])
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        R25 = float(np.array(grp['R25_KPC']))
        aRadius /= R25
        ubRadius = np.array(grp['RADIUS_KPC']) / R25
        Radius = list2bin(aRadius, binlist, binmap)
        # H2 = map2bin(np.array(grp['HERACLES']), binlist, binmap)
        SFR = map2bin(np.array(grp['SFR']), binlist, binmap)
        SMSD = map2bin(np.array(grp['SMSD']), binlist, binmap)
        sed_unc = np.array(grp['HERSCHEL_011111_UNCMAP'])
        bkgcov = np.array(grp['HERSCHEL_011111_BKGCOV'])
    # Calculating Image scale
    extent = np.array([-ubRadius[:, 0].max(), ubRadius[:, -1].max(),
                       -ubRadius[0, :].max(), ubRadius[-1, :].max()])

    fn = 'output/Dust_data_' + name + '.h5'
    with File(fn, 'r') as hf:
        SigmaD = list2bin(10**np.array(hf['Dust_surface_density_log']),
                          binlist, binmap)
        # with np.errstate(invalid='ignore'):
        #     DGR = SigmaD / SigmaGas
        SigmaD_dexerr = \
            list2bin(np.array(hf['Dust_surface_density_err_dex']), binlist,
                     binmap)
        T = list2bin(np.array(hf['Dust_temperature']), binlist, binmap)
        T_err = list2bin(np.array(hf['Dust_temperature_err']), binlist, binmap)
        # Beta = list2bin(np.array(hf['beta']), binlist, binmap)
        # Beta_err = list2bin(np.array(hf['beta_err']), binlist, binmap)
        # SigmaDs = 10**np.array(hf['logsigmas'])
        # aPDFs = np.array(hf['PDF'])
        Ts = np.array(hf['Ts'])
        a_T_PDFs = np.array(hf['PDF_T'])
        # aModel = np.array(hf['Best_fit_model'])

    fn_fb = 'output/Dust_data_fb_' + name + '.h5'
    with File(fn_fb, 'r') as hf:
        SigmaD_fb = list2bin(10**np.array(hf['Dust_surface_density_log']),
                             binlist, binmap)
        # with np.errstate(invalid='ignore'):
        #     DGR_fb = SigmaD_fb / SigmaGas
        SigmaD_dexerr_fb = \
            list2bin(np.array(hf['Dust_surface_density_err_dex']), binlist,
                     binmap)
        T_fb = list2bin(np.array(hf['Dust_temperature']), binlist, binmap)
        T_err_fb = list2bin(np.array(hf['Dust_temperature_err']), binlist,
                            binmap)
        # Beta_fb = list2bin(np.array(hf['beta']), binlist, binmap)
        # SigmaDs_fb = 10**np.array(hf['logsigmas'])
        # aPDFs_fb = np.array(hf['PDF'])
        Ts_fb = np.array(hf['Ts'])
        a_T_PDFs_fb = np.array(hf['PDF_T'])
        # aModel = np.array(hf['Best_fit_model'])

    fn_fbft = 'output/Dust_data_fbft_' + name + '.h5'
    with File(fn_fbft, 'r') as hf:
        SigmaD_fbft = list2bin(10**np.array(hf['Dust_surface_density_log']),
                               binlist, binmap)
        # with np.errstate(invalid='ignore'):
        #     DGR_fbft = SigmaD_fbft / SigmaGas
        SigmaD_dexerr_fbft = \
            list2bin(np.array(hf['Dust_surface_density_err_dex']), binlist,
                     binmap)
        aT_fbft = np.array(hf['Dust_temperature'])
        T_fbft = list2bin(aT_fbft, binlist, binmap)
        # T_err_fbft = list2bin(np.array(hf['Dust_temperature_err']), binlist,
        #                       binmap)
        aBeta_fbft = np.array(hf['beta'])
        # Beta_fbft = list2bin(np.array(hf['beta']), binlist, binmap)
        SigmaDs_fbft = 10**np.array(hf['logsigmas'])
        aPDFs_fbft = np.array(hf['PDF'])

    """
    2X2: dust surface density & error maps for varying & fixed beta
    """
    rows, columns = 2, 2
    fig, ax = plt.subplots(rows, columns, figsize=(14, 12))
    temp = [SigmaD,
            SigmaD_dexerr,
            SigmaD_fb,
            SigmaD_dexerr_fb]
    titles = [r'Dust map ($M_\odot/pc^2$)',
              'Error map (dex)',
              r'Dust map ($\beta=2$) ($M_\odot/pc^2$)',
              r'Error map ($\beta=2$) (dex)']
    maxs = [np.nanmax(np.append(temp[0], temp[2])),
            np.nanmax(np.append(temp[1], temp[3]))]
    mins = [np.nanmin(np.append(temp[0], temp[2])),
            np.nanmin(np.append(temp[1], temp[3]))]
    for i in range(rows * columns):
        p, q = i // rows, i % columns
        cax = ax[p, q].imshow(temp[i], origin='lower', cmap=cmap0,
                              extent=extent, vmax=maxs[q], vmin=mins[q],
                              norm=LogNorm())
        ax[p, q].set_title(titles[i], size=20)
        if p == 1:
            ax[p, q].set_xlabel('r25', size=16)
            ax[p, q].set_xticklabels(ax[p, q].get_xticks(), fontsize=16)
        else:
            ax[p, q].set_xticklabels([], fontsize=16)
        if q == 0:
            ax[p, q].set_ylabel('r25', size=16)
            ax[p, q].set_yticklabels(ax[p, q].get_yticks(), fontsize=16)
        else:
            ax[p, q].set_yticklabels([], fontsize=16)
        fig.colorbar(cax, ax=ax[p, q])
    fig.tight_layout()
    with PdfPages('output/_2X2Dust.pdf') as pp:
        pp.savefig(fig)
    del temp, titles, p, q, cax, fig, ax, rows, columns, maxs, mins
    """
    2X2: dust temperature & error maps for varying & fixed beta
    """
    rows, columns = 2, 2
    fig, ax = plt.subplots(rows, columns, figsize=(14, 12))
    temp = [T,
            T_err,
            T_fb,
            T_err_fb]
    titles = [r'Temperature map ($K$)',
              'Error map',
              r'Temperature map ($\beta=2$) ($K$)',
              r'Error map ($\beta=2$)']
    maxs = [np.nanmax(np.append(temp[0], temp[2])),
            np.nanmax(np.append(temp[1], temp[3]))]
    mins = [np.nanmin(np.append(temp[0], temp[2])),
            np.nanmin(np.append(temp[1], temp[3]))]
    for i in range(rows * columns):
        p, q = i // rows, i % columns
        cax = ax[p, q].imshow(temp[i], origin='lower', cmap=cmap0,
                              extent=extent, vmax=maxs[q], vmin=mins[q])
        ax[p, q].set_title(titles[i], size=20)
        if p == 1:
            ax[p, q].set_xlabel('r25', size=16)
            ax[p, q].set_xticklabels(ax[p, q].get_xticks(), fontsize=16)
        else:
            ax[p, q].set_xticklabels([], fontsize=16)
        if q == 0:
            ax[p, q].set_ylabel('r25', size=16)
            ax[p, q].set_yticklabels(ax[p, q].get_yticks(), fontsize=16)
        else:
            ax[p, q].set_yticklabels([], fontsize=16)
        fig.colorbar(cax, ax=ax[p, q])
    fig.tight_layout()
    with PdfPages('output/_2X2Temperature.pdf') as pp:
        pp.savefig(fig)
    del temp, titles, p, q, cax, fig, ax, rows, columns, maxs, mins
    """
    3X1: Temperature gradient & Sigma_SFR + Sigma_* profile
    """
    tic = clock()
    bl = len(binlist)
    r, t, w = np.array([]), np.array([]), np.array([])
    for i in range(bl):
        temp_GM = aGas[i] * (binmap == binlist[i]).sum()
        mask = a_T_PDFs[i] > a_T_PDFs[i].max() / 1000
        temp_T, temp_P = Ts[mask], a_T_PDFs[i][mask]
        temp_P = temp_P / np.sum(temp_P) * temp_GM
        r = np.append(r, [aRadius[i]] * len(temp_T))
        for j in range(len(temp_T)):
            t = np.append(t, temp_T[j])
            w = np.append(w, temp_P[j])
        if i % (bl // 10) == 0:
            print(' --computing bin:', i, 'of', bl)
    nanmask = np.isnan(r + t + w)
    r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
    rbins = np.linspace(np.min(r), np.max(r), rbin)
    tbins = np.linspace(np.min(t), np.max(t), tbin)
    print(' --Counting hist2d...')
    counts, _, _ = np.histogram2d(r, t, bins=(rbins, tbins), weights=w)
    del r, t, w
    # Fixing temperature
    r, t, w = np.array([]), np.array([]), np.array([])
    for i in range(bl):
        temp_GM = aGas[i] * (binmap == binlist[i]).sum()
        mask = a_T_PDFs_fb[i] > a_T_PDFs_fb[i].max() / 1000
        temp_T, temp_P = Ts_fb[mask], a_T_PDFs_fb[i][mask]
        temp_P = temp_P / np.sum(temp_P) * temp_GM
        r = np.append(r, [aRadius[i]] * len(temp_T))
        for j in range(len(temp_T)):
            t = np.append(t, temp_T[j])
            w = np.append(w, temp_P[j])
        if i % (bl // 10) == 0:
            print(' --computing bin:', i, 'of', bl)
    nanmask = np.isnan(r + t + w)
    r, t, w = r[~nanmask], t[~nanmask], w[~nanmask]
    tbins_fb = np.linspace(np.min(t), np.max(t), tbin)
    print(' --Counting hist2d...')
    counts_fb, _, _ = np.histogram2d(r, t, bins=(rbins, tbins_fb), weights=w)
    del r, t, w
    counts, counts_fb = counts.T, counts_fb.T
    tbins2 = (tbins[:-1] + tbins[1:]) / 2
    tbins2_fb = (tbins[:-1] + tbins[1:]) / 2
    rbins2 = (rbins[:-1] + rbins[1:]) / 2
    T_Exp, T_Exp_fb = np.array([]), np.array([])
    n_zeromask = np.full(counts.shape[1], True, dtype=bool)
    n_zeromask_fb = np.full(counts_fb.shape[1], True, dtype=bool)
    print(' --Calculating PDFs at each radial bin...')
    assert counts.shape[1] == counts_fb.shape[1]
    for i in range(counts.shape[1]):
        if np.sum(counts[:, i]) > 0:
            counts[:, i] /= np.sum(counts[:, i])
            T_Exp = np.append(T_Exp, np.sum(counts[:, i] * tbins2))
        else:
            n_zeromask[i] = False
    for i in range(counts_fb.shape[1]):
        if np.sum(counts_fb[:, i]) > 0:
            counts_fb[:, i] /= np.sum(counts_fb[:, i])
            T_Exp_fb = np.append(T_Exp_fb, np.sum(counts_fb[:, i] * tbins2_fb))
        else:
            n_zeromask_fb[i] = False
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
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
    temp1 = [tbins2, tbins2_fb]
    temp2 = [counts, counts_fb]
    temp3 = [T_Exp, T_Exp_fb]
    temp4 = [n_zeromask, n_zeromask_fb]
    del tbins2, tbins2_fb, counts, counts_fb, T_Exp, T_Exp_fb
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
    """
    1X2: Fitted vs. predicted temperature: residue map and radial profile
    """
    rows, columns = 1, 2
    fig, ax = plt.subplots(rows, columns, figsize=(12, 6))
    titles = ['Residue map',
              'Radial profile']
    cax = ax[0].imshow(T_fbft - T_fb, origin='lower', cmap=cmap0,
                       extent=extent)
    ax[0].set_title(titles[0], size=20)
    ax[0].set_xlabel('r25', size=16)
    ax[0].set_xticklabels(ax[0].get_xticks(), fontsize=16)
    ax[0].set_ylabel('r25', size=16)
    ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=16)
    fig.colorbar(cax, ax=ax[0])
    R_Tfb, Tfb_profile = simple_profile(T_fb, Radius, 100, SigmaGas)
    R_Tfbft, Tfbft_profile = simple_profile(T_fbft, Radius, 100, SigmaGas)
    ax[1].plot(R_Tfb, Tfb_profile, label=r'$\beta=2$')
    ax[1].plot(R_Tfbft, Tfbft_profile, label=r'$\beta=2$, pred temperature')
    ax[1].set_xlabel(r'Radius ($R25$)', size=16)
    ax[1].set_ylabel(r'Temperature ($K$)', size=16)
    ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=12)
    ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=12)
    ax[1].legend(fontsize=14)
    fig.tight_layout()
    with PdfPages('output/_1X2Tempfb_fbft.pdf') as pp:
        pp.savefig(fig)
    del titles, cax, fig, ax, rows, columns, R_Tfb, Tfb_profile, R_Tfbft, \
        Tfbft_profile
    """
    1X2: dust surface density & error map for fixed beta & fixed temperature
    """
    rows, columns = 1, 2
    fig, ax = plt.subplots(rows, columns, figsize=(12, 6))
    temp = [SigmaD_fbft,
            SigmaD_dexerr_fbft]
    titles = [r'Dust map ($M_\odot/pc^2$)',
              r'Error map (dex)']
    for i in range(rows * columns):
        p, q = i // rows, i % columns
        cax = ax[p].imshow(temp[i], origin='lower', cmap=cmap0,
                           extent=extent,
                           norm=LogNorm())
        ax[p].set_title(titles[i], size=20)
        if p == 1:
            ax[p].set_xlabel('r25', size=16)
            ax[p].set_xticklabels(ax[p].get_xticks(), fontsize=16)
        if q == 0:
            ax[p].set_ylabel('r25', size=16)
            ax[p].set_yticklabels(ax[p].get_yticks(), fontsize=16)
        else:
            ax[p].set_yticklabels([], fontsize=16)
        fig.colorbar(cax, ax=ax[p])
    fig.tight_layout()
    with PdfPages('output/_1X2Dust_fbft.pdf') as pp:
        pp.savefig(fig)
    del temp, titles, p, q, cax, fig, ax, rows, columns
    """
    1X1: DGR profile fbft
    """
    print('Generating Gas-weighted SigmaD PDFs...')
    tic = clock()
    r = d = w = np.array([])  # radius, dgr, weight
    for i in range(bl):
        temp_G = aGas[i]
        temp_R = aRadius[i]
        mask = aPDFs_fbft[i] > aPDFs_fbft[i].max() / 1000
        temp_DGR, temp_P = SigmaDs_fbft[mask] / temp_G, aPDFs_fbft[i][mask]
        temp_P = temp_P / np.sum(temp_P) * temp_G * \
            (binmap == binlist[i]).sum()
        r = np.append(r, [temp_R] * len(temp_P))
        d = np.append(d, temp_DGR)
        w = np.append(w, temp_P)
        if i % (bl // 10) == 0:
            print(' --computing bin:', i, 'of', bl)
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
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
    #
    fig = plt.figure(figsize=(10, 7.5))
    plt.pcolormesh(rbins2, dbins2, counts, norm=LogNorm(), cmap=cmap1,
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
    with PdfPages('output/_1X1DGR_PDF.pdf') as pp:
        pp.savefig(fig)
    """
    1X1: DGR vs. metallicity with other papers
    """
    GD_dist = gal_data(name).field('DIST_MPC')[0]
    df = pd.read_csv("data/Tables/Remy-Ruyer_2014.csv")
    # My DGR gradient with Remy-Ruyer data and various models
    fig, ax = plt.subplots(figsize=(10, 7.5))
    xbins2 = (8.715 - 0.027 * rbins2 * R25 * 7.4 / GD_dist)
    ax.pcolor(xbins2, dbins2, counts, norm=LogNorm(),
              cmap='Reds', vmin=1E-3)
    ax.set_ylim([1E-5, 1E0])
    ax.set_yscale('log')
    ax.plot(xbins2[n_zeromask], DGR_LExp, 'g',
            label='This work (Log Expectation)')
    ax.set_xlabel('12 + log(O/H)', size=16)
    ax.set_ylabel('DGR', size=16)
    r_ = (8.715 - df['12+log(O/H)'].values) / 0.027 * GD_dist / 7.4 / R25
    r__ = np.linspace(np.nanmin(r_), np.nanmax(r_), 50)
    x__ = (8.715 - 0.027 * r__ * R25 * 7.4 / GD_dist - solar_oxygen_bundance)
    ax.plot(x__ + solar_oxygen_bundance, 10**(1.62 * x__ - 2.21),
            'k--', alpha=0.6, label='R14 power law')
    ax.plot(x__ + solar_oxygen_bundance, BPL_DGR(x__, 'MW'), 'k:', alpha=0.6,
            label='R14 broken power law')
    ax.plot(x__ + solar_oxygen_bundance, 10**(x__) / 150, 'k', alpha=0.6,
            label='D14 power law')
    ax.scatter(df['12+log(O/H)'], df['DGR_MW'], c='b', s=15,
               label='R14 data (MW)')
    zl = np.log10(1.81 * np.exp(-18 / 19))
    zu = np.log10(1.81 * np.exp(-8 / 19))
    z_ = np.linspace(zl, zu, 50)
    ax.plot(z_ + solar_oxygen_bundance, 10**z_ / 150, 'c', label='ZB12 range',
            linewidth=3.0)
    ax.legend(fontsize=16)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    with PdfPages('output/_1X1DGR_Z_Models.pdf') as pp:
        pp.savefig(fig)
    del counts, counts2, rbins, dbins, rbins2, dbins2, DGR_Median
    """
    1X1: Example PDF
    1X1: Example Model
    """
    #        pp = PdfPages('output/' + pn + '.pdf')
    x, y = 55, 55
    for i in range(len(binlist)):
        bin_ = binmap == binlist[i]
        if bin_[y, x]:
            # PDF
            pdf = aPDFs_fbft[i] / np.sum(aPDFs_fbft[i])
            csp = np.cumsum(pdf)[:-1]
            csp = np.append(0, csp / csp[-1])
            sss = np.interp([0.16, 0.5, 0.84], csp, SigmaDs_fbft).tolist()
            mask = pdf > (pdf.max() * 1.0E-5)
            pdf, SigmaDs_fbft = pdf[mask], SigmaDs_fbft[mask]
            temp_y = np.linspace(0, pdf.max(), 10)
            # LogSigmaD_Med = sss[1]
            SigmaD_Exp = 10**np.sum(pdf * np.log10(SigmaDs_fbft))
            SigmaD_Max = SigmaDs_fbft[np.argmax(pdf)]
            xticks = np.logspace(np.log10(np.nanmin(SigmaDs_fbft)),
                                 np.log10(np.nanmax(SigmaDs_fbft)), 5)
            for j in range(len(xticks)):
                xticks[j] = round(xticks[j], 2)
            """
            Begin plotting
            """
            fig, ax = plt.subplots(figsize=(10, 7.5))
            ax.semilogx([sss[0]] * 10, temp_y, alpha=0.5,
                        label='16%')
            ax.semilogx([sss[2]] * 10, temp_y, alpha=0.5,
                        label='84%')
            ax.semilogx([sss[1]] * 10, temp_y, label='Median')
            ax.semilogx([SigmaD_Exp] * 10, temp_y,
                        label='Expectation')
            ax.semilogx([SigmaD_Max] * 10, temp_y,
                        label='Maximum')
            ax.semilogx(SigmaDs_fbft, pdf, label='PDF')
            ax.set_xlabel(r'$\log_{10}\Sigma_d$', size=16)
            ax.set_ylabel('Probability', size=16)
            ax.legend(fontsize=16)
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(ax.get_xticks(), fontsize=12)
            ax.set_yticklabels(ax.get_yticks(), fontsize=12)
            with PdfPages('output/_1X1Example_PDF.pdf') as pp:
                pp.savefig(fig)
            """
            Model Calculation
            """
            ccf = np.zeros(5)
            pacs_rsrf = pd.read_csv("data/RSRF/PACS_RSRF.csv")
            pacs_wl = pacs_rsrf['Wavelength'].values
            pacs_nu = (c / pacs_wl / u.um).to(u.Hz)
            pacs100dnu = \
                pacs_rsrf['PACS_100'].values * pacs_rsrf['dnu'].values[0]
            pacs160dnu = \
                pacs_rsrf['PACS_160'].values * pacs_rsrf['dnu'].values[0]
            del pacs_rsrf
            #
            pacs_models = \
                B(aT_fbft[i] * u.K, pacs_nu) * pacs_wl**(-aBeta_fbft[i])
            del pacs_nu
            ccf[0] = np.sum(pacs100dnu * pacs_wl / wl[0]) / \
                np.sum(pacs100dnu * pacs_models /
                       (B(aT_fbft[i] * u.K, nu[0]) * wl[0]**(-aBeta_fbft[i])))
            ccf[1] = np.sum(pacs160dnu * pacs_wl / wl[1]) / \
                np.sum(pacs160dnu * pacs_models /
                       (B(aT_fbft[i] * u.K, nu[1]) * wl[1]**(-aBeta_fbft[i])))
            #
            del pacs_wl, pacs100dnu, pacs160dnu, pacs_models
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
            spire_models = \
                B(aT_fbft[i] * u.K, spire_nu) * spire_wl**(-aBeta_fbft[i])
            del spire_nu
            ccf[2] = np.sum(spire250dnu * spire_wl / wl[2]) / \
                np.sum(spire250dnu * spire_models /
                       (B(aT_fbft[i] * u.K, nu[2]) * wl[2]**(-aBeta_fbft[i])))
            ccf[3] = np.sum(spire350dnu * spire_wl / wl[3]) / \
                np.sum(spire350dnu * spire_models /
                       (B(aT_fbft[i] * u.K, nu[3]) * wl[3]**(-aBeta_fbft[i])))
            ccf[4] = np.sum(spire500dnu * spire_wl / wl[4]) / \
                np.sum(spire500dnu * spire_models /
                       (B(aT_fbft[i] * u.K, nu[4]) * wl[4]**(-aBeta_fbft[i])))
            #
            del spire_wl, spire250dnu, spire350dnu, spire500dnu
            del spire_models
            sed_plot = aSED[i] * ccf
            #
            sed_unc_plot = np.sqrt(np.nanmean(sed_unc[bin_]**2, axis=0))
            wl_plot = np.linspace(51, 549, 100)
            pdf /= np.nanmax(pdf) / 0.5
            model_exp = model(wl_plot, SigmaD_Exp, aT_fbft[i], aBeta_fbft[i],
                              (c / wl_plot / u.um).to(u.Hz))
            """
            Begin plotting
            """
            fig, ax = plt.subplots(figsize=(10, 7.5))
            ax.errorbar(wl, sed_plot, yerr=sed_unc_plot, color='red',
                        fmt='o', capsize=10, label='Herschel data')
            for j in range(len(pdf)):
                ax.plot(wl_plot,
                        model(wl_plot, SigmaDs_fbft[j], aT_fbft[i],
                              aBeta_fbft[i], (c / wl_plot / u.um).to(u.Hz)),
                        'k', alpha=pdf[j])
            ax.plot(wl_plot, model_exp, label='Expectation', linewidth=3)
            ax.legend(fontsize=20)
            ax.set_ylim([0.0, np.nanmax(model_exp) * 1.2])
            ax.set_xlabel(r'Wavelength ($\mu m$)', size=20)
            ax.set_ylabel(r'SED ($MJy$ $sr^{-1}$)', size=20)
            ax.set_xticklabels(ax.get_xticks(), fontsize=12)
            ax.set_yticklabels(ax.get_yticks(), fontsize=12)
            with PdfPages('output/_1X1Example_Model.pdf') as pp:
                pp.savefig(fig)
            break
    plt.close('all')
    """
    Print BKG Covariance Matrix
    """
    for i in range(5):
        print('   ', round(bkgcov[i, 0], 3), end='')
        for j in range(1, 5):
            print(' &', round(bkgcov[i, j], 3), end='')
        print(' \\\\')


def plot_single_pixel(name='NGC5457', cmap0='gist_heat', plot_model=True):
    plt.close('all')
    plt.ioff()
    with File('output/Dust_data_fb_' + name + '.h5', 'r') as hf:
        asopt = np.array(hf['Dust_surface_density_log'])
        logsigmas = np.array(hf['logsigmas'])
        apdfs = np.array(hf['PDF'])
        aT = np.array(hf['Dust_temperature'])
        aBeta = np.array(hf['beta'])
    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        sed = np.array(grp['HERSCHEL_011111'])
        sed_unc = np.array(grp['HERSCHEL_011111_UNCMAP'])
        ased = np.array(grp['Herschel_SED'])
        binmap = np.array(grp['Binmap'])
        binlist = np.array(grp['Binlist'])
    print("Input x coordinate:")
    x = int(input())
    print("Input y coordinate:")
    y = int(input())
    sopt = list2bin(asopt, binlist, binmap)
    for i in range(len(binlist)):
        bin_ = binmap == binlist[i]
        if bin_[y, x]:
            pn = name + '_bin-' + str(i) + '_x-' + str(x) + '_y-' + str(y)
            pp = PdfPages('output/' + pn + '.pdf')
            # location
            temp = sopt
            temp[bin_] = np.nanmax(sopt) + 1
            fig, ax = plt.subplots(figsize=(10, 7.5))
            ax.imshow(temp, origin='lower', cmap=cmap0)
            ax.set_title('Location on dust map')
            pp.savefig(fig)
            # PDF
            pdf = apdfs[i] / np.sum(apdfs[i])
            csp = np.cumsum(pdf)[:-1]
            csp = np.append(0, csp / csp[-1])
            sss = np.interp([0.16, 0.5, 0.84], csp, logsigmas).tolist()
            mask = pdf > pdf.max() / 1000
            pdf, logsigmas = pdf[mask], logsigmas[mask]
            fig, ax = plt.subplots(figsize=(10, 7.5))
            ax.plot(logsigmas, pdf, label='PDF')
            temp_y = np.linspace(0, pdf.max(), 10)
            # LogSigmaD_Med = sss[1]
            LogSigmaD_Exp = np.sum(pdf * logsigmas)
            LogSigmaD_Max = logsigmas[np.argmax(pdf)]
            ax.plot([sss[0]] * 10, temp_y, alpha=0.5, label='16%')
            ax.plot([sss[2]] * 10, temp_y, alpha=0.5, label='84%')
            ax.plot([sss[1]] * 10, temp_y, label='Median')
            ax.plot([LogSigmaD_Exp] * 10, temp_y, label='Expectation (Log)')
            ax.plot([LogSigmaD_Max] * 10, temp_y, label='Max Likelihood')
            ax.set_xlabel(r'$\Sigma_d$', size=16)
            ax.set_ylabel('PDF', size=16)
            ax.legend(fontsize=16)
            ax.set_xticklabels(ax.get_xticks(), fontsize=12)
            ax.set_yticklabels(ax.get_yticks(), fontsize=12)
            pp.savefig(fig)
            # SED distribution
            titles = ['PACS100', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']
            fig, ax = plt.subplots(2, 3, figsize=(10, 7.5))
            sed = sed[bin_]
            for j in range(5):
                temp = sed[:, j][~np.isnan(sed[:, j])]
                ax[j // 3, j % 3].hist(temp, bins=10)
                ax[j // 3, j % 3].set_title(titles[j])
            pp.savefig(fig)
            if plot_model:
                ccf = np.zeros(5)
                pacs_rsrf = pd.read_csv("data/RSRF/PACS_RSRF.csv")
                pacs_wl = pacs_rsrf['Wavelength'].values
                pacs_nu = (c / pacs_wl / u.um).to(u.Hz)
                pacs100dnu = pacs_rsrf['PACS_100'].values * \
                    pacs_rsrf['dnu'].values[0]
                pacs160dnu = pacs_rsrf['PACS_160'].values * \
                    pacs_rsrf['dnu'].values[0]
                del pacs_rsrf
                #
                pacs_models = B(aT[i] * u.K, pacs_nu) * pacs_wl**(-aBeta[i])
                del pacs_nu
                ccf[0] = np.sum(pacs100dnu * pacs_wl / wl[0]) / \
                    np.sum(pacs100dnu * pacs_models /
                           (B(aT[i] * u.K, nu[0]) * wl[0]**(-aBeta[i])))
                ccf[1] = np.sum(pacs160dnu * pacs_wl / wl[1]) / \
                    np.sum(pacs160dnu * pacs_models /
                           (B(aT[i] * u.K, nu[1]) * wl[1]**(-aBeta[i])))
                #
                del pacs_wl, pacs100dnu, pacs160dnu, pacs_models
                ##
                spire_rsrf = pd.read_csv("data/RSRF/SPIRE_RSRF.csv")
                spire_wl = spire_rsrf['Wavelength'].values
                spire_nu = (c / spire_wl / u.um).to(u.Hz)
                spire250dnu = spire_rsrf['SPIRE_250'].values * \
                    spire_rsrf['dnu'].values[0]
                spire350dnu = spire_rsrf['SPIRE_350'].values * \
                    spire_rsrf['dnu'].values[0]
                spire500dnu = spire_rsrf['SPIRE_500'].values * \
                    spire_rsrf['dnu'].values[0]
                del spire_rsrf
                #
                spire_models = B(aT[i] * u.K, spire_nu) * spire_wl**(-aBeta[i])
                del spire_nu
                ccf[2] = np.sum(spire250dnu * spire_wl / wl[2]) / \
                    np.sum(spire250dnu * spire_models /
                           (B(aT[i] * u.K, nu[2]) * wl[2]**(-aBeta[i])))
                ccf[3] = np.sum(spire350dnu * spire_wl / wl[3]) / \
                    np.sum(spire350dnu * spire_models /
                           (B(aT[i] * u.K, nu[3]) * wl[3]**(-aBeta[i])))
                ccf[4] = np.sum(spire500dnu * spire_wl / wl[4]) / \
                    np.sum(spire500dnu * spire_models /
                           (B(aT[i] * u.K, nu[4]) * wl[4]**(-aBeta[i])))
                #
                del spire_wl, spire250dnu, spire350dnu, spire500dnu
                del spire_models
                sed_plot = ased[i] * ccf
                #
                sed_unc_plot = np.sqrt(np.nanmean(sed_unc[bin_]**2, axis=0))
                #
                fig, ax = plt.subplots(figsize=(10, 7.5))
                ax.errorbar(wl, sed_plot, yerr=sed_unc_plot, color='red',
                            fmt='o', capsize=10, label='Herschel data')
                wl_plot = np.linspace(51, 549, 100)
                pdf /= np.nanmax(pdf) / 0.5
                for j in range(len(pdf)):
                    ax.plot(wl_plot,
                            model(wl_plot, 10**logsigmas[j], aT[i], aBeta[i],
                                  (c / wl_plot / u.um).to(u.Hz)), 'k',
                            alpha=pdf[j])
                ax.plot(wl_plot,
                        model(wl_plot, 10**LogSigmaD_Exp, aT[i], aBeta[i],
                              (c / wl_plot / u.um).to(u.Hz)),
                        label='Model: Expectation (Log)',
                        linewidth=3)
                ax.legend(fontsize=20)
                ax.set_xlabel(r'Wavelength ($\mu m$)', size=20)
                ax.set_ylabel(r'SED ($MJy$ $sr^{-1}$)', size=20)
                ax.set_xticklabels(ax.get_xticks(), fontsize=12)
                ax.set_yticklabels(ax.get_yticks(), fontsize=12)
                pp.savefig(fig)
                pp.close()
            plt.close('all')
            return 0
