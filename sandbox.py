import numpy as np
import matplotlib.pyplot as plt
from h5py import File
from idc_lib import gal_data
# New imports
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr
from idc_lib.idc_io import MGS


M101 = ['NGC5457']  # Currently focusing on NGC5457
all_surveys = ['THINGS', 'SPIRE_500', 'SPIRE_350', 'SPIRE_250',
               'PACS_160', 'PACS_100', 'HERACLES', 'MIPS_24', 'GALEX_FUV',
               'IRAC_3.6']
all_kernels = ['Gauss_25', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100',
               'IRAC_3.6', 'MIPS_24', 'GALEX_FUV']
fine_surveys = ['THINGS', 'SPIRE_350', 'SPIRE_250', 'PACS_160',
                'PACS_100', 'HERACLES', 'IRAC_3.6', 'MIPS_24', 'GALEX_FUV']
cut_surveys = ['RADIUS_KPC', 'SFR', 'SMSD']


def generator(samples=M101):
    mgs = MGS(samples, all_surveys)
    mgs.add_kernel(all_kernels, 'SPIRE_500')
    mgs.matching_PSF(samples, fine_surveys, 'SPIRE_500')
    mgs.WCS_congrid(samples, fine_surveys, 'SPIRE_500')
    mgs.SFR(samples)
    mgs.SMSD(samples)
    mgs.total_gas(samples)
    mgs.cut_image(samples, all_surveys)
    mgs.cut_image(samples, cut_surveys, unc=False)
    mgs.June09_test()
    return mgs.df

# df = generator()


def f(x, a, b):
    return a * x + b


def corr_test(name='NGC5457', bins=30, off=45., cmap0='gist_heat',
              dr25=0.025, ncmode=False, cmap2='seismic', fixed_beta=True):
    plt.close('all')
    if fixed_beta:
        fn = 'output/' + name + '_dust_data_fb.h5'
    else:
        fn = 'output/' + name + '_dust_data.h5'
    with File(fn, 'r') as hf:
        binmap = np.array(hf['Binmap'])
        binlist = np.array(hf['Binlist'])
        atopt = np.array(hf['Dust_temperature'])
        D = float(np.array(hf['Galaxy_distance']))

    with File('output/RGD_data.h5', 'r') as hf:
        grp = hf[name]
        ubradius = np.array(grp['DP_RADIUS'])
        logSFR = np.array(grp['logSFR'])

    topt = np.full_like(binmap, np.nan, dtype=float)
    for i in range(len(binlist)):
        mask = binmap == binlist[i]
        topt[mask] = atopt[i]

    R25 = gal_data.gal_data([name]).field('R25_DEG')[0]
    R25 *= (np.pi / 180.) * (D * 1E3)
    ubradius /= R25

    max_r = np.nanmax(ubradius)
    rus = np.linspace(0, max_r, bins)
    rs, cs = [], []
    ubradius[np.isnan(ubradius)] = max_r + 1
    for i in range(1, len(rus)):
        rs.append((rus[i] + rus[i - 1]) / 2)
        mask = (ubradius <= rus[i]) * (~np.isnan(logSFR + topt))
        cs.append(pearsonr(topt[mask], logSFR[mask])[0])

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(rs, cs)
    ax.set_xlabel('Radius (r25)', size=16)
    ax.set_ylabel('Correlation Coefficient', size=16)
    ax.set_xlim([0.0, 1.5])
    fig.savefig('output/correlation_T_SFR_vs_radius.png')

    # Try to fit Temperature map?
    # Plot Residue map; plot temperature trend
    r_bdr = 0.7
    mask = (ubradius <= r_bdr) * (~np.isnan(logSFR + topt))
    popt, pcov = curve_fit(f, logSFR[mask], topt[mask])
    t_pred = logSFR * popt[0] + popt[1]
    nanmask = np.isnan(logSFR + topt)
    t_pred[nanmask], topt[nanmask] = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(12, 9))
    cax = ax.imshow(topt / t_pred, origin='lower', cmap=cmap2, vmin=-1, vmax=3)
    fig.colorbar(cax, ax=ax)
    fig.savefig('output/Topt_d_Tpred.png')

    to, tp = [], []
    for i in range(1, len(rus)):
        mask = (rus[i - 1] < ubradius) * (ubradius <= rus[i]) * \
            (~np.isnan(logSFR + topt))
        to.append(np.mean(topt[mask]))
        tp.append(np.mean(t_pred[mask]))

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(rs, to, label='Fitted temperature')
    ax.plot(rs, tp, label='Predicted temperature')
    ax.set_xlabel('Radius (r25)', size=16)
    ax.set_ylabel('Temperature (K)', size=16)
    ax.set_xlim([0.0, 1.5])
    ax.legend(fontsize=16)
    fig.savefig('output/predicted_T.png')
    plt.show()
    plt.close('all')
