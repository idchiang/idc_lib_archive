import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.constants as const
import astropy.units as u
import h5py
from astropy.io import fits
from idc_lib.idc_functions import SEMBB, PowerLaw, save_fits_gz
from idc_lib.z0mg_RSRF import z0mg_RSRF

datapath = 'data/UTOMO18_dust/'
projectpath = 'Projects/UTOMO18/'
bands = ['100', '160', '250', '350', '500']
bands2 = ['pacs100', 'pacs160', 'spire250', 'spire350', 'spire500']
names = ['smc', 'lmc', 'm31', 'm33']
# names = ['m31']
plt.ioff()


# Generate PACS100 SED from fitted parameters
wls = np.arange(14, 501)
for name in ['smc', 'lmc']:
    # FB
    with h5py.File('hdf5_MBBDust/Calibration_1.8.h5', 'r') as hf:
        kappa160 = hf['FB']['kappa160'].value
    res_path = projectpath + name + '/res_13pc_FB/'
    fns = os.listdir(res_path)
    for fn in fns:
        temp = fn.split('_')
        if len(temp) > 1:
            if temp[1] == 'dust.surface.density':
                s = fits.getdata(res_path + fn)[0]
            elif temp[1] == 'dust.temperature':
                t = fits.getdata(res_path + fn)[0]
    pacs100 = np.empty_like(s)
    x, y = pacs100.shape
    for i in range(x):
        for j in range(y):
            sed = SEMBB(wls, s[i, j], t[i, j], beta=1.8, kappa160=kappa160)
            pacs100[i, j] = z0mg_RSRF(wls, sed, bands=['PACS_100'])[0]
    save_fits_gz(res_path + 'pacs100_fit.fits', pacs100, None)
    # PL
    with h5py.File('hdf5_MBBDust/Calibration_1.8.h5', 'r') as hf:
        kappa160 = hf['PL']['kappa160'].value
    res_path = projectpath + name + '/res_13pc_PL/'
    fns = os.listdir(res_path)
    for fn in fns:
        temp = fn.split('_')
        if len(temp) > 1:
            if temp[1] == 'dust.surface.density':
                s = fits.getdata(res_path + fn)[0]
            elif temp[1] == 'alpha':
                a = fits.getdata(res_path + fn)[0]
            elif temp[1] == 'gamma':
                g = fits.getdata(res_path + fn)[0]
            elif temp[1] == 'logUmin':
                lu = fits.getdata(res_path + fn)[0]
    pacs100 = np.empty_like(s)
    x, y = pacs100.shape
    for i in range(x):
        for j in range(y):
            sed = PowerLaw(wls, s[i, j], a[i, j], gamma=g[i, j],
                           logUmin=lu[i, j], logUmax=3.0,
                           beta=1.8, kappa160=kappa160)
            pacs100[i, j] = z0mg_RSRF(wls, sed, bands=['PACS_100'])[0]
    save_fits_gz(res_path + 'pacs100_fit.fits', pacs100, None)

sys.exit()

# Compare m33 temperature with resolution
name = 'm33'
resolutions = [167, 300, 500, 1000, 2000]
ress = ['res_' + str(r) + 'pc' for r in resolutions]
Ts = []
ss = []
for res in ress:
    res_path = 'Projects/UTOMO18/' + name + '/' + res + '/'
    fns = os.listdir(res_path)
    s, t = 0, 0
    for fn in fns:
        temp = fn.split('_')
        if len(temp) > 1:
            if temp[1] == 'dust.surface.density':
                s = fits.getdata(res_path + fn)[0]
            elif temp[1] == 'dust.temperature':
                t = fits.getdata(res_path + fn)[0]
    mask = np.isfinite(s + t)
    r = int(res.strip('res_').strip('pc'))
    Ts.append(np.sum((s * t)[mask]) / np.sum(s[mask]))
    ss.append(np.sum(s[mask] * r**2))
Tdfs = []
sdfs = []
areas = []
for res in ress:
    res_path = 'Projects/UTOMO18_oringal_covariance/' + name + '/' + res + '/'
    fns = os.listdir(res_path)
    s, t = 0, 0
    for fn in fns:
        temp = fn.split('_')
        if len(temp) > 1:
            if temp[1] == 'dust.surface.density':
                s = fits.getdata(res_path + fn)[0]
            elif temp[1] == 'dust.temperature':
                t = fits.getdata(res_path + fn)[0]
    mask = np.isfinite(s + t)
    r = int(res.strip('res_').strip('pc'))
    area = np.sum(mask) * r**2
    areas.append(area)
    Tdfs.append(np.sum((s * t)[mask]) / np.sum(s[mask]))
    sdfs.append(np.sum(s[mask]) * r**2)
areas = np.array(areas)
print(areas)
fig, ax = plt.subplots()
ax.plot(resolutions, Ts, label='resolved')
ax.plot(resolutions, Tdfs, label='original covariance')
ax.set_xscale('log')
ax.set_title(name + r' : integrated $T_d$')
ax.set_xlabel('resolution (pc)')
ax.set_ylabel(r'$T_d$ (K)')
ax.legend()
# ax[i].set_yscale('log')
fig.tight_layout()
fig.savefig('output/' + name + '_Td.png')
plt.close()
#
fig, ax = plt.subplots()
mm = ss[0]
ax.plot(resolutions, np.array(ss) / mm, label='resolved')
ax.plot(resolutions, np.array(sdfs) / mm,
        label='original covariance')
ax.set_xscale('log')
ax.set_title(name + r' : integrated $M_t$')
ax.set_xlabel('resolution (pc)')
ax.set_ylabel(r'$M_t$ ($M_\odot$)')
ax.legend()
# ax[i].set_yscale('log')
fig.tight_layout()
fig.savefig('output/' + name + '_Mt.png')
plt.close()

sys.exit()

# Generate the TIR
nus = np.linspace((const.c / (1 * u.um)).to(u.Hz).value,
                  (const.c / (10 * u.um)).to(u.Hz).value, 1000)
dnu = np.abs(np.mean(nus[1:] - nus[:-1])) * u.Hz
wls = (const.c / (nus * u.Hz)).to(u.um).value
beta = 1.8
with h5py.File('hdf5_MBBDust/Calibration_1.8.h5', 'r') as hf:
    kappa160 = hf['FB']['kappa160'].value
for name in names:
    if name in ['m31', 'm33']:
        res = 'res_167pc'
        area = (167 * u.pc)**2
    elif name in ['smc', 'lmc']:
        res = 'res_13pc'
        area = (13 * u.pc)**2
    res_path = 'Projects/UTOMO18/' + name + '/' + res + '/'
    fns = os.listdir(res_path)
    for fn in fns:
        temp = fn.split('_')
        if len(temp) == 6:
            if temp[1] == 'dust.surface.density':
                se, hdr = fits.getdata(res_path + fn, header=True)
                se = se[0]
            elif temp[1] == 'dust.temperature':
                te = fits.getdata(res_path + fn)[0]
    TIR = np.full_like(se, np.nan, dtype=float)
    for i in range(se.shape[0]):
        for j in range(se.shape[1]):
            if np.isfinite(se[i, j] + te[i, j]):
                SED = SEMBB(wls, se[i, j], te[i, j], beta=beta,
                            kappa160=kappa160)
                TIR[i, j] = np.log10(np.sum(SED))
    TIR += np.log10((area * dnu * u.MJy).to(u.W).value)
    fn = res_path + name + '_TIR.fits'
    hdr['NAXIS'] = 2
    hdr['BUNIT'] = 'LogL_TIR'
    hdr.comments['BUNIT'] = 'log(W/sr)'
    del hdr['NAXIS3']
    save_fits_gz(fn, TIR, hdr)

sys.exit()

# Plot temperature with resolution
for name in names:
    if name in ['m31', 'm33']:
        resolutions = [167, 300, 500, 1000, 2000, 4000, 8000, 15000]
    elif name in ['smc', 'lmc']:
        resolutions = [13, 30, 50, 100, 167, 300, 500, 1000, 2000, 4000]
    ress = ['integrated_res_' + str(r) + 'pc' for r in resolutions]
    Tints = []
    Terrs = []
    sints = []
    for res in ress:
        res_path = 'Projects/UTOMO18/' + name + '/' + res + '/'
        df = pd.read_csv(res_path + name + '_integrated.csv')
        Tints.append(df['dust.temperature'].values[0])
        sints.append(df['dust.surface.density'].values[0])
        Terrs.append(df['dust.temperature.err'].values[0])
    ress = ['res_' + str(r) + 'pc' for r in resolutions]
    sdfints = []
    Tdfints = []
    for res in ress:
        res_path = 'Projects/UTOMO18_DF/' + name + '/integrated_' + res + '/'
        df = pd.read_csv(res_path + name + '_integrated.csv')
        Tdfints.append(df['dust.temperature'].values[0])
        sdfints.append(df['dust.surface.density'].values[0])
    ress = ['res_' + str(r) + 'pc' for r in resolutions]
    Ts = []
    ss = []
    for res in ress:
        res_path = 'Projects/UTOMO18/' + name + '/' + res + '/'
        fns = os.listdir(res_path)
        s, t = 0, 0
        for fn in fns:
            temp = fn.split('_')
            if len(temp) > 1:
                if temp[1] == 'dust.surface.density':
                    s = fits.getdata(res_path + fn)[0]
                elif temp[1] == 'dust.temperature':
                    t = fits.getdata(res_path + fn)[0]
        mask = np.isfinite(s + t)
        r = int(res.strip('res_').strip('pc'))
        Ts.append(np.sum((s * t)[mask]) / np.sum(s[mask]))
        ss.append(np.sum(s[mask] * r**2))
    Tdfs = []
    sdfs = []
    areas = []
    for res in ress:
        res_path = 'Projects/UTOMO18_DF/' + name + '/' + res + '/'
        fns = os.listdir(res_path)
        s, t = 0, 0
        for fn in fns:
            temp = fn.split('_')
            if len(temp) > 1:
                if temp[1] == 'dust.surface.density':
                    s = fits.getdata(res_path + fn)[0]
                elif temp[1] == 'dust.temperature':
                    t = fits.getdata(res_path + fn)[0]
        mask = np.isfinite(s + t)
        r = int(res.strip('res_').strip('pc'))
        area = np.sum(mask) * r**2
        areas.append(area)
        Tdfs.append(np.sum((s * t)[mask]) / np.sum(s[mask]))
        sdfs.append(np.sum(s[mask]) * r**2)
    areas = np.array(areas)
    print(areas)
    fig, ax = plt.subplots()
    ax.plot(resolutions, Ts, label='resolved')
    ax.plot(resolutions, Tdfs, label='deep field covariance')
    ax.plot(resolutions, Tints, label='integrated')
    ax.plot(resolutions, Tdfints, label='integrated DF cov')
    ax.set_xscale('log')
    ax.set_title(name + r' : integrated $T_d$')
    ax.set_xlabel('resolution (pc)')
    ax.set_ylabel(r'$T_d$ (K)')
    ax.legend()
    # ax[i].set_yscale('log')
    fig.tight_layout()
    fig.savefig('output/' + name + '_Td.png')
    plt.close()
    #
    fig, ax = plt.subplots()
    mm = ss[0]
    ax.plot(resolutions, np.array(ss) / mm, label='resolved')
    ax.plot(resolutions, np.array(sdfs) / mm,
            label='deep field covariance')
    ax.plot(resolutions, np.array(sints) * areas / mm, label='integrated')
    ax.plot(resolutions, np.array(sdfints) * areas / mm,
            label='integrated DF cov')
    ax.set_xscale('log')
    ax.set_title(name + r' : integrated $M_t$')
    ax.set_xlabel('resolution (pc)')
    ax.set_ylabel(r'$M_t$ ($M_\odot$)')
    ax.legend()
    # ax[i].set_yscale('log')
    fig.tight_layout()
    fig.savefig('output/' + name + '_Mt.png')
    plt.close()


sys.exit(0)
plt.ion()

# Plot covariance / correlation
names = ['m33']
for name in names:
    if name in ['m31', 'm33']:
        resolutions = [167, 300, 500, 1000, 2000, 4000, 8000, 15000]
    elif name in ['smc', 'lmc']:
        resolutions = [13, 30, 50, 100, 167, 300, 500, 1000, 2000, 4000]
    ress = ['res_' + str(r) + 'pc' for r in resolutions]
    bkgcovs = []
    corrs = []
    for res in ress:
        fn = datapath + name + '/' + res + '/' + name + '_bkgcov.fits'
        bkgcov = fits.getdata(fn)
        bkgcovs.append(bkgcov)
        std = np.sqrt(np.diagonal(bkgcov))
        corr = np.copy(bkgcov)
        for i in range(5):
            for j in range(5):
                corr[i, j] /= (std[i] * std[j])
        corrs.append(corr)
    corrs_fit = np.array(corrs)
    bkgcovs = np.array(bkgcovs)
    #
    corrs = []
    bkgcovs_df = []
    for res in ress:
        fn = datapath + name + '/' + res + '/' + name + '_bkgcov_df.fits.gz'
        try:
            bkgcov = fits.getdata(fn)[::-1, ::-1]
        except FileNotFoundError:
            bkgcov = np.zeros([5, 5])
        bkgcovs_df.append(bkgcov)
        std = np.sqrt(np.diagonal(bkgcov))
        corr = np.copy(bkgcov)
        for i in range(5):
            for j in range(5):
                corr[i, j] /= (std[i] * std[j])
        corrs.append(corr)
    corrs_df = np.array(corrs)
    bkgcovs_df = np.array(bkgcovs_df)
    #
    bkgcovs_raw = []
    corrs = []
    for res in ress:
        rpath = datapath + name + '/' + res + '/'
        fns = os.listdir(rpath)
        seds = []
        for b in ['pacs100', 'pacs160', 'spire250', 'spire350', 'spire500']:
            for fn in fns:
                temp = fn.split('_')
                if (len(temp) == 5) and temp[1] == b:
                    seds.append(fits.getdata(rpath + fn))
                    break
        assert len(seds) == 5
        seds = np.array(seds)
        finite_mask = np.all(np.isfinite(seds), axis=0)
        diskmask = fits.getdata(rpath + name + '_diskmask.fits').astype(bool)
        outlier_mask = np.zeros_like(seds[0], dtype=bool)
        for i in range(5):
            AD = np.abs(seds[i] - np.nanmedian(seds[i]))
            MAD = np.nanmedian(AD)
            outlier_mask += AD > 3 * MAD
        bkg_mask = (~outlier_mask) * (~diskmask) * finite_mask
        bkgcov = np.cov(seds[:, bkg_mask])
        bkgcovs_raw.append(bkgcov)
        std = np.sqrt(np.diagonal(bkgcov))
        corr = np.copy(bkgcov)
        for i in range(5):
            for j in range(5):
                corr[i, j] /= (std[i] * std[j])
        corrs.append(corr)
    corrs_raw = np.array(corrs)
    bkgcovs_raw = np.array(bkgcovs_raw)
    #
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            ax[i, j].loglog(resolutions, bkgcovs[:, i, j],
                            marker='o', ms=5, color='blue', label='FIT')
            ax[i, j].loglog(resolutions, bkgcovs_df[:, i, j],
                            marker='o', ms=5, color='red', label='DF')
            ax[i, j].loglog(resolutions, bkgcovs_raw[:, i, j],
                            marker='o', ms=5, color='orange', label='RAW')
            if i + j == 0:
                ax[i, j].set_title(name + ' : ' + bands[i] + '-' + bands[j])
                ax[i, j].legend()
            else:
                ax[i, j].set_title(bands[i] + '-' + bands[j])
            if i == 4:
                ax[i, j].set_xlabel('resolution (pc)')
    fig.tight_layout()
    fig.savefig('output/' + name + '+df.png')
    plt.close()
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            ax[i, j].semilogx(resolutions, corrs_fit[:, i, j],
                              marker='o', ms=5, color='blue', label='FIT')
            ax[i, j].semilogx(resolutions, corrs_df[:, i, j],
                              marker='o', ms=5, color='red', label='DF')
            ax[i, j].semilogx(resolutions, corrs_raw[:, i, j],
                              marker='o', ms=5, color='orange', label='RAW')
            if i + j == 0:
                ax[i, j].set_title(name + ' : ' + bands[i] + '-' + bands[j])
                ax[i, j].legend()
            else:
                ax[i, j].set_title(bands[i] + '-' + bands[j])
            if i == 4:
                ax[i, j].set_xlabel('resolution (pc)')
    fig.tight_layout()
    fig.savefig('output/' + name + '_corr+df.png')
    plt.close()

sys.exit(0)
plt.ion()


# Plot gamma distribution from the integrated PL fitting RL cubes
names_ = np.array([['SMC', 'LMC'], ['M31', 'M33']])
wl5 = [100, 160, 250, 350, 500]
rs = {'SMC': 13, 'LMC': 13, 'M31': 167, 'M33': 167}
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
fig2, ax2 = plt.subplots(2, 2, figsize=(8, 8))
gammas = np.arange(-3.1, 0.1, 0.2)
alphas = np.arange(1.0, 5.2, 0.2)
for i in range(2):
    for j in range(2):
        n = names_[i, j]
        fp = projectpath + n + '_PL/integrated_res_' + str(rs[n]) + 'pc/'
        fns = os.listdir(fp)
        for fn in fns:
            temp = fn.split('_')
            if len(temp) > 1:
                if temp[1] == 'gamma.rlcube':
                    data = np.log10(fits.getdata(fp + fn)[:, 0, 0])
                elif temp[1] == 'alpha.rlcube':
                    data2 = fits.getdata(fp + fn)[:, 0, 0]
        a = ax[i, j]
        a.hist(data, bins=gammas)
        a.set_xlabel(r'log$_{10}$($\gamma$)')
        a.set_title(n + ' (integrated)')
        a2 = ax2[i, j]
        a2.hist(data2, bins=alphas)
        a2.set_xlabel(r'$\alpha$')
        a2.set_title(n + ' (integrated)')
fig.tight_layout()
fig.savefig('output/gamma_distribution.png')
fig2.tight_layout()
fig2.savefig('output/alpha_distribution.png')
plt.close('all')

# Plot recovered SED from resolved/integrated FB/PL
names_ = np.array([['SMC', 'LMC'], ['M31', 'M33']])
wl5 = [100, 160, 250, 350, 500]
integrated_sed = {'SMC': [6.041486, 7.998705, 4.216746, 2.464281, 1.167984],
                  'LMC': [14.71198, 17.54551, 9.208895, 4.736695, 2.005008],
                  'M31': [6.80247, 16.5233, 10.666, 5.77729, 2.49725],
                  'M33': [8.658589, 12.81489, 7.588042, 4.227531, 1.981157]}
# density, temperature
resolved_FB = {'SMC': [0.0215, 17.9211],
               'LMC': [0.03474, 19.958576],
               'M31': [0.067755, 16.8463],
               'M33': [0.05, 16.6]}
integrated_FB = {'SMC': [0.018287, 19.59329],
                 'LMC': [0.030923, 21.24215],
                 'M31': [0.06705, 17.3748],
                 'M33': [0.034871, 19.18061]}
# density, alpha, logUmin, gamma
resolved_PL = {'SMC': [0.022, 2.90, -0.19, 0.485],
               'LMC': [0.035, 3.60, 0.30, 0.439],
               'M31': [0.068, 3.60, 0.00, 0.340],
               'M33': [0.051, 4.20, 0.48, 0.08]}
integrated_PL = {'SMC': [0.045, 1.61, -0.53, 0.207],
                 'LMC': [0.042, 2.15, 0.12, 0.087],
                 'M31': [0.066, 3.31, -0.11, 0.015],
                 'M33': [0.060, 1.82, -0.29, 0.133]}
wl = np.linspace(100, 500)
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
for i in range(2):
    for j in range(2):
        a = ax[i, j]
        n = names_[i, j]
        sigma, temp = integrated_FB[n]
        a.plot(wl, SEMBB(wl, sigma, temp, beta=1.8, kappa160=18.68),
               label='I FB')
        sigma, temp = resolved_FB[n]
        a.plot(wl, SEMBB(wl, sigma, temp, beta=1.8, kappa160=18.68),
               label='R FB')
        sigma, alpha, logUmin, gamma = integrated_PL[n]
        a.plot(wl, PowerLaw(wl, sigma, alpha, gamma, logUmin, logUmax=3.0,
                            beta=1.8, kappa160=20.3764).reshape(-1),
               label='I PL')
        sigma, alpha, logUmin, gamma = resolved_PL[n]
        a.plot(wl, PowerLaw(wl, sigma, alpha, gamma, logUmin, logUmax=3.0,
                            beta=1.8, kappa160=20.3764).reshape(-1),
               label='R PL')
        a.scatter(wl5, integrated_sed[n], c='r', label='I SED')
        a.legend()
        a.set_xlabel(r'$\lambda$ [$\mu$m]')
        a.set_ylabel(r'$F_\nu$ [MJy sr$^{-1}$]')
        a.set_title(n)
fig.tight_layout()
fig.savefig('output/Recovered_SEDs.png')
plt.close()


# Plot covariance
for name in names:
    if name in ['m31', 'm33']:
        resolutions = [167, 300, 500, 1000, 2000, 4000, 8000, 15000]
    elif name in ['smc', 'lmc']:
        resolutions = [13, 30, 50, 100, 167, 300, 500, 1000, 2000, 4000]
    ress = ['res_' + str(r) + 'pc' for r in resolutions]
    bkgcovs = []
    for res in ress:
        fn = datapath + name + '/' + res + '/' + name + '_bkgcov.fits'
        bkgcov = fits.getdata(fn)
        bkgcovs.append(bkgcov)
    bkgcovs = np.array(bkgcovs)
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            ax[i, j].loglog(resolutions, bkgcovs[:, i, j],
                            marker='o', ms=5)
            if i + j == 0:
                ax[i, j].set_title(name + ' : ' + bands[i] + '-' + bands[j])
            else:
                ax[i, j].set_title(bands[i] + '-' + bands[j])
            if i == 4:
                ax[i, j].set_xlabel('resolution (pc)')
    fig.tight_layout()
    fig.savefig('output/' + name + '.png')
    plt.close()

# Plot temperature with resolution
for name in names:
    if name in ['m31', 'm33']:
        resolutions = [167, 300, 500, 1000, 2000, 4000, 8000, 15000]
    elif name in ['smc', 'lmc']:
        resolutions = [13, 30, 50, 100, 167, 300, 500, 1000, 2000, 4000]
    ress = ['integrated_res_' + str(r) + 'pc' for r in resolutions]
    Tints = []
    Terrs = []
    for res in ress:
        res_path = 'Projects/UTOMO18/' + name + '/' + res + '/'
        df = pd.read_csv(res_path + name + '_integrated.csv')
        Tints.append(df['dust.temperature'].values[0])
        Terrs.append(df['dust.temperature.err'].values[0])
    ress = ['res_' + str(r) + 'pc' for r in resolutions]
    Ts = []
    for res in ress:
        res_path = 'Projects/UTOMO18/' + name + '/' + res + '/'
        fns = os.listdir(res_path)
        s, t = 0, 0
        for fn in fns:
            temp = fn.split('_')
            if len(temp) > 1:
                if temp[1] == 'dust.surface.density':
                    s = fits.getdata(res_path + fn)[0]
                elif temp[1] == 'dust.temperature':
                    t = fits.getdata(res_path + fn)[0]
        mask = np.isfinite(s + t)
        Ts.append(np.sum((s * t)[mask]) / np.sum(s[mask]))
    fig, ax = plt.subplots()
    ax.plot(resolutions, Ts, label='resolved')
    ax.plot(resolutions, Tints, label='integrated')
    ax.set_xscale('log')
    ax.set_title(name + r' : integrated $T_d$')
    ax.set_xlabel('resolution (pc)')
    ax.set_ylabel(r'$T_d$ (K)')
    ax.legend()
    # ax[i].set_yscale('log')
    fig.tight_layout()
    fig.savefig('output/' + name + '_Td.png')
    plt.close()

# Plot standard deviation
for name in names:
    if name in ['m31', 'm33']:
        resolutions = [167, 300, 500, 1000, 2000, 4000]
    elif name in ['smc', 'lmc']:
        resolutions = [13, 30, 50, 100, 167, 300, 500, 1000]
    ress = ['res_' + str(r) + 'pc' for r in resolutions]
    bkgcovs = []
    for res in ress:
        res_path = projectpath + name + '/' + res + '/'
        fns = os.listdir(res_path)
        for fn in fns:
            temp = fn.split('_')
            if len(temp) > 2:
                if temp[1] == 'bkgcov':
                    bkgcovs.append(fits.getdata(res_path + fn))
                    break
    bkgcovs = np.array(bkgcovs)
    fig, ax = plt.subplots(1, 5, figsize=(10, 3))
    for i in range(5):
        ax[i].plot(1 / np.array(resolutions), np.sqrt(bkgcovs[:, i, i]),
                   marker='o', ms=5)
        if i == 0:
            ax[i].set_title(name + ' : ' + bands[i])
        else:
            ax[i].set_title(bands[i])
        ax[i].set_xlabel('1 / resolution (pc)')
        # ax[i].set_yscale('log')
        # ax[i].set_xscale('log')
    fig.tight_layout()
    fig.savefig('output/' + name + '_standard_deviation.png')
    plt.close()


# Plot correlation
for name in names:
    if name in ['m31', 'm33']:
        resolutions = [167, 300, 500, 1000, 2000, 4000]
    elif name in ['smc', 'lmc']:
        resolutions = [13, 30, 50, 100, 167, 300, 500, 1000]
    ress = ['res_' + str(r) + 'pc' for r in resolutions]
    bkgcovs = []
    for res in ress:
        res_path = projectpath + name + '/' + res + '/'
        fns = os.listdir(res_path)
        for fn in fns:
            temp = fn.split('_')
            if len(temp) > 2:
                if temp[1] == 'bkgcov':
                    bkgcovs.append(fits.getdata(res_path + fn))
                    break
    bkgcovs = np.array(bkgcovs)
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            ax[i, j].plot(resolutions,
                          bkgcovs[:, i, j] /
                          np.sqrt(np.abs(bkgcovs[:, i, i] * bkgcovs[:, j, j])),
                          marker='o', ms=5)
            # ax[i, j].set_xscale('log')
            if i + j == 0:
                ax[i, j].set_title(name + ' : ' + bands[i] + '-' + bands[j])
            else:
                ax[i, j].set_title(bands[i] + '-' + bands[j])
            if i == 4:
                ax[i, j].set_xlabel('resolution (pc)')
    fig.tight_layout()
    fig.savefig('output/' + name + '_correlation.png')
    plt.close()

# Plot bkg info
for name in names:
    if name in ['m31', 'm33']:
        resolutions = [167, 300, 500, 1000, 2000, 4000]
    elif name in ['smc']:
        resolutions = [13, 30, 50, 100, 167, 300, 500, 1000]
    elif name in ['lmc']:
        resolutions = [13, 30, 50, 100, 300, 500, 1000]
    ress = ['res_' + str(r) + 'pc' for r in resolutions]
    bkgmeans = [[] for i in range(5)]
    for r in range(len(ress)):
        res = ress[r]
        res_path = datapath + name + '/' + res + '/'
        fns = os.listdir(res_path)
        diskmask = []
        sed = []
        # build diskmask, read all sed
        for band in bands2:
            for fn in fns:
                temp = fn.split('_')
                if len(temp) < 4:
                    continue
                if temp[-4] == band:
                    if temp[-1] == 'mask.fits':
                        diskmask.append(fits.getdata(res_path +
                                                     fn).astype(bool))
                    else:
                        sed.append(fits.getdata(res_path + fn))
        assert len(diskmask) == 5
        assert len(sed) == 5
        diskmask = np.all(diskmask, axis=0)
        # build bkgmask
        bkgmask = (~diskmask) * np.all(np.isfinite(sed), axis=0)
        # save bkg info
        for i in range(5):
            bkgmeans[i].append(np.nanmean(sed[i][bkgmask]))
        if r == 0:
            bkgs = [sed[i][bkgmask] for i in range(5)]
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(5):
        ax[0, i].plot(resolutions, bkgmeans[i], marker='o', ms=5)
        if i == 0:
            ax[0, i].set_title(name + ' : ' + bands2[i] + '-bkg mean')
        else:
            ax[0, i].set_title(bands2[i] + '-bkg mean flux')
        ax[0, i].set_xlabel('resolution (pc)')
        bkgs[i].sort()
        temp = np.arange(len(bkgs[i])) / len(bkgs[i])
        xlim = np.interp([0.01, 0.99], temp, bkgs[i])
        ax[1, i].hist(bkgs[i], bins=20, range=xlim)
        ax[1, i].set_title('best res hist')
    fig.tight_layout()
    fig.savefig('output/' + name + '_bkg.png')
    plt.close()
