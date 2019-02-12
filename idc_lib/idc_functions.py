import os
import gzip
import shutil
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.constants import c, h, k_B
from astropy.modeling.blackbody import blackbody_lambda
from scipy.stats import pearsonr
from time import clock, ctime

hkB_KHz = (h / k_B).to(u.K / u.Hz).value
B_const = 2e20 * (h / c**2).to(u.J * u.s**3 / u.m**2).value
c_ums = c.to(u.um / u.s).value
MBB_const = 0.00020884262122368297
MWT = 18.0
WDT = 40.0
logUmax = 3.0
Umax = 10**logUmax

"""
In order in increase the speed, I replace all B() with B_fast(),
which cancels all the unit calculation in B() but becomes unit limited

the same for freq(Hz) and wl(um) in the related functions
"""


def SEMBB_lambda(wl, sigma, T, beta, kappa160=9.6 * np.pi):
    return MBB_const * kappa160 * (160.0 / wl)**beta * sigma * \
        blackbody_lambda(wl * u.um, T * u.K).value


def map2bin(data, binlist, binmap):
    nanmask = np.isnan(data + binmap)
    binmap[nanmask] = binlist.max() + 1
    for bin_ in binlist:
        mask = binmap == bin_
        data[mask] = np.nanmean(data[mask])
    return data


def list2bin(listData, binlist, binmap):
    assert len(listData) == len(binlist)
    data = np.full_like(binmap, np.nan, dtype=float)
    for i in range(len(binlist)):
        data[binmap == binlist[i]] = listData[i]
    return data


def bin2list(data, binlist, binmap):
    assert data.shape == binmap.shape
    return np.array([data[binmap == b][0] for b in binlist])


def B_fast(T, freq):
    """ Return blackbody SED of temperature T(with unit) in MJy """
    """ Get T and freq w/o unit, assuming K and Hz """
    with np.errstate(over='ignore'):
        return B_const * freq**3 / (np.exp(hkB_KHz * freq / T) - 1)


def SEMBB(wl, sigma, T, beta, kappa160=9.6 * np.pi):
    freq = c_ums / wl
    return MBB_const * kappa160 * (160.0 / wl)**beta * sigma * \
        B_fast(T, freq)


def WD(wl, TOTsigma, T, beta, WDfrac, kappa160=29.0, WDT=WDT,
       WDbeta=2.0):
    return SEMBB(wl, TOTsigma * (1 - WDfrac), T, beta, kappa160) + \
        SEMBB(wl, TOTsigma * WDfrac, WDT, WDbeta, kappa160)


def BEMBB(wl, sigma, T, beta, lambda_c, beta2, kappa160=24.8):
    """Return fitted SED in MJy"""
    freq = c_ums / wl
    ans = B_fast(T, freq) * MBB_const * kappa160 * sigma * 160.0**beta
    try:
        # Only allows 1-dim for all parameters. No error detection
        nwl = len(wl)
        del nwl
        small_mask = wl < lambda_c
        ans[small_mask] *= (1.0 / wl[small_mask])**beta
        ans[~small_mask] *= \
            (lambda_c**(beta2 - beta)) * wl[~small_mask]**(-beta2)
        return ans
    except TypeError:
        small_mask = wl < lambda_c
        ans[small_mask] *= (1.0 / wl)**beta
        ans[~small_mask] *= \
            (lambda_c[~small_mask]**(beta2[~small_mask] - beta)) * \
            wl**(-beta2[~small_mask])
        return ans


def PowerLaw(wl, Meff, alpha, gamma, logUmin, logUmax=logUmax, beta=2.0,
             kappa160=1e2, dlogU=0.02, MWT=MWT):
    freq = c_ums / wl
    x = beta + 4.0
    idx = (1 - alpha)
    #
    Umax = 10**logUmax
    Umin = 10**logUmin
    Tmin = MWT * Umin**(1/x)
    Us = 10**np.arange(np.min(logUmin) + dlogU / 2, logUmax, dlogU)
    Ts = MWT * Us**(1/x)
    #
    try:
        shape_ = list(alpha.shape)
    except AttributeError:
        shape_ = [1]
    try:
        ans = np.zeros(shape_ + [len(wl)])
    except TypeError:
        ans = np.zeros(shape_)
    #
    try:
        # Input has single wavelength
        len(Tmin)
        for i in range(len(Us)):
            mask = Tmin <= Ts[i]
            ans[mask] += Us[i]**idx[mask] * B_fast(Ts[i], freq)
    except (TypeError, SyntaxError):
        # Input has an array of wavelength
        for i in range(len(Us)):
            ans += Us[i]**idx * B_fast(Ts[i], freq)
    const = gamma * idx / (Umax**idx - Umin**idx)
    ans *= const * dlogU

    ans += (1 - gamma) * B_fast(Tmin, freq)
    return kappa160 * (160.0 / wl)**beta * MBB_const * Meff * ans


def normalize_pdf(pdf, axis_id):
    sum_axes = tuple([i for i in range(pdf.ndim) if i != axis_id])
    pdf = np.sum(pdf, axis=sum_axes)
    return pdf / np.sum(pdf)


def fit_DataY(DataX, DataY, DataY_err, err_bdr=0.3, quiet=False):
    print('Calculating predicted parameter... (' + ctime() + ')')
    tic = clock()
    nanmask = np.isnan(DataY + DataY_err + DataX)
    mask = ((DataY_err / DataY) < err_bdr) * ~nanmask
    if not quiet:
        print('Number of nan points:', np.sum(nanmask))
        print('Number of negative yerr:', np.sum(DataY_err < 0))
        print('Number of zero yerr:', np.sum(DataY_err == 0))
        print('Number of available data:', mask.sum())
        print('Correlation between DataX and DataY:',
              pearsonr(DataX[mask], DataY[mask])[0])
    popt, pcov = np.polyfit(x=DataX, y=DataY, deg=1, w=1/DataY_err, cov=True)
    perr = np.sqrt(np.diag(pcov))
    if not quiet:
        print('Fitting coef:', popt)
        print('Fitting err :', perr)
    DataX[nanmask] = 0.0
    DataY_pred = DataX * popt[0] + popt[1]
    DataY_pred[nanmask] = np.nan
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.\n")
    return DataY_pred, popt


def best_fit_and_error(var, pr, varname=None, quiet=True, islog=False,
                       minmax=False):
    best_est = 'exp'
    # Calculate expectation value
    Z = np.sum(pr)
    expected = np.sum(var * pr) / Z
    # "Calculate" the maximum likelihood value
    most_likely = var[np.unravel_index(np.argmax(pr), pr.shape)]
    # Calculate 16-50-84
    var_f, pr_f = var.flatten(), pr.flatten()
    idx = np.argsort(var_f)
    sorted_var, sorted_pr = var_f[idx], pr_f[idx]
    csp = np.cumsum(sorted_pr)
    csp = csp / csp[-1]
    p16, p84 = np.interp([0.16, 0.84], csp, sorted_var)
    # Calculate simple error
    if best_est == 'exp':
        best = expected
    elif best_est == 'max':
        best = most_likely
    err = max((best - p16), (p84 - best))
    # Print results to screen
    if not quiet:
        print('Best', varname + ':', best)
        print('    ' + ' ' * len(varname) + 'error:', err)
    # Return the results
    if minmax:
        if islog:
            return 10**best, 10**p16, 10**p84
        else:
            return best, p16, p84
    else:
        if islog:
            return 10**best, err, \
                max((10**best - 10**p16), (10**p84 - 10**best))
        else:
            return best, err


def reasonably_close(a, b, pct_err):
    return np.sqrt(np.sum((a - b)**2) / np.sum(a**2)) < (pct_err / 100)


def save_fits_gz(fn, data, hdr):
    if os.path.isfile(fn + '.gz'):
        os.remove(fn + '.gz')
    if data.dtype == bool:
        fits.writeto(fn, data.astype(int), hdr, overwrite=True)
    else:
        fits.writeto(fn, data, hdr, overwrite=True)
    # os.system("gzip -f " + fn)
    with open(fn, 'rb') as f_in:
        with gzip.open(fn + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(fn)


def error_msg(fn, msg):
    file = open(fn, 'a')
    file.write('\n' + msg + '\n')
    file.close()
