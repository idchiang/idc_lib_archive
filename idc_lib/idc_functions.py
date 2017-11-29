import numpy as np
import astropy.units as u
from astropy.constants import c, h, k_B
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)


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


# Probability functions & model functions for fitting (internal)
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


def SEMBB(wl, sigma, T, beta):
    freq = (c / wl / u.um).to(u.Hz)
    const = 2.0891E-4
    kappa160 = 9.6 * np.pi
    return const * kappa160 * (160.0 / wl)**beta * sigma * \
        B(T * u.K, freq)


def WD(wl, TOTsigma, T, beta, WDfrac, WDT=40, WDbeta=2.0):
    return SEMBB(wl, TOTsigma * (1 - WDfrac), T, beta) + \
        SEMBB(wl, TOTsigma * WDfrac, WDT, WDbeta)


def BEMBB(wl, sigma, T, beta, lambda_c, beta2):
    """Return fitted SED in MJy"""
    freq = (c / wl / u.um).to(u.Hz)
    const = 2.0891E-4
    kappa160 = 11.6 * np.pi
    ans = B(T * u.K, freq) * const * kappa160 * sigma * 160.0**beta
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
