import numpy as np
import astropy.units as u
from astropy.constants import c, h, k_B
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])
nu = (c / wl / u.um).to(u.Hz)
nun = nu.value
hkB_KHz = (h / k_B).to(u.K / u.Hz).value
B_const = 2e20 * (h / c**2).to(u.J * u.s**3 / u.m**2).value
c_ums = c.to(u.um / u.s).value
MBB_const = 2.0891E-4
MWT = 18.0
WDT = 40.0
logUmax = 7.0
Umax = 10**logUmax

"""
In order in increase the speed, I replace all B() with B_fast(),
which cancels all the unit calculation in B() but becomes unit limited

the same for freq(Hz) and wl(um) in the related functions
"""


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
                ).to(u.MJy).value


def B_fast(T, freq=nun):
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


def PowerLaw(wl, Meff, alpha, gamma, logUmin, logUmax=7.0, beta=2.0,
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
