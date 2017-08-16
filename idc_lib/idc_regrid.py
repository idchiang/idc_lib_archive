import numpy as np
from scipy.interpolate import interp2d
# from scipy.interpolate import griddata
from astropy.convolution import convolve_fft


# If rm_bad_pts becomes less than threshold after convolution, remove.
threshold = 0.9


def Kernel_regrid(kernel, ps, ps_old, method='cubic'):
    # Check if the kernel is squared / odd pixel
    assert kernel.shape[0] == kernel.shape[1]
    assert len(kernel) % 2
    # Generating grid points. Ref Anaino total dimension ~729", half 364.5"
    s = (len(kernel) - 1) // 2
    x = np.arange(-s, s + 1) * ps_old[0] / ps[0]
    y = np.arange(-s, s + 1) * ps_old[1] / ps[1]
    lxn, lyn = (s * ps_old[0]) // ps[0], (s * ps_old[1]) // ps[1]
    xn, yn = np.arange(-lxn, lxn + 1), np.arange(-lyn, lyn + 1)
    # Start binning
    k = interp2d(x, y, kernel, kind=method, fill_value=np.nan)
    n_kernel = k(xn, yn)
    n_kernel /= np.sum(n_kernel)
    print(' --Kernel regrid required. Regrid sum:', np.sum(n_kernel))
    return n_kernel


def matching_PSF(kernel, FWHM1, FWHM2, map0, uncmap0):
    ratio = 1E18 if np.nanmax(uncmap0) > 1E18 else 1.0
    uncmap0 /= ratio
    rm_bad_pts = np.full_like(map0, 1)
    rm_bad_pts[np.isnan(map0)] = 0.0
    rm_bad_pts[np.isnan(uncmap0)] = 0.0
    # Convolve map
    map1 = convolve_fft(map0, kernel, quiet=True,
                        allow_huge=True)
    rm_bad_pts = convolve_fft(rm_bad_pts, kernel, quiet=True,
                              allow_huge=True)
    map1[np.isnan(map0)] = np.nan
    map1[rm_bad_pts < threshold] = np.nan
    # Convolve uncertainty map
    uncmap1 = convolve_fft(uncmap0 ** 2, kernel, quiet=True,
                           allow_huge=True)
    with np.errstate(invalid='ignore'):
        uncmap1 = np.sqrt(uncmap1) * FWHM1 / FWHM2
    uncmap1[np.isnan(uncmap0)] = np.nan
    uncmap1[rm_bad_pts < threshold] = np.nan
    uncmap1 *= ratio

    map0[rm_bad_pts < threshold] = np.nan
    f1, f2 = np.nansum(map0), np.nansum(map1)
    print(" --Normalized flux variation:", round(np.abs(f1 - f2) / f1, 4))
    return map1, uncmap1


"""
def WCS_congrid(map0, uncmap0, w1, w2, l2, method='linear'):
    ratio = 1E18 if np.nanmax(uncmap0) > 1E18 else 1.0
    uncmap0 /= ratio

    naxis12, naxis11 = map0.shape    # RA, Dec
    naxis22, naxis21 = l2  # RA, Dec

    xg, yg = np.meshgrid(np.arange(naxis11), np.arange(naxis12))
    xwg, ywg = w1.wcs_pix2world(xg, yg, 1)
    xg, yg = w2.wcs_world2pix(xwg, ywg, 1)
    xng, yng = np.meshgrid(np.arange(naxis21), np.arange(naxis22))

    assert np.size(map0) == np.size(xg) == np.size(yg)
    s = np.size(map0)
    points = np.concatenate((xg.reshape(s, 1), yg.reshape(s, 1)), axis=1)
    map0, uncmap0 = map0.reshape(s), uncmap0.reshape(s)
    map1 = griddata(points, map0, (xng, yng), method=method)
    uncmap1 = np.sqrt(griddata(points, uncmap0**2, (xng, yng), method=method))
    uncmap1 /= ratio
    return map1, uncmap1
"""
