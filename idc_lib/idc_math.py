from __future__ import absolute_import, division, print_function, \
                       unicode_literals
import numpy as np
range = xrange


FWHM2sigma = 0.5 / np.sqrt(2*np.log(2))


def Gaussian_rot(x_in, y_in, x_c, y_c, bpa, bmaj, bmin, ps):
    """
    Inputs:
        x_in, y_in: <numpy array of int>
            The grid points if output PSF. Shape (l, l).
        x_c, y_c: <float>
            Central pixel of the image.
        bpa: <float>
            Position angle of the Gaussian in degrees.
        bmaj, bmin: <float>
            y and x axis FWHM of the Gaussian in arcsec.
        ps: <list of float>
            Pixel scale of the PSF in arcsec. Shape (2).
    Outputs:
        Gaussian_PSF: <numpy array of float>
            The generated PSF. Shape [l, l]
    """
    x = (x_in - x_c) * ps[0]
    y = (y_in - y_c) * ps[1]
    bpa = bpa * np.pi / 180
    cosbpa = np.cos(bpa)
    sinbpa = np.sin(bpa)
    bmaj *= FWHM2sigma
    bmin *= FWHM2sigma
    a = cosbpa**2 / 2 / bmin**2 + sinbpa**2 / 2 / bmaj**2
    b = (1 / bmin**2 - 1 / bmaj**2) * np.sin(2 * bpa) / 4
    d = sinbpa**2 / 2 / bmin**2 + cosbpa**2 / 2 / bmaj**2
    return np.exp(-(a * x**2 + 2 * b * x * y + d * y**2))


def Gaussian_Kernel_C1(ps, bpa, bmaj, bmin, FWHM=25):
    """
    Inputs:
        ps: <numpy array of float>
            Pixel scale of the Kernel in arcsec. Shape [2]
        bpa: <float>
            Position angle of the first Gaussian in degrees.
        bmaj, bmin: <float>
            y and x axis FWHM of the first Gaussian in arcsec.
        FWHM: <float>
            FWHM of the second Gaussian in arcsec.
    Outputs:
        Gaussian_Kernel_C1: <numpy array of float>
        The generated Kernel with pixel scale ps.
    """
    # Converting scales
    bpa *= np.pi / 180
    sigma_x_sq = (FWHM**2 - bmin**2) * FWHM2sigma**2 / ps[0]**2
    sigma_y_sq = (FWHM**2 - bmaj**2) * FWHM2sigma**2 / ps[1]**2
    # Generating grid points. Ref Anaino total dimension ~729", half 364.5"
    lx, ly = 364.5 // ps[0], 364.5 // ps[1]
    x, y = np.meshgrid(np.arange(-lx, lx + 1), np.arange(-ly, ly + 1))
    cosbpa, sinbpa = np.cos(bpa), np.sin(bpa)
    xp, yp = cosbpa * x + sinbpa * y, cosbpa * y - sinbpa * x
    result = np.exp(-0.5 * (xp**2 / sigma_x_sq + yp**2 / sigma_y_sq))
    return result / np.sum(result)


def reasonably_close(a, b, pct_err):
    return np.sqrt(np.sum((a - b)**2) / np.sum(a**2)) < (pct_err / 100)
