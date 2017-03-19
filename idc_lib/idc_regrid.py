import numpy as np
from scipy.interpolate import interp2d, griddata
from astropy.convolution import convolve_fft
from time import clock
from .idc_math import reasonably_close, Gaussian_Kernel_C1


# If rm_bad_pts becomes less than threshold after convolution, remove.
threshold = 0.9


def Kernel_regrid(kernels, ps_new, name1, name2, target_name=None,
                  method='cubic'):
    """
    Inputs:
        kernels: <pandas DataFrame>
            A DataFrame storaging kernel data and information
        ps_new: <list of float>
            Target pixel scale.
        name1, name2: <str>
            Survey names corresponds to the Kernel needs to be regridded.
        target_name: <str>
            Name of target image. Will be set to name1 one if not entered.
        method: <str>
            Interpolate method. 'linear', 'cubic', 'quintic'
    Outputs:
        n_kernel: <numpy array>
            The regridded kernel array.
    """
    try:
        # Grabbing data
        kernel = kernels.loc[name1, name2].KERNEL
        ps = kernels.loc[name1, name2].PS
        if not target_name:
            target_name = name1
        """
        ps_new = df.xs(target_name, level = 1).PS.values[0]
        """
        # Check if the kernel is squared / odd pixel
        assert kernel.shape[0] == kernel.shape[1]
        assert len(kernel) % 2
        # Generating grid points. Ref Anaino total dimension ~729", half 364.5"
        l = (len(kernel) - 1) // 2
        x = np.arange(-l, l + 1) * ps[0] / ps_new[0]
        y = np.arange(-l, l + 1) * ps[1] / ps_new[1]
        lxn, lyn = (l * ps[0]) // ps_new[0], (l * ps[1]) // ps_new[1]
        xn, yn = np.arange(-lxn, lxn + 1), np.arange(-lyn, lyn + 1)
        # Start binning
        print("Start regridding \"" + name1 + " to " + name2 +
              "\" kernel to match " + target_name + " map...")
        tic = clock()
        k = interp2d(x, y, kernel, kind=method, fill_value=np.nan)
        n_kernel = k(xn, yn)
        n_kernel /= np.sum(n_kernel)
        """
        self.kernels.set_value((name1, name2), 'REGRID', n_kernel)
        self.kernels.set_value((name1, name2), 'PSR', ps_new)
        """
        print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
        print("Kernel sum:", np.sum(n_kernel))
        return n_kernel

    except KeyError:
        print("Warning: kernel or target does not exist.")


def matching_PSF_1step(df, kernels, name, survey1, survey2):
    """
    Inputs:
        df: <pandas DataFrame>
            DataFrame contains map information
        kernels: <pandas DataFrame>
            DataFrame contains kernel information
        name, survey1: <str>
            Object name and survey name to be convolved.
        survey2: <str>
            Name of target PSF.
    Outputs:
        image1_2: <numpy array>
            Convolved image
        ps: <numpy array>
            New pixel scale of kernel
        kernel: <numpy array>
            Regridded kernel
    """
    # Grabbing information
    ps1 = df.xs(survey1, level=1).PS.values[0]
    ps2 = kernels.loc[survey1, survey2].PS
    if reasonably_close(ps1, ps2, 2.0):
        kernel = kernels.loc[survey1, survey2].KERNEL
    else:
        ps2 = kernels.loc[survey1, survey2].RGDPS
        if reasonably_close(ps1, ps2, 2.0):
            kernel = kernels.loc[survey1, survey2].RGDKERNEL
        else:
            ps2 = ps1
            kernel = Kernel_regrid(kernels, ps1, survey1, survey2)

    map0 = df.loc[name, survey1].MAP
    uncmap0 = df.loc[name, survey1].UNCMAP

    ratio = 1E18 if np.nanmax(uncmap0) > 1E18 else 1.0
    uncmap0 /= ratio

    rm_bad_pts = np.full_like(map0, 1)
    rm_bad_pts[np.isnan(map0)] = 0.0
    rm_bad_pts[np.isnan(uncmap0)] = 0.0
    print("Convolving " + name + " " + survey1 + " map (1/1)...")
    tic = clock()
    # Convolve map
    map1 = convolve_fft(map0, kernel, interpolate_nan=False)
    rm_bad_pts = convolve_fft(rm_bad_pts, kernel, interpolate_nan=False)
    map1[np.isnan(map0)] = np.nan
    map1[rm_bad_pts < threshold] = np.nan
    # Convolve uncertainty map
    uncmap1 = convolve_fft(uncmap0 ** 2, kernel, interpolate_nan=False)
    with np.errstate(invalid='ignore'):
        uncmap1 = np.sqrt(uncmap1) * kernels.loc[survey1, survey2].FWHM1 / \
            kernels.loc[survey1, survey2].FWHM2
    uncmap1[np.isnan(uncmap0)] = np.nan
    uncmap1[rm_bad_pts < threshold] = np.nan

    uncmap1 *= ratio
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    map0[rm_bad_pts < threshold] = np.nan
    f1, f2 = np.nansum(map0), np.nansum(map1)
    print("Normalized flux variation:", np.abs(f1 - f2) / f1)
    return map1, uncmap1, ps1, kernel


def matching_PSF_2step(df, kernels, name, survey1, k2_survey1, k2_survey2):
    """
    Inputs:
        df: <pandas DataFrame>
            DataFrame contains map information
        kernels: <pandas DataFrame>
            DataFrame contains kernel information
        name, survey1: <str>
            Object name and survey name to be convolved.
        k2_survey1, k2_survey2: <str>
            Names of second kernel.
    Outputs:
        image1_2: <numpy array>
            Convolved image
        ps2: <numpy array>
            New pixel scale of kernel
        kernel2: <numpy array>
            Regridded kernel
    """
    ps1 = df.xs(survey1, level=1).PS.values[0]
    ps2 = kernels.loc[k2_survey1, k2_survey2].PS
    if reasonably_close(ps1, ps2, 2.0):
        kernel2 = kernels.loc[k2_survey1, k2_survey2].KERNEL
    else:
        ps2 = kernels.loc[k2_survey1, k2_survey2].RGDPS
        if reasonably_close(ps1, ps2, 2.0):
            kernel2 = kernels.loc[k2_survey1, k2_survey2].RGDKERNEL
        else:
            ps2 = ps1
            kernel2 = Kernel_regrid(kernels, ps1, k2_survey1, k2_survey2)

    bpa = df.loc[name, survey1].BPA
    bmaj = df.loc[name, survey1].BMAJ
    bmin = df.loc[name, survey1].BMIN
    map0 = df.loc[name, survey1].MAP
    uncmap0 = df.loc[name, survey1].UNCMAP
    FWHM2 = kernels.loc[k2_survey1, k2_survey2].FWHM1
    rm_bad_pts = np.full_like(map0, 1)
    rm_bad_pts[np.isnan(map0)] = 0.0
    rm_bad_pts[np.isnan(uncmap0)] = 0.0

    ratio = 1E18 if np.nanmax(uncmap0) > 1E18 else 1.0
    uncmap0 /= ratio

    print("Convolving " + name + " " + survey1 + " map (1/2)...")
    tic = clock()
    kernel1 = Gaussian_Kernel_C1(ps1, bpa, bmaj, bmin, FWHM2)
    map1 = convolve_fft(map0, kernel1, interpolate_nan=False)
    rm_bad_pts = convolve_fft(rm_bad_pts, kernel1, interpolate_nan=False)
    map1[np.isnan(map0)] = np.nan
    uncmap1 = convolve_fft(uncmap0 ** 2, kernel1, interpolate_nan=False)
    with np.errstate(invalid='ignore'):
        uncmap1 = np.sqrt(uncmap1) * np.sqrt(bmaj * bmin) / FWHM2
    uncmap1[np.isnan(uncmap0)] = np.nan
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    print("Convolving " + name + " " + survey1 + " map (2/2)...")
    tic = clock()
    map2 = convolve_fft(map1, kernel2, interpolate_nan=False)
    rm_bad_pts = convolve_fft(rm_bad_pts, kernel2, interpolate_nan=False)
    map2[np.isnan(map0)] = np.nan
    map2[rm_bad_pts < threshold] = np.nan
    uncmap2 = convolve_fft(uncmap1 ** 2, kernel2, interpolate_nan=False)
    with np.errstate(invalid='ignore'):
        uncmap2 = np.sqrt(uncmap2) * FWHM2 / \
            kernels.loc[k2_survey1, k2_survey2].FWHM2
    uncmap2[np.isnan(uncmap0)] = np.nan
    uncmap2[rm_bad_pts < threshold] = np.nan

    uncmap2 *= ratio
    print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
    map0[rm_bad_pts < threshold] = np.nan
    map1[rm_bad_pts < threshold] = np.nan
    f1, f2, f3 = np.nansum(map0), np.nansum(map1), np.nansum(map2)
    print("Normalized flux variation. first step: ", np.abs(f1 - f2) / f1)
    print("                           second step:", np.abs(f2 - f3) / f2)
    print("                           overall:    ", np.abs(f1 - f3) / f1)
    return map2, uncmap2, ps2, kernel2


def WCS_congrid(df, name, fine_survey, course_survey, method='linear'):
    """
    Inputs:
        df: <pandas DataFrame>
            DataFrame contains map information
        name, fine_survey: <str>
            Object name and fine survey name to be regrdidded.
        course_survey: <str>
            Course survey name to be regridded.
        method: <str>
            Fitting method. 'linear', 'nearest', 'cubic'
    Outputs:
        image1_1: <numpy array>
            Regridded image
    """
    map0 = df.loc[name, fine_survey].CVL_MAP
    uncmap0 = df.loc[name, fine_survey].CVL_UNC

    ratio = 1E18 if np.nanmax(uncmap0) > 1E18 else 1.0
    uncmap0 /= ratio

    assert map0.shape == uncmap0.shape
    if len(map0) == 1:
        print(name + " " + fine_survey +
              " map has not been convolved. Please convolve first.")
        pass
    else:
        print("Start matching " + name + " " + fine_survey +
              " grid to match " + course_survey + "...")
        tic = clock()
        w1 = df.loc[name, fine_survey].WCS
        naxis12, naxis11 = df.loc[name, fine_survey].L    # RA, Dec
        w2 = df.loc[name, course_survey].WCS
        naxis22, naxis21 = df.loc[name, course_survey].L  # RA, Dec

        xg, yg = np.meshgrid(np.arange(naxis11), np.arange(naxis12))
        xwg, ywg = w1.wcs_pix2world(xg, yg, 1)
        xg, yg = w2.wcs_world2pix(xwg, ywg, 1)
        xng, yng = np.meshgrid(np.arange(naxis21), np.arange(naxis22))

        assert np.size(map0) == np.size(xg) == np.size(yg)
        s = np.size(map0)
        points = np.concatenate((xg.reshape(s, 1), yg.reshape(s, 1)), axis=1)
        map0 = map0.reshape(s)
        uncmap0 = uncmap0.reshape(s)
        map1 = griddata(points, map0, (xng, yng), method=method)
        uncmap1 = griddata(points, uncmap0**2, (xng, yng), method=method)
        uncmap1 = np.sqrt(uncmap1) / ratio
        print("Done. Elapsed time:", round(clock()-tic, 3), "s.")
        return map1, uncmap1
