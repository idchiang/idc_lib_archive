from h5py import File
import numpy as np
from .idc_functions import list2bin
from .idc_fitting import cali_mat2


def fake_generation(name='NGC5457', beta_f=2.0, lambda_c_f=300.0,
                    method_abbr='SE', del_fake=True):
    ndims = {'SE': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4, 'WD': 3,
             'BE': 3, 'PL': 4}
    try:
        ndims[method_abbr]
    except KeyError:
        raise KeyError("The input method \"" + method_abbr +
                       "\" is not supported yet!")
    print('')
    print('################################################')
    print('Calculating ' + name + '-' + method_abbr + ' fake observation')
    print('################################################')
    with File('hdf5_MBBDust/' + name + '.h5', 'r') as hf:
        grp = hf['Regrid']
        sed_unc = grp['HERSCHEL_011111_UNCMAP'].value
        bkgcov = grp['HERSCHEL_011111_BKGCOV'].value
        #
        grp = hf['Bin']
        binlist = grp['BINLIST'].value
        binmap = grp['BINMAP'].value
        #
        grp = hf['Fitting_results']
        subgrp = grp[method_abbr]
        aSED = subgrp['Best_fit_sed'].value
    #
    SED = np.full_like(sed_unc, np.nan)
    for i in range(5):
        SED[:, :, i] = list2bin(aSED[:, i], binlist, binmap)
    #
    fake_SED = np.full_like(sed_unc, np.nan)
    x, y, _ = fake_SED.shape
    for i in range(x):
        for j in range(y):
            if np.isnan(SED[i, j, 0]):
                continue
            sed_vec = SED[i, j].reshape(1, 5)
            unc2cov = np.identity(5) * sed_unc[i, j]**2
            calcov = sed_vec.T * cali_mat2 * sed_vec
            cov = bkgcov + unc2cov + calcov
            fake_SED[i, j] = np.random.multivariate_normal(SED[i, j], cov)
    #
    with File('hdf5_MBBDust/' + name + '.h5', 'a') as hf:
        grp = hf.require_group('Fake')
        subgrp = grp.require_group('SED')
        try:
            del subgrp[method_abbr]
        except KeyError:
            pass
        subgrp[method_abbr] = fake_SED
