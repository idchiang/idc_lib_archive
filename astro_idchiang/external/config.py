import numpy as np
import os


if os.getcwd().startswith('/Users/alexialewis/'):
    #_GALBASE_DIR = '/Users/alexialewis/research/galbase/'
    #_GALDATA_DIR = '/Users/alexialewis/research/galbase/galbase/gal_data/'
    _GALBASE_DIR = '/Users/alexialewis/python/akl_galbase/gal_data'
    _GALDATA_DIR = _GALBASE_DIR
elif os.getcwd().startswith('C:\\Users\\jiang'):
    this_dir, this_filename = os.path.split(__file__)
    _GALBASE_DIR = os.path.join(this_dir, "gal_data")
    _GALDATA_DIR = os.path.join(this_dir, "gal_data")
else:
    _GALBASE_DIR = '/home/maury/leroy.42/idl/galbase/gal_data/'
    _GALDATA_DIR = _GALBASE_DIR


PROP_ARRAY = np.asarray([['name', object, ''],
              ['pgc', int, 0],
              ['alias', object, ''],
              ['tags', object, ''],
              ['dist_mpc', float, np.nan],
              ['e_dist', float, np.nan],
              ['ref_edist', object, 'LEDA'],
              ['vhel_kms', float, np.nan],
              ['e_vhel_kms', float, np.nan],
              ['ref_vhel', object, 'LEDA'],
              ['vrad_kms', float, np.nan],
              ['e_vrad_kms', float, np.nan],
              ['ref_vrad', object, 'LEDA'],
              ['vvir_kms', float, np.nan],
              ['e_vvir_kms', float, np.nan],
              ['ref_vvir', object, 'LEDA'],
              ['ra_deg', float, np.nan],
              ['dec_deg', float, np.nan],
              ['ref_pos', object, 'LEDA'],
              ['posang_deg', float, np.nan],
              ['e_posang', float, np.nan],
              ['ref_posang', object, 'LEDA'],
              ['incl_deg', float, np.nan],
              ['e_incl', float, np.nan],
              ['ref_incl', object, 'LEDA'],
              ['log_raxis', float, np.nan],
              ['e_log_raxis', float, np.nan],
              ['ref_log_raxis', object, 'LEDA'],
              ['t', float, np.nan],
              ['e_t', float, np.nan],
              ['ref_t', object, 'LEDA'],
              ['morph', object, ''],
              ['bar', int, 0],
              ['ring', int, 0],
              ['multiple', int, 0],
              ['ref_morph', object, 'LEDA'],
              ['av_sf11', float, np.nan],
              ['r25_deg', float, np.nan],
              ['e_r25_deg', float, np.nan],
              ['ref_r25', object, 'LEDA'],
              ['vmaxg_kms', float, np.nan],
              ['e_vmaxg_kms', float, np.nan],
              ['ref_vmaxg', object, 'LEDA'],
              ['vrot_kms', float, np.nan],
              ['e_vrot_kms', float, np.nan],
              ['ref_vrot', object, 'LEDA'],
              ['hi_msun', float, np.nan],
              ['ref_hi', object, 'LEDA'],
              ['lfir_lsun', float, np.nan],
              ['ref_ir', object, 'LEDA'],
              ['btcmag', float, np.nan],
              ['ref_btc', object, 'LEDA'],
              ['ubtc_mag', float, np.nan],
              ['ref_ubtc', object, 'LEDA'],
              ['bvtc_mag', float, np.nan],
              ['ref_bvtc', object, 'LEDA'],
              ['itc_mag', float, np.nan],
              ['ref_itc', object, 'LEDA']])


INIT_VALS = PROP_ARRAY[:,2]
COLUMNS = PROP_ARRAY[:,0]
COL_TYPES = PROP_ARRAY[:,1]
