import numpy as np
import pandas as pd
from astropy.constants import c
import astropy.units as u


c = c.to(u.um/u.s).value


def Gordon_RSRF():
    """
    Read raw files
    """
    surveys = ['SPIRE_500', 'SPIRE_350', 'SPIRE_250', 'PACS_160', 'PACS_100']
    SPIRE_RSRF, PACS_RSRF = 0, 0
    for survey in surveys:
        if survey in ['SPIRE_500', 'SPIRE_350', 'SPIRE_250']:
            filename = 'data/RSRF/Gordon/' + survey.replace('_', '') + \
                       '_resp_ext.dat'
            data = pd.read_csv(filename, skiprows=3, header=None)
            data.columns = ['Raw']
            for i in range(len(data)):
                temp = data.iloc[i]['Raw'].strip(' ').split(' ')
                data.set_value(i, 'Wavelength', float(temp[0]))
                data.set_value(i, survey, float(temp[-1]))
            del data['Raw']
            if type(SPIRE_RSRF) == int:
                SPIRE_RSRF = data
            else:
                SPIRE_RSRF = SPIRE_RSRF.merge(data, how='inner',
                                              on='Wavelength')
        elif survey in ['PACS_160', 'PACS_100']:
            filename = 'data/RSRF/Gordon/' + survey.replace('_', '') + \
                       '_resp.dat'
            data = pd.read_csv(filename, skiprows=2, header=None)
            data.columns = ['Raw']
            for i in range(len(data)):
                temp = data.iloc[i]['Raw'].strip(' ').split(' ')
                data.set_value(i, 'Wavelength', float(temp[0]))
                data.set_value(i, survey, float(temp[-1]))
            del data['Raw']
            if type(PACS_RSRF) == int:
                PACS_RSRF = data
            else:
                PACS_RSRF = PACS_RSRF.merge(data, how='inner', on='Wavelength')
    """
    Create d\nu column
    """
    nu = c / SPIRE_RSRF['Wavelength'].values
    SPIRE_RSRF['dnu'] = np.mean(nu[1:] - nu[:-1])
    #
    nu = c / PACS_RSRF['Wavelength'].values
    PACS_RSRF['dnu'] = np.mean(nu[1:] - nu[:-1])
    """
    Delete empty rows
    """
    mask = np.ones(len(SPIRE_RSRF)).astype(bool)
    for i in range(len(SPIRE_RSRF)):
        temp = SPIRE_RSRF.iloc[i]
        mask[i] = (temp['SPIRE_500'] != 0) + (temp['SPIRE_350'] != 0) + \
            (temp['SPIRE_250'] != 0)
    SPIRE_RSRF = SPIRE_RSRF[mask]
    mask = np.ones(len(PACS_RSRF)).astype(bool)
    for i in range(len(PACS_RSRF)):
        temp = PACS_RSRF.iloc[i]
        mask[i] = (temp['PACS_160'] != 0) + (temp['PACS_100'] != 0)
    PACS_RSRF = PACS_RSRF[mask]
    """
    Save to files
    """
    SPIRE_RSRF.to_csv("data/RSRF/SPIRE_RSRF.csv")
    PACS_RSRF.to_csv("data/RSRF/PACS_RSRF.csv")
