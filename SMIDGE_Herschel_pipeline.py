import os
from idc_lib.idc_fitting_JC import fit_dust_density as fdd

os.system('clear')  # on linux / os x
dpath2 = 'data/SMIDGE/'
project_name = 'SMIDGE_Herschel'

ppath = 'Projects/' + project_name + '/'
bands = ['pacs100', 'pacs160', 'spire250', 'spire350', 'spire500']


def mp_fit(o, bands):
    observe_fns = []
    for band in bands:
        observe_fns.append(dpath2 + o + '.HERITAGE.' + band.upper() + '.fits')
    mask_fn = dpath2 + o + '.background.fits'
    #
    fdd(o, method_abbr='FB', del_model=False,
        nop=10, beta_f=2.0, Voronoi=False, save_pdfs=False,
        project_name=project_name, observe_fns=observe_fns,
        mask_fn=mask_fn, subdir='', notes='',
        bands=bands, rand_cube=True, better_bkgcov=None,
        galactic_integrated=False)


if __name__ == "__main__":
    #
    for o in ['SMC']:
        mp_fit(o, bands)
