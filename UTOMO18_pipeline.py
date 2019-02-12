# from generating import generator
import os
import numpy as np
from astropy.io import fits
from idc_lib.idc_fitting import fit_dust_density as fdd
from idc_lib.idc_fitting import bkgcov_UTOMO18, diskmask_UTOMO18
from idc_lib.idc_plot import Dust_Plots
# from astropy.io import fits

"""
lens:
    m31: 5472.6 arcsec. up to 1000 pc
    lmc: 30896 arcsec. up to 500 pc
    smc: 15637.8 arcsec. up to 300 pc
    m33: 5678.77 arcsec. up to 2000 pc
"""

os.system('clear')  # on linux / os x

diskmask_generating = 0
bkgcov_generating = 0

fitting = 1
fitting_integrated = 0
fitting_smith12 = 0
plotting = 0

corner_plot = 0
example_models = 0
color_vs_temp = 0
integrated_model = 1

all_resolution = 0
selected_resolution = ['res_13pc']
resolution_dict = \
    {'lmc': ['res_13pc', 'res_30pc', 'res_50pc', 'res_100pc', 'res_167pc',
             'res_300pc', 'res_500pc'],
     'smc': ['res_13pc', 'res_30pc', 'res_50pc', 'res_100pc', 'res_167pc',
             'res_300pc', 'res_500pc'],
     'm31': ['res_167pc', 'res_300pc', 'res_500pc', 'res_1000pc',
             'res_2000pc'],
     'm33': ['res_167pc', 'res_300pc', 'res_500pc', 'res_1000pc',
             'res_2000pc']}
resolution_good_dict = \
    {'lmc': ['res_13pc', 'res_30pc', 'res_50pc', 'res_100pc', 'res_167pc',
             'res_300pc', 'res_500pc'],
     'smc': ['res_13pc', 'res_30pc', 'res_50pc', 'res_100pc', 'res_167pc',
             'res_300pc'],
     'm31': ['res_167pc', 'res_300pc', 'res_500pc', 'res_1000pc'],
     'm33': ['res_167pc', 'res_300pc', 'res_500pc', 'res_1000pc',
             'res_2000pc']}

all_objects = 0
selected_objects = ['lmc', 'smc']
if not all_objects:
    color_vs_temp = 0

default_bands = 1
selected_bands = ['pacs100', 'pacs160', 'spire250', 'spire350', 'spire500']

diagnostic = 0
nickname = selected_objects[0] + '_' + selected_resolution[0]


project_name = 'UTOMO18'
datapath = 'data/UTOMO18_dust/'
objects = ['m31', 'lmc', 'smc', 'm33'] if all_objects else selected_objects
bands = ['pacs100', 'pacs160', 'spire250', 'spire350', 'spire500'] if \
    default_bands else selected_bands
model = 'FB' 
beta_f = 1.8
nop = 20 

if __name__ == "__main__":
    #
    if diskmask_generating:
        for object_ in objects:
            if all_resolution:
                resolutions = resolution_dict[object_]
            else:
                resolutions = selected_resolution
            diskmask_UTOMO18(object_, resolutions)
    if bkgcov_generating:
        for object_ in objects:
            if all_resolution:
                resolutions = resolution_dict[object_]
            else:
                resolutions = selected_resolution
            bkgcov_UTOMO18(object_,
                           res_good=resolution_good_dict[object_],
                           res_all=resolutions)
    if fitting:
        for object_ in objects:
            object_path = datapath + object_
            if all_resolution:
                resolutions = resolution_dict[object_]
            else:
                resolutions = selected_resolution
            for resolution in resolutions:
                filepath = object_path + '/' + resolution
                if not os.path.isdir(filepath):
                    continue
                """
                bkgcov_in_df = fits.getdata(filepath + '/' + object_ +
                                            '_bkgcov_df.fits.gz')[::-1, ::-1]
                bkgcov_in_fit = fits.getdata(filepath + '/' + object_ +
                                             '_bkgcov.fits')
                std_fit = np.sqrt(np.diagonal(bkgcov_in_fit))
                std_df = np.sqrt(np.diagonal(bkgcov_in_df))
                bkgcov_in = np.zeros([5, 5])
                corr = np.copy(bkgcov_in_df)
                for i in range(5):
                    for j in range(5):
                        corr[i, j] /= (std_df[i] * std_df[j])
                        bkgcov_in[i, j] = corr[i, j] * std_fit[i] * std_fit[j]
                """
                bkgcov_in = None
                #
                filelist = os.listdir(filepath)
                observe_fns = []
                notes = ''
                for band in bands:
                    for fn in filelist:
                        temp = fn.split('_')
                        if len(temp) < 4:
                            continue
                        if (temp[-4] == band) and \
                                (temp[-1] != 'mask.fits') and \
                                (temp[-1] != 'mask.smaller.fits') and \
                                (temp[-1] != 'mask.bigger.fits'):
                            observe_fns.append(filepath + '/' + fn)
                            if notes == '':
                                notes = ('_'.join(temp[-3:])).strip('.fits')
                            break
                #
                mask_fn = filepath + '/' + object_ + '_diskmask.smaller.fits' 
                if not os.path.isfile(mask_fn):
                    diskmask_UTOMO18(object_, resolutions)
                #
                # Start input bkgcov 
                #
                nwl = len(bands)
                mask_fn2 = filepath + '/' + object_ + '_diskmask.preXmas.fits'
                diskmask, hdr = fits.getdata(mask_fn2, header=True)
                diskmask = diskmask.astype(bool)
                list_shape = list(diskmask.shape)
                sed = np.empty(list_shape + [nwl])
                for i in range(nwl):
                    sed[:, :, i] = fits.getdata(observe_fns[i], header=False)
                non_nanmask = np.all(np.isfinite(sed), axis=-1)
                diskmask = diskmask * non_nanmask
                bkgmask = (~diskmask) * non_nanmask
                # method_abbr: SE, FB, BE, WD, PL
                # implement outlier rejection
                outliermask = np.zeros_like(bkgmask, dtype=bool)
                for i in range(nwl):
                    AD = np.abs(sed[:, :, i] - np.median(sed[bkgmask][i]))
                    MAD = np.median(AD[bkgmask])
                    with np.errstate(invalid='ignore'):
                        outliermask += AD > 3 * MAD
                bkgmask = bkgmask * (~outliermask)
                new_bkgmask = bkgmask * (~outliermask)
                # assert np.sum(bkgmask) > 10
                bkgcov_in = np.cov(sed[new_bkgmask].T)
                #
                # End input bkgcov 
                #
                subdir = resolution if 'mips160' not in bands else \
                    resolution + '_mips160'
                assert len(observe_fns) == 5
                fdd(object_, method_abbr=model, del_model=False,
                    nop=nop, beta_f=beta_f, Voronoi=False, save_pdfs=True,
                    project_name=project_name, observe_fns=observe_fns,
                    mask_fn=mask_fn, subdir=subdir, notes=notes,
                    bands=bands, rand_cube=True,
                    better_bkgcov=bkgcov_in)
    #
    if fitting_integrated:
        for object_ in objects:
            if all_resolution:
                resolutions = resolution_dict[object_]
            else:
                resolutions = selected_resolution
            object_path = datapath + object_
            """
            if object_ in ['m31', 'm33']:
                resolutions = ['res_167pc']
            elif object_ in ['lmc', 'smc']:
                resolutions = ['res_13pc']
            """
            for resolution in resolutions:
                filepath = object_path + '/' + resolution
                if not os.path.isdir(filepath):
                    continue
                bkgcov_in = np.zeros([5, 5], dtype=float)
                #
                filelist = os.listdir(filepath)
                observe_fns = []
                notes = ''
                for band in bands:
                    for fn in filelist:
                        temp = fn.split('_')
                        if len(temp) < 4:
                            continue
                        if (temp[-4] == band) and \
                                (temp[-1] != 'mask.fits') and \
                                (temp[-1] != 'mask.smaller.fits') and \
                                (temp[-1] != 'mask.bigger.fits'):
                            observe_fns.append(filepath + '/' + fn)
                            if notes == '':
                                notes = ('_'.join(temp[-3:])).strip('.fits')
                            break
                subdir = 'integrated_' + resolution
                assert len(observe_fns) == 5
                mask_fn = filepath + '/' + object_ + '_diskmask.fits'
                if not os.path.isfile(mask_fn):
                    diskmask_UTOMO18(object_, resolutions)
                fdd(object_, method_abbr=model, del_model=False,
                    nop=1, beta_f=beta_f, Voronoi=False, save_pdfs=False,
                    project_name=project_name, observe_fns=observe_fns,
                    mask_fn=mask_fn, subdir=subdir, notes=notes,
                    bands=bands, rand_cube=True, galactic_integrated=True,
                    better_bkgcov=bkgcov_in)
    #
    """
    if fitting_smith12:
        object_ = 'm31'
        object_path = datapath + object_
        resolutions = ['res_167pc']
        for resolution in resolutions:
            filepath = object_path + '/' + resolution
            if not os.path.isdir(filepath):
                continue
            bkgcov_in = fits.getdata(filepath + '/' + object_ +
                                     '_bkgcov.fits')
            beta_in = fits.getdata(filepath + '/' + object_ +
                                   '_beta.fits')
            #
            filelist = os.listdir(filepath)
            observe_fns = []
            notes = ''
            for band in bands:
                for fn in filelist:
                    temp = fn.split('_')
                    if len(temp) < 4:
                        continue
                    if (temp[-4] == band) and \
                            (temp[-1] != 'mask.fits'):
                        observe_fns.append(filepath + '/' + fn)
                        if notes == '':
                            notes = ('_'.join(temp[-3:])).strip('.fits')
                        break
            #
            mask_fn = filepath + '/' + object_ + '_diskmask.fits'
            if not os.path.isfile(mask_fn):
                diskmask_UTOMO18(object_, resolutions)
            #
            subdir = resolution if 'mips160' not in bands else \
                resolution + '_mips160'
            assert len(observe_fns) == 5
            fdd(object_, method_abbr=model, del_model=False,
                nop=nop, beta_f=2012, Voronoi=False, save_pdfs=False,
                project_name=project_name, observe_fns=observe_fns,
                mask_fn=mask_fn, subdir=subdir, notes=notes,
                bands=bands, rand_cube=True, better_bkgcov=bkgcov_in,
                import_beta=True, beta_in=beta_in)
    """
    #
    if plotting:
        plots = Dust_Plots(TRUE_FOR_PNG=False)
        for object_ in objects:
            savepath = 'Projects/' + project_name + '/'
            object_path = savepath + object_
            if all_resolution:
                resolutions = os.listdir(object_path)
                if 'desktop.ini' in resolutions:
                    resolutions.remove('desktop.ini')
            else:
                resolutions = selected_resolution
            for resolution in resolutions:
                temp_res = resolution
                bands = ['pacs100', 'pacs160', 'spire250', 'spire350',
                         'spire500']
                temp = resolution.split('_')
                if temp[-1] == 'mips160':
                    bands[1] = 'mips160'
                    temp_res = '_'.join(temp[:-1])
                nickname = object_ + '_' + resolution
                filepath = object_path + '/' + resolution
                """
                plots.New_Dataset(nickname=nickname,
                                  object_=object_,
                                  model=model,
                                  beta_f=beta_f,
                                  project_name=project_name,
                                  filepath=filepath,
                                  datapath=datapath + object_ + '/' +
                                  temp_res + '/',
                                  bands=bands)
                """
                if corner_plot:
                    plots.corner_plots(nickname, plot_chi2=True)
                if example_models:
                    plots.example_model(nickname, num=10)
        if color_vs_temp:
            common_resolutions = ['_res_167pc', '_res_300pc', '_res_500pc',
                                  '_res_1000pc']
            for cr in common_resolutions:
                plots.color_vs_temperature([object_ + cr for object_ in
                                            objects])
        if diagnostic:
            plots.diagnostic_interactive(nickname)
        if integrated_model:
            plots.integrated_model_FBPL()
            pass
