# from generating import generator
import os
from idc_lib.idc_fitting import fit_dust_density as fdd
from idc_lib.idc_plot import Dust_Plots

os.system('clear')  # on linux / os x


fitting = 0
plotting = 1


project_name = 'UTOMO18'
datapath = 'data/UTOMO18_dust/'
objects = ['m31', 'lmc', 'smc', 'm33']
lastname = {'smc': 'mwsub.fits', 'lmc': 'mwsub.fits',
            'm31': 'bgsub.fits', 'm33': 'bgsub.fits'}
bands = ['pacs100', 'pacs160', 'spire250', 'spire350', 'spire500']
model = 'FB'
beta_f = 1.8
nop = 10

objects = ['m33'] 
bands = ['pacs100', 'mips160', 'spire250', 'spire350', 'spire500'] 

if __name__ == "__main__":
    #
    if fitting:
        for object_ in objects:
            object_path = datapath + object_
            resolutions = os.listdir(object_path)
            if 'desktop.ini' in resolutions:
                resolutions.remove('desktop.ini')
            resolutions = ['res_260pc'] 
            for resolution in resolutions:
                filepath = object_path + '/' + resolution
                if not os.path.isdir(filepath):
                    continue
                filelist = os.listdir(filepath)
                observe_fns = []
                notes = ''
                for band in bands:
                    for fn in filelist:
                        temp = fn.split('_')
                        if len(temp) < 4:
                            continue
                        if (temp[-4] == band) and \
                                (temp[-1] == lastname[object_]):
                            observe_fns.append(filepath + '/' + fn)
                            if notes == '':
                                notes = ('_'.join(temp[-3:])).strip('.fits')
                            break
                for fn in filelist:
                    temp = fn.split('_')
                    if len(temp) < 4:
                        continue
                    if (temp[-4] == 'pacs100') and (temp[-1] == 'mask.fits'):
                        mask_fn = filepath + '/' + fn
                        break
                assert len(observe_fns) == 5
                subdir = resolution if 'mips160' not in bands else \
                    resolution + '_mips160'
                fdd(object_, method_abbr=model, del_model=False,
                    nop=nop, beta_f=beta_f, Voronoi=False,
                    project_name=project_name, observe_fns=observe_fns,
                    mask_fn=mask_fn, subdir=subdir, notes=notes,
                    bands=bands)
    #
    if plotting:
        plots = Dust_Plots(TRUE_FOR_PNG=True)
        for object_ in objects:
            savepath = 'Projects/' + project_name + '/'
            object_path = savepath + object_
            resolutions = os.listdir(object_path)
            if 'desktop.ini' in resolutions:
                resolutions.remove('desktop.ini')
            # resolutions = ['res_260pc_mips160'] 
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
                plots.New_Dataset(nickname=nickname,
                                  object_=object_,
                                  model=model,
                                  beta_f=beta_f,
                                  project_name=project_name,
                                  filepath=filepath,
                                  datapath=datapath + object_ + '/' +
                                  temp_res + '/',
                                  bands=bands)
                # plots.corner_plots(nickname, plot_chi2=True)
                # plots.example_model(nickname, num=10)
        common_resolutions = ['_res_167pc', '_res_300pc', '_res_500pc',
                              '_res_1000pc']
        # for cr in common_resolutions:
        #     plots.color_vs_temperature([object_ + cr for object_ in objects])
        nickname = 'm33' + '_' + 'res_1000pc'
        plots.diagnostic_interactive(nickname)
