# This script concat the ms files.
# Perform ms transform, and make the final image and moment maps
import os
import sys
import imp
from shutil import rmtree
# from shutil import rmtree
pipepath = '/home/idchiang/VLA/VLA_scripted_pipeline/'
pipepath = '/data/scratch/idchiang/VLA_scripted_pipeline/'
# Import 3rd-party scripts
idcim = imp.load_source('idchiang_image_funcs2',
                        pipepath + 'idchiang_image_funcs2.py')
# Avoid sys package syntax error
print('#\nExecuted with Python ' + sys.version + '\n#\n')

if False:
    msmd, concat, mstransform = 0

# pbmask / pblimit
pbs = {'IC342': [0.15, 0.15],
       'NGC2787': [0.6, 0.5],
       'NGC3227': [0.8, 0.7],
       'NGC3608': [0.5, 0.5],
       'NGC3898': [0.8, 0.7],
       'NGC4365': [0.2, 0.2],
       'NGC4374': [0.9, 0.85],
       'NGC4477': [0.5, 0.5],
       'NGC4494': [0.5, 0.5],
       'NGC4496A': [0.9, 0.85],
       'NGC4501': [0.85, 0.85],
       'NGC4596': [0.5, 0.5],
       'NGC4636': [0.5, 0.5],
       'NGC4649': [0.5, 0.5],
       'NGC5728': [0.5, 0.5]}

###############################################################################
#
# The pipeline starts here
#
###############################################################################

#
# Read the info of the galaxy and list of ms files
#
target, mss, final_vis, glx_ctr = idcim.target_info()
# Grab restfreq
msmd.open(mss[0])
freqs = msmd.chanfreqs(spw=0)
restfreq_Hz_num = (freqs[0] + freqs[-1]) / 2
msmd.close()
restfreq = str(round(restfreq_Hz_num / 1000000, 4)) + 'MHz'
del freqs
print('#\nTarget galaxy: ' + target + '\n' +
      'restfreq: ' + restfreq + '\n' +
      'Phase center: ' + glx_ctr + '\n#\n')
#
for i in range(len(mss)):
    ms_active = mss[i]
    mstransform(vis=ms_active, outputvis=ms_active+'.mstransform',
                mode='velocity', nchan=36, start='-216km/s', width='12km/s',
                restfreq=restfreq, outframe='LSRK', veltype='radio',
                datacolumn='data', regridms=True)
    mss[i] = ms_active+'.mstransform'
#
if os.path.isdir(final_vis):
    print("...Removing previous final vis file")
    rmtree(final_vis)
print("...Concating " + str(len(mss)) + " ms files for " +
      target)
concat(vis=mss, concatvis=final_vis)
#
print('#\nFinal vis name: ' + final_vis + '\n#\n')

#
# Calculate image dimensions
#
cellsize_str, cellsize_arcsec, imsize = \
    idcim.image_dimensions(vis=final_vis,
                           oversamplerate=5)
print('#\nCalculated cell size: ' + cellsize_str + '\n' +
      'Image size: [' + str(imsize[0]) + ', ' + str(imsize[1]) + ']\n')

#
# Begin imageing
#
if target != 'IC342':
    glx_ctr = ''
    scales = [0, 5, 20, 70, 200]
else:
    scales = [0, 5, 20, 70]
try:
    pbmask, pblimit = pbs[target]
except KeyError:
    pbmask, pblimit = 0.4, 0.4
#
idcim.imaging(vis=final_vis,
              trial_name=target + '.trial01',
              interactive=False,
              # Image dimension
              imsize=imsize,
              cellsize=cellsize_str,
              glx_ctr=glx_ctr,
              restfreq=restfreq,
              specmode='cubedata',
              outframe='LSRK',
              veltype='radio',
              # Restore to common beam?
              restoringbeam='common',
              # Weighting
              weighting='briggs',
              robust=0.5,
              # Methods
              scales=scales,
              gain=0.1,
              smallscalebias=0.6,
              dogrowprune=False,
              growiterations=0,
              noisethreshold=2.0,  # Same as Dyas
              minbeamfrac=3.0,  # Same as Dyas
              sidelobethreshold=1.0,
              gridder='mosaic',
              pbmask=pbmask,
              pblimit=pblimit,
              # Stopping criteria
              threshold='1.0mJy',
              niter_in=5000000,  # don't want to hit
              nsigma=2.5,  # 1.0 too deep, 3.0 with cyclefactor 1.0 too shallow
              cyclefactor=1.0,
              minpsffraction=0.2)