try:
    target
except NameError:
    target = 'NGC5728'
    all_sdm = {'NGC5728': ['14A-468.sb29639312.eb29693960.56920.90377532407',
                           '14B-396.sb30126560.eb30158093.57038.48830790509',
                           '14B-396.sb30126560.eb30168231.57040.471906631945']}

# Just for HI
restfreq='1.420405752GHz'

# Need to vary through samples
# Need some way to automatically assign these values
glx_ctr = {'NGC5728': 'J2000 14h42m23.9 -17d15m11'}
start_vel = {'NGC5728': None}
nchan = {'NGC5728': 128}
include_chan='0~1600;2700~4095'

# Might Change
weighting = 'natural'
robust = 0.5
cellsize = '0.5arcsec'
scales=[0, 5, 25]
imsize = [2048, 2048]

""" File names """
mss = [SDM_name + 'spw_' + few_spws + '.ms' for SDM_name in all_sdm[target]]]
trial_name = '_trial01'
final_vis = target + '.ms'
imname = trial_name + '.cube'
m0name = trial_name + '.m0'
m1name = trial_name + '.m1'

""" Continuum subtraction & split + regrid"""
print("...Subtracting continuum")
for ms in mss:
    uvcontsub(vis=ms, fitspw=include_chan, want_cont=True, intent='OBS*')
print("...Concating ms files")
concat(vis=[ms + '.contsub' for ms in mss],
       concatvis=target + '.concat.ms')
print("...Performing mstransform")
mstransform(vis=target + '.concat.ms',
            outputvis=final_vis,
            regridms=True,
            mode='velocity',
            nchan=nchan[target],
            start=start_vel[target],
            width='2.5km/s',
            restfreq=restfreq,
            outframe='LSRK',
            veltype='radio')
os.remove(target + '.concat.ms')
for ms in mss:
    os.remove(ms + '.contsub')

""" Imaging """
# Making clean masks
print("...Making a dirty image")
tclean(vis=final_vis,
       imagename=imname,
       interpolation='linear',
       restfreq=restfreq,
       weighting=weighting,
       robust=robust,
       interactive=False,
       threshold='0mJy',
       niter=0,
       specmode='cube',
       gridder='mosaic',
       cell=cellsize,
       imsize=imsize,
       outframe='LSRK',
       veltype='radio',
       usemask='pb',
       pbmask=0.2)
os.remove(dirtyname + '.image')
print("...Calculating threshold")
immath(imagename=imname + '.pb',
       outfile=imname + '.temp_pb.image',
       expr='iff(IM0 > 0.2,1.0,0.0)')
cubestat = imstat(imagename=imname + '.residual',
                  mask=imname + '.temp_pb.image')
cube_rms = cube_stat['medabsdevmed'][0] / 0.6745
mask_thresh = str(3.0 * cube_rms) + "Jy"

# Make real image. This would be the new model column
print("...Proceeding with clean")
tclean(vis=final_vis,
       imagename=imname,
       restfreq=restfreq,
       weighting=weighting,
       robust=robust,
       interactive=False,
       threshold=mask_thresh,
       niter=500000,
       specmode='cube',
       outframe='LSRK',
       veltype='radio',
       deconvolver='multiscale',
       scales=scales,
       imsize=imsize,
       cell=cellsize,
       phasecenter=glx_ctr[target],
       usemask='pb',
       pbmask=0.2,
       gridder='mosaic')

# Smooth
print("...Smoothing to a 4x4 arcsecond beam.")
imsmooth(imagename=imname + 'image',
         outfile=imname + '.formask.image',
         targetres=True,
         major='4.0arcsec', minor='4.0arcsec', pa='0deg',
         overwrite=True)

immoments(imagename=imname + '.image', moments=[0], axis='spectral', outfile=m0name)
immoments(imagename=imname + '.image', moments=[1], axis='spectral', outfile=m1name)
exportfits(imagename=m0name, fitsimage=m0name + '.fits')
exportfits(imagename=m1name, fitsimage=m1name + '.fits')
