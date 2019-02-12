import tarfile
import os
import sys
import time
# import imp
import numpy as np
from shutil import rmtree

just_for_debug = False
if just_for_debug:
    object_, file_no, pipepath, casa, execfile, casalogger, casalog, split = 0
    flagdata, keyboardException, generalException, pipeline_save = 0
    ms_active, tb, uvcontsub, ms, gencal, default, vishead = 0

start_time = time.localtime()

# Path to pipeline and data. Change them when needed.
# pipepath = '/home/idchiang/VLA/VLA_scripted_pipeline/'
pipepath = '/data/scratch/idchiang/VLA_scripted_pipeline/'
# datapath = '/home/idchiang/Data_EVLA/'

# Variables for reducing .ms size
few_spw_ans = False
few_spws = '8'

# Hanning smooth. Deal with this only in continuum pipeline
myHanning = False

# Remove things in the final version
remove_sdm = True
remove_original_ms = False

# Project information
projectCode = '14A-468;14B-396;16A-275;17A-073'
piName = 'Dr. Karin Sandstrom'


######################################################################
#
# Definitions for the pipeline
#
######################################################################
version = "1.4.0"
svnrevision = '11nnn'
date = "2017Mar08"
print('#\nExecuted with Python ' + sys.version + '\n#\n')


def print_time_stamp():
    # Print the current time
    print('# Current time: ' +
          time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    print('# Executing ' + os.getcwd() + '\n#')


# version_check()
print('####################################################')
print('# Executing Phase01: EVLA calibration pipeline')
print('####################################################')
#
# Read file name and decide what to do.
#
tgzname = None
SDM_name = None
msname = None

# filelist = os.listdir(datapath + '/' + object_ + '/' + file_no)
filelist = os.listdir(os.getcwd())
for fn in filelist:
    temp = fn.split('.')
    if len(temp) == 5:
        SDM_name = fn
    elif len(temp) == 6:
        msname = fn
        if SDM_name is None:
            SDM_name = msname[:-3]
    elif (len(temp) == 8) and (temp[-2:] == ['tar', 'gz']):
        tgzname = fn
        if SDM_name is None:
            SDM_name = '.'.join(temp[:5])

assert SDM_name is not None
print_time_stamp()

if not os.path.isdir(SDM_name):
    assert tgzname is not None
    with tarfile.open(tgzname) as tar:
        print('#')
        print('# tar: SDM file not found. Start untar ' + tgzname)
        tar.extractall()
        print('# tar: Finished untar ' + tgzname)
        print('#')
        print_time_stamp()

# This is the default time-stamped casa log file, in case we
#     need to return to it at any point in the script
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
#
try:
    maincasalog = casalogger.func_globals['theLogFile']  # 5.3+
except KeyError:
    maincasalog = casalogger.func_globals['thelogfile']  # 5.0


def logprint(msg, logfileout=maincasalog):
    print(msg)
    casalog.setlogfile(logfileout)
    casalog.post(msg)
    casalog.setlogfile(maincasalog)
    casalog.post(msg)
    return


# Create timing profile list
if 'time_list' in globals():
    del time_list
time_list = []

timing_file = 'logs/timing.log'
timelog = open(timing_file, 'a')


def runtiming(pipestate, status):
    # Determine profile for a given state/stage of the pipeline
    time_list.append({'pipestate': pipestate, 'time': time.time(),
                      'status': status})
    if (status == "end"):
        timelog = open(timing_file, 'a')
        timelog.write(pipestate + ': ' +
                      str(time_list[-1]['time'] - time_list[-2]['time']) +
                      ' sec \n')
        timelog.flush()
        timelog.close()
        # with open(maincasalog, 'a') as casalogfile:
        #     tempfile = open('logs/'+pipestate+'.log','r')
        #     casalogfile.write(tempfile.read())
        #     tempfile.close()
        # casalogfile.close()
    return time_list


######################################################################
# The following script includes all the definitions and functions and
# prior inputs needed by a run of the pipeline.
time_list = runtiming('startup', 'start')
execfile(pipepath + 'EVLA_pipe_startup.py')
time_list = runtiming('startup', 'end')
pipeline_save()
######################################################################

# IMPORT THE DATA TO CASA
execfile(pipepath + 'EVLA_pipe_import.py')  # modified to importasdm
print('#')

# REMOVE SDM FILE TO REDUCE DISK SPACE
if remove_sdm:
    print('Removing the SDM file.')
    rmtree(SDM_name)
print('#')

# MOVE THIS HERE DUE TO HANNING
logprint('Checking if rq caltable is required')
ms.open(msname)
ms_summary = ms.summary()
ms.close()
startdate = float(ms_summary['BeginTime'])
if startdate >= 55616.6:
    default(gencal)
    vis = msname
    caltable = 'requantizergains.g'
    caltype = 'rq'
    spw = ''
    antenna = ''
    pol = ''
    parameter = []
    gencal()
print('#')

# HANNING SMOOTH (OPTIONAL, MAY BE IMPORTANT IF THERE IS NARROWBAND
# RFI)
if myHanning:
    execfile(pipepath + 'EVLA_pipe_hanning.py')
    print('#')

# GET SOME INFORMATION FROM THE MS THAT WILL BE NEEDED LATER, LIST
#     THE DATA, AND MAKE SOME PLOTS
execfile(pipepath + 'EVLA_pipe_msinfo.py')
print('#')

# QUACK
execfile(pipepath + 'idchiang_HI_flagquack.py')
print('#')

# DETERMINISTIC FLAGGING:
# TIME-BASED: online flags, shadowed data, zeroes, pointing scans,
# quacking
# CHANNEL-BASED: end 5% of channels of each spw, 10 end channels at
# edges of basebands
execfile(pipepath + 'EVLA_pipe_flagall.py')
print('#')

# ONLY KEEP THE SPW WE NEED

if few_spw_ans:
    new_ms = ms_active.strip('.ms') + '.reduced.ms'
    print('Reducing ms filesize.')
    split(ms_active, outputvis=new_ms, spw=few_spws, datacolumn='data')
    if remove_original_ms:
        print('Removing the complete ms file.')
        rmtree(ms_active)
        rmtree(ms_active + '.flagversions')
    ms_active = new_ms
    msname = new_ms
    print('Re-run msinfo.')
    execfile(pipepath + 'EVLA_pipe_msinfo.py')
    print('#')

# REMOVE CHANNELS POLUTED BY MW HI
tb.open(ms_active + '/SPECTRAL_WINDOW')
if few_spw_ans:
    freqs = tb.getcol('CHAN_FREQ').reshape(-1)
else:
    freqs = tb.getcell('CHAN_FREQ', 8)
restfreq = 1420405752
min_freq = restfreq - 1953 * 200
max_freq = restfreq + 1953 * 200
mask = (freqs < max_freq) * (freqs > min_freq)
if np.sum(mask) > 0:
    temp = np.arange(len(mask), dtype=int)
    chan_ids = temp[mask]
    min_chan, max_chan = str(np.min(chan_ids)), str(np.max(chan_ids) + 1)
    if few_spw_ans:
        rm_spw = '*:' + min_chan + '~' + max_chan
    else:
        rm_spw = '8:' + min_chan + '~' + max_chan
    flagdata(vis=ms_active, spw=rm_spw, intent='*CALIBRATE_AMPLI*')
    del rm_spw, chan_ids, temp, min_chan, max_chan
del mask, max_freq, min_freq

#
bandpass_fillgaps_n = 402
min_freq = restfreq - 1953 * 100
max_freq = restfreq + 1953 * 100
mask = (freqs < max_freq) * (freqs > min_freq)
if np.sum(mask) > 0:
    temp = np.arange(len(mask), dtype=int)
    chan_ids = temp[mask]
    min_chan, max_chan = str(np.min(chan_ids)), str(np.max(chan_ids) + 1)
    if few_spw_ans:
        rm_spw = '*:' + min_chan + '~' + max_chan
    else:
        rm_spw = '8:' + min_chan + '~' + max_chan
    flagdata(vis=ms_active, spw=rm_spw, intent='*CALIBRATE_BANDPASS*')
    del rm_spw, chan_ids, temp
del mask, max_freq, min_freq, freqs, restfreq
tb.close()
print('#')
#

# PREPARE FOR CALIBRATIONS
# Fill model columns for primary calibrators
execfile(pipepath + 'EVLA_pipe_calprep.py')
print('#')
# PRIOR CALIBRATIONS
# Gain curves, opacities, antenna position corrections,
# requantizer gains (NB: requires CASA 4.1 or later!).
# Also plots switched
# power tables, but these are not currently used in the calibration

# MANUAL FLAGS
execfile(pipepath + 'idchiang_HI_flagmanual.py')
print('#')

# Switch power moved to above
execfile(pipepath + 'EVLA_pipe_priorcals.py')
print('#')
# INITIAL TEST CALIBRATIONS USING BANDPASS AND DELAY CALIBRATORS
execfile(pipepath + 'EVLA_pipe_testBPdcals.py')
print('#')
# IDENTIFY AND FLAG BASEBANDS WITH BAD DEFORMATTERS OR RFI BASED ON
# BP TABLE AMPS
execfile(pipepath + 'EVLA_pipe_flag_baddeformatters.py')
print('#')
# IDENTIFY AND FLAG BASEBANDS WITH BAD DEFORMATTERS OR RFI BASED ON
# BP TABLE PHASES
execfile(pipepath + 'EVLA_pipe_flag_baddeformattersphase.py')
print('#')
# FLAG POSSIBLE RFI ON BP CALIBRATOR USING RFLAG
execfile(pipepath + 'EVLA_pipe_checkflag.py')
print('#')

# DO SEMI-FINAL DELAY AND BANDPASS CALIBRATIONS
# (semi-final because we have not yet determined the spectral index
# of the bandpass calibrator)
execfile(pipepath + 'EVLA_pipe_semiFinalBPdcals1.py')
print('#')
# Use flagdata again on calibrators
execfile(pipepath + 'EVLA_pipe_checkflag_semiFinal.py')
print('#')

# Improved calibrator flagging on customized script after the semifinals
execfile(pipepath + 'idchiang_HI_flagcali.py')
print('#')

# RE-RUN semiFinalBPdcals.py FOLLOWING rflag
execfile(pipepath + 'EVLA_pipe_semiFinalBPdcals2.py')
print('#')

# Improved calibrator flagging on customized script after the semifinals
execfile(pipepath + 'idchiang_HI_flagcali.py')
print('#')

# DETERMINE SOLINT FOR SCAN-AVERAGE EQUIVALENT
execfile(pipepath + 'EVLA_pipe_solint.py')
print('#')
# DO TEST GAIN CALIBRATIONS TO ESTABLISH SHORT SOLINT
execfile(pipepath + 'EVLA_pipe_testgains.py')
print('#')
# MAKE GAIN TABLE FOR FLUX DENSITY BOOTSTRAPPING
# Make a gain table that includes gain and opacity corrections for
# final amp cal, for flux density bootstrapping
execfile(pipepath + 'EVLA_pipe_fluxgains.py')
print('#')
# FLAG GAIN TABLE PRIOR TO FLUX DENSITY BOOTSTRAPPING
# NB: need to break here to flag the gain table interatively, if
# desired; not included in real-time pipeline
# execfile(pipepath + 'EVLA_pipe_fluxflag.py')

# DO THE FLUX DENSITY BOOTSTRAPPING -- fits spectral index of
# calibrators with a power-law and puts fit in model column
execfile(pipepath + 'EVLA_pipe_fluxboot.py')
print('#')
# MAKE FINAL CALIBRATION TABLES
execfile(pipepath + 'EVLA_pipe_finalcals.py')
print('#')
# APPLY ALL CALIBRATIONS AND CHECK CALIBRATED DATA
execfile(pipepath + 'EVLA_pipe_applycals.py')
print('#')
# NOW RUN ALL CALIBRATED DATA (INCLUDING TARGET) THROUGH rflag
# Note, TARGET is commented out for spectral data
execfile(pipepath + 'EVLA_pipe_targetflag.py')
print('#')

# Improved target flagging on customized script after the final applycal
execfile(pipepath + 'idchiang_HI_flagtarget.py')
print('#')

# CALCULATE DATA WEIGHTS BASED ON ST. DEV. WITHIN EACH SPW
# Note: TARGET is excluded
execfile(pipepath + 'EVLA_pipe_statwt.py')
print('#')
# MAKE FINAL UV PLOTS
execfile(pipepath + 'EVLA_pipe_plotsummary.py')
print('#')
# COLLECT RELEVANT PLOTS AND TABLES
execfile(pipepath + 'EVLA_pipe_filecollect.py')
print('#')
# WRITE WEBLOG
execfile(pipepath + 'EVLA_pipe_weblog.py')
print('#')

# DIAGNOSTIC PLOTS
print("...Plotting extra diagnostic plots")
execfile(pipepath + 'idchiang_HI_diagnostic_plots.py')
print('#')

# CONT SUB
# Not using continuum spws...
if few_spw_ans:
    excspw = '*:'
else:
    excspw = '8:'
try:
    excspw += min_chan + '~' + max_chan + ';'
except NameError:
    pass
#
tb.open(ms_active + '/SPECTRAL_WINDOW')
if few_spw_ans:
    freqs = tb.getcol('CHAN_FREQ').reshape(-1)
else:
    freqs = tb.getcell('CHAN_FREQ', 8)
restfreq = (freqs[0] + freqs[-1]) / 2  # Hz
c_sol = 299792.458
min_freq = restfreq * (c_sol - 300.0) / c_sol
max_freq = restfreq * (c_sol + 300.0) / c_sol
mask = (freqs < max_freq) * (freqs > min_freq)
if np.sum(mask) > 0:
    temp = np.arange(len(mask), dtype=int)
    chan_ids = temp[mask]
    min_chan, max_chan = str(np.min(chan_ids)), str(np.max(chan_ids) + 1)
    excspw += min_chan + '~' + max_chan
tb.close()
print("Starting uvcontsub.")
print("...SPWS to exclude: " + excspw)
if few_spw_ans:
    uvcontsub(vis=ms_active, fitspw=excspw, excludechans=True,
              want_cont=False, fitorder=1)  # only use spw=8
else:
    uvcontsub(vis=ms_active, fitspw=excspw, excludechans=True, spw='8',
              want_cont=False, fitorder=1)  # only use spw=8
print("#")
print("...Splitting and interpolating ms files")
split(vis=ms_active + '.contsub', outputvis=ms_active + '.split',
      intent='*TARGET*', width=4, datacolumn='data')
# set spw name to avoid further warning
temp = vishead(ms_active + '.split', mode='get', hdkey='spw_name')
temp = list(temp[0][0])
temp[-1] = '0'
temp = ''.join(temp)
vishead(ms_active + '.split', mode='put', hdkey='spw_name', hdvalue=temp)
print("#")
print("...Removing contsub ms file")
rmtree(ms_active + '.contsub')
print("#")
#
print("...Split and plotting done. Tar files:")
print("... .flagversions")
source_dir = ms_active + ".flagversions"
if os.path.isdir(source_dir):
    print("...Remove .flagversions")
    rmtree(source_dir)
print("...Tar .ms")
source_dir = ms_active
if os.path.isdir(source_dir):
    with tarfile.open(source_dir + ".tar.gz", "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print("...Remove .ms")
    rmtree(source_dir)
#
print('# Start time: ' +
      time.strftime("%a, %d %b %Y %H:%M:%S", start_time))
print('# End time:   ' +
      time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
