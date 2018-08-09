import tarfile
import os
import time
from platform import system
from shutil import copytree, rmtree


# Variables for HI problem
rm_Galatic_HI = False
default_rm_spw = '0:2300~2350'

# Variables for reducing .ms size
few_spw_ans = True
few_spws = '8'

# Hanning smooth. Deal with this afterwards.
hanning = False

just_for_debug = system() == 'Windows'
if just_for_debug:
    object_, file_no = 0, 0
    pipepath = 0
    casa = 0
    execfile = 0
    casalogger = 0
    casalog = 0
    split = 0
    flagdata = 0
    keyboardException = 0
    generalException = 0
    pipeline_save = 0
    ms_active = 0
    datapath = 0


def print_time_stamp(object_, file_no, SDM_name):
    # Print the current time
    print('# Current time: ' +
          time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    print('# Executing ' + object_ + ' file ' + file_no + ': ' +
          SDM_name + '\n#')


print('####################################################')
print('# Executing Phase01: EVLA calibration pipeline')
print('####################################################')
#
# Read file name and decide what to do.
#
tgzname = None
SDM_name = None
msname = None
mscopyname = '.ms.copy'
reducedmsname = None
reducedmscopyname = '.reduced.ms.copy'
# All the .copy things should be removed in the final version

filelist = os.listdir(datapath + '/' + object_ + '/' + file_no)
for fn in filelist:
    if (fn[0] != '1') or (fn[-2:] == 'es') or (fn[-2:] == 'ns'):
        if os.path.isdir(fn):
            rmtree(fn)
        else:
            os.remove(fn)
    elif fn[-16:] == '.reduced.ms.copy':
        reducedmscopyname = fn
    elif fn[-11:] == '.reduced.ms':
        reducedmsname = fn
        rmtree(fn)
    elif fn[-8:] == '.ms.copy':
        mscopyname = fn
        if msname is None:
            msname = mscopyname[:-5]
    elif fn[-3:] == '.ms':
        msname = fn
        rmtree(fn)
    elif fn[-7:] == '.tar.gz':
        tgzname = fn
        if SDM_name is None:
            temp_filename = tgzname[:-7].split('.')
            SDM_name = '.'.join(temp_filename[:-1])

if (SDM_name is None) and (msname is not None):
    SDM_name = msname[:-3]

assert SDM_name is not None
print_time_stamp(object_, file_no, SDM_name)

if not os.path.isdir(SDM_name):
    assert tgzname is not None
    with tarfile.open(tgzname) as tar:
        print('#')
        print('# tar: SDM file not found. Start untar ' + tgzname)
        tar.extractall()
        print('# tar: Finished untar ' + tgzname)
        print('#')
        print_time_stamp(object_, file_no, SDM_name)

# This is the default time-stamped casa log file, in case we
#     need to return to it at any point in the script
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
#
maincasalog = casalogger.func_globals['thelogfile']


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
# CHECK IF A REDUCED MSFILE EXISTS
if os.path.isdir(reducedmscopyname):
    print('#')
    print('# copy: .reduced.ms.copy exists.')
    if os.path.isdir(reducedmsname):
        print('# copy: .reduced.ms exists. Start copying ' + reducedmsname)
        rmtree(reducedmsname)
    print('# copy: .reduced.ms.copy exists. Start copying ' +
          reducedmscopyname)
    copytree(reducedmscopyname, reducedmsname)
    print('# copy: Finished copy ' + reducedmscopyname)
    print('#')
    print_time_stamp(object_, file_no, SDM_name)
    msname = reducedmsname
    ms_active = reducedmsname
    execfile(pipepath + 'EVLA_pipe_import.py')
    print('#')
    execfile(pipepath + 'EVLA_pipe_msinfo.py')
else:
    print('#')
    print('# copy: .reduced.ms.copy not found')
    print('#')
    if os.path.isdir(mscopyname):
        print('#')
        print('# copy: .ms.copy exists.')
        if os.path.isdir(msname):
            print('# copy: .ms exists. Start removing ' + msname)
            rmtree(msname)
        print('# copy: .ms.copy exists. Start copying ' + mscopyname)
        copytree(mscopyname, msname)
        print('# copy: Finished copy ' + mscopyname)
        print('#')
        print_time_stamp(object_, file_no, SDM_name)
    else:
        print('#')
        print('# copy: .ms.copy not found')
        print('#')
    # IMPORT THE DATA TO CASA
    execfile(pipepath + 'EVLA_pipe_import.py')
    print('#')
    if not os.path.isdir(msname + '.copy'):
        print('#')
        print('# copy: .ms file created. Start backing up ' + msname)
        copytree(msname, msname + '.copy')
        print('# copy: Finished backing up ' + msname)
        print('#')
    """
    # HANNING SMOOTH (OPTIONAL, MAY BE IMPORTANT IF THERE IS NARROWBAND
    # RFI)
    execfile(pipepath + 'EVLA_pipe_hanning.py')
    print('#')
    """
    # GET SOME INFORMATION FROM THE MS THAT WILL BE NEEDED LATER, LIST
    #     THE DATA, AND MAKE SOME PLOTS
    execfile(pipepath + 'EVLA_pipe_msinfo.py')
    print('#')
    # DETERMINISTIC FLAGGING:
    # TIME-BASED: online flags, shadowed data, zeroes, pointing scans,
    # quacking
    # CHANNEL-BASED: end 5% of channels of each spw, 10 end channels at
    # edges of basebands
    execfile(pipepath + 'EVLA_pipe_flagall.py')
    print('#')
    # Only keep some spw
    if few_spw_ans:
        new_ms = SDM_name + '.reduced.ms'
        old_ms = msname
        old_ms_copy = msname + '.copy'
        print('#')
        print('# split: Start splitting ' + SDM_name + '.ms')
        split(vis=ms_active, outputvis=new_ms, spw=few_spws,
              datacolumn='all')
        print('# split: Finished splitting ' + SDM_name + '.ms')
        print('#')
        msname = new_ms
        ms_active = new_ms
        print('#')
        print('# copy: .reduced.ms file created. Start backing up ' + msname)
        copytree(msname, msname + '.copy')
        print('# copy: Finished backing up ' + msname)
        print('# remove: .reduced.ms created. Start removing ' + old_ms)
        rmtree(old_ms)
        print('# remove: .reduced.ms created. Start removing ' + old_ms_copy)
        rmtree(old_ms_copy)
        print('#')
        execfile(pipepath + 'EVLA_pipe_import.py')
        print('#')
        execfile(pipepath + 'EVLA_pipe_msinfo.py')
    print('#')

# Remove poluted channels...
if rm_Galatic_HI:
    flagdata(vis=msname, spw=default_rm_spw, intent='CALI*')
    logprint("Remove spw=" + default_rm_spw + ", intent=CALI*",
             logfileout='logs/import.log')
print('#')
# PREPARE FOR CALIBRATIONS
# Fill model columns for primary calibrators
execfile(pipepath + 'EVLA_pipe_calprep.py')
print('#')
# PRIOR CALIBRATIONS
# Gain curves, opacities, antenna position corrections,
# requantizer gains (NB: requires CASA 4.1 or later!).
# Also plots switched
# power tables, but these are not currently used in the calibration
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
# Improved calibrator flagging on customized script
execfile(pipepath + 'idchiang_flagcali.py')
print('#')
# RE-RUN semiFinalBPdcals.py FOLLOWING rflag
execfile(pipepath + 'EVLA_pipe_semiFinalBPdcals2.py')
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
execfile(pipepath + 'EVLA_pipe_targetflag.py')
print('#')
# CALCULATE DATA WEIGHTS BASED ON ST. DEV. WITHIN EACH SPW
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
if just_for_debug:
    open(timelog)
