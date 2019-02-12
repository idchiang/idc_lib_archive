# General imports
import traceback
import os
import tarfile
import numpy as np
from shutil import rmtree
# CASA imports
from h_init_cli import h_init_cli as h_init
from hifv_importdata_cli import hifv_importdata_cli as hifv_importdata
# from hifv_hanning_cli import hifv_hanning_cli as hifv_hanning
from hifv_flagdata_cli import hifv_flagdata_cli as hifv_flagdata
from hifv_vlasetjy_cli import hifv_vlasetjy_cli as hifv_vlasetjy
from hifv_priorcals_cli import hifv_priorcals_cli as hifv_priorcals
# from hif_refant_cli import hif_refant_cli as hif_refant
from hifv_testBPdcals_cli import hifv_testBPdcals_cli as hifv_testBPdcals
from hifv_flagbaddef_cli import hifv_flagbaddef_cli as hifv_flagbaddef
# from hifv_uncalspw_cli import hifv_uncalspw_cli as hifv_uncalspw
from hifv_checkflag_cli import hifv_checkflag_cli as hifv_checkflag
from hifv_semiFinalBPdcals_cli import hifv_semiFinalBPdcals_cli as \
    hifv_semiFinalBPdcals
from hifv_solint_cli import hifv_solint_cli as hifv_solint
from hifv_fluxboot_cli import hifv_fluxboot_cli as hifv_fluxboot
from hifv_finalcals_cli import hifv_finalcals_cli as hifv_finalcals
# from hifv_circfeedpolcal_cli import hifv_circfeedpolcal_cli as \
#     hifv_circfeedpolcal
from hifv_applycals_cli import hifv_applycals_cli as hifv_applycals
from hifv_targetflag_cli import hifv_targetflag_cli as hifv_targetflag
# from hifv_statwt_cli import hifv_statwt_cli as hifv_statwt
from hifv_plotsummary_cli import hifv_plotsummary_cli as hifv_plotsummary
from hif_makeimlist_cli import hif_makeimlist_cli as hif_makeimlist
from hif_makeimages_cli import hif_makeimages_cli as hif_makeimages
# from hifv_exportdata_cli import hifv_exportdata_cli as hifv_exportdata
from h_save_cli import h_save_cli as h_save
# Pipeline imports
import pipeline.infrastructure.casatools as casatools

"""
# Make sure CASA exceptions are rethrown
if False:
    __rethrow_casa_exceptions = 0
try:
    if not __rethrow_casa_exceptions:
        def_rethrow = False
    else:
        def_rethrow = __rethrow_casa_exceptions
except ValueError:
    def_rethrow = False
"""
def_rethrow = False
# Pipeline definitions
IMPORT_ONLY = 'Import only'
__rethrow_casa_exceptions = True
importonly = False
pipelinemode = 'automatic'
echo_to_screen = True

if False:
    flagdata, msmd, execfile, uvcontsub, split = 0

######################################################################
#
# My parameters and my functions here
#
######################################################################

remove_sdm = False
pipepath = '/data/scratch/idchiang/VLA_scripted_pipeline/'


def readfile_IDC():
    sdmname_IDC = None
    filelist = os.listdir(os.getcwd())
    for fn in filelist:
        temp = fn.split('.')
        if len(temp) == 5:
            sdmname_IDC = fn
        elif (len(temp) == 8) and (temp[-2:] == ['tar', 'gz']):
            tgzname_IDC = fn
            if sdmname_IDC is None:
                sdmname_IDC = '.'.join(temp[:5])
    # end for
    assert sdmname_IDC is not None
    msname_IDC = sdmname_IDC + '.ms'
    # end msname
    if not os.path.isdir(sdmname_IDC):
        assert tgzname_IDC is not None
        with tarfile.open(tgzname_IDC) as tar:
            casatools.post_to_log('SDM file not found. Start untar ' +
                                  tgzname_IDC)
            tar.extractall()
            casatools.post_to_log('Finished untar ' + tgzname_IDC)
    return msname_IDC, sdmname_IDC


def MW_and_quack_IDC(msname='', spw_line='8~12', target_spw=8):
    # Quack
    casatools.post_to_log('IDC: Quack data')
    # QUACK EVERYTHING
    flagdata(msname, mode='quack', action='apply', quackinterval=20,
             quackmode='endb', freqcutoff=100.0, timecutoff=100.0)
    flagdata(msname, mode='quack', action='apply', quackinterval=20,
             quackmode='beg',  freqcutoff=100.0, timecutoff=100.0)
    # QUACK THE PRIMARY CALIBRATOR (LINES)
    flagdata(msname, mode='quack', action='apply', quackinterval=30,
             quackmode='endb', intent='*BANDPASS*', spw=spw_line,
             freqcutoff=100.0, timecutoff=100.0)
    flagdata(msname, mode='quack', action='apply', quackinterval=30,
             quackmode='beg', intent='*BANDPASS*', spw=spw_line,
             freqcutoff=100.0, timecutoff=100.0)
    # Flag MW foreground
    casatools.post_to_log('IDC: Flag MW foreground')
    # Identify frequency
    restfreq = 1420405752
    msmd.open(msname)
    freqs = msmd.chanfreqs(spw=target_spw)
    msmd.close()
    # Complex calibrator
    min_freq = restfreq - 1953 * 200
    max_freq = restfreq + 1953 * 200
    mask = (freqs < max_freq) * (freqs > min_freq)
    if np.sum(mask) > 0:
        temp = np.arange(len(mask), dtype=int)
        chan_ids = temp[mask]
        min_chan, max_chan = str(np.min(chan_ids)), str(np.max(chan_ids) + 1)
        rm_spw = '8:' + min_chan + '~' + max_chan
        flagdata(vis=msname, spw=rm_spw, intent='*CALIBRATE_AMPLI*')
        del rm_spw, chan_ids, temp, min_chan, max_chan
    del mask, max_freq, min_freq
    # Bandpass calibrator
    min_freq = restfreq - 1953 * 100
    max_freq = restfreq + 1953 * 100
    mask = (freqs < max_freq) * (freqs > min_freq)
    min_chan, max_chan = '', ''
    if np.sum(mask) > 0:
        temp = np.arange(len(mask), dtype=int)
        chan_ids = temp[mask]
        min_chan, max_chan = str(np.min(chan_ids)), str(np.max(chan_ids) + 1)
        rm_spw = '8:' + min_chan + '~' + max_chan
        flagdata(vis=msname, spw=rm_spw, intent='*CALIBRATE_BANDPASS*')
    return min_chan, max_chan


def flagcali_IDC(msname):
    # Narrow-time, broad-band RFI flag at both calibrators at non-continuum
    # Extend flag is broken. So separate this to a two-stage flagging
    flagdata(vis=msname, intent='*CALI*', mode='tfcrop',
             datacolumn='corrected',
             extendflags=False, action='apply', display='',
             freqcutoff=3.0, timecutoff=2.0,
             combinescans=True,
             winsize=1, flagdimension='timefreq')
    flagdata(vis=msname, intent='*CALI*', mode='extend',
             datacolumn='corrected',
             extendflags=True, action='apply', display='',
             freqcutoff=100.0, timecutoff=100.0,
             growtime=70.0, growfreq=40.0,  # weak time due to quack
             extendpols=True, combinescans=False,
             winsize=1, flagdimension='timefreq')


def consub_and_split(msname, target_spw=8, min_chan='', max_chan='',
                     tarms=True):
    # CONT SUB
    # Not using continuum spws...
    excspw = str(target_spw) + ':'
    if (min_chan != '') and (max_chan != ''):
        excspw += min_chan + '~' + max_chan + ';'
    #
    msmd.open(msname)
    freqs = msmd.chanfreqs(spw=target_spw)
    msmd.close()
    centfreq = (freqs[0] + freqs[-1]) / 2  # Hz
    min_freq = centfreq - 1953 * 200
    max_freq = centfreq + 1953 * 200
    mask = (freqs < max_freq) * (freqs > min_freq)
    if np.sum(mask) > 0:
        temp = np.arange(len(mask), dtype=int)
        chan_ids = temp[mask]
        min_chan, max_chan = str(np.min(chan_ids)), str(np.max(chan_ids) + 1)
        excspw += min_chan + '~' + max_chan
    casatools.post_to_log('IDC: Starting uvcontsub.')
    casatools.post_to_log('IDC: SPWS to exclude: ' + excspw)
    uvcontsub(vis=msname, fitspw=excspw, excludechans=True,
              spw=str(target_spw), want_cont=False, fitorder=1)
    casatools.post_to_log('IDC: Splitting and interpolating ms files')
    split(vis=msname + '.contsub', outputvis=msname + '.split',
          intent='*TARGET*', width=4, datacolumn='data')
    casatools.post_to_log('IDC: Removing contsub files')
    rmtree(msname + '.contsub')
    # Split and remove the final ms
    # tar the split ms
    msname_r = msname + '.spw' + str(target_spw)
    casatools.post_to_log('IDC: Starting splitting the final ms.')
    split(vis=msname, outputvis=msname_r,
          spw=str(target_spw), datacolumn='corrected')
    casatools.post_to_log('IDC: Removing the final ms.')
    rmtree(msname)
    if tarms:
        casatools.post_to_log('IDC: Tar the reduced ms.')
        with tarfile.open(msname_r + '.tar.gz', 'w:gz') as tar:
            tar.add(msname_r, arcname=os.path.basename(msname_r))
        casatools.post_to_log('IDC: Removing the untar ms.')
        rmtree(msname_r)


######################################################################
#
# Read files here
#
######################################################################

msname_IDC, sdmname_IDC = readfile_IDC()

######################################################################
#
# Original hifv function starts here
#
######################################################################

casatools.post_to_log("Beginning VLA pipeline run ...")
# Initialize the pipeline
context = h_init()
context.set_state('ProjectSummary', 'observatory',
                  'Karl G. Jansky Very Large Array')
context.set_state('ProjectSummary', 'telescope', 'EVLA')

try:
    # Load the data
    hifv_importdata(vis=[sdmname_IDC], pipelinemode=pipelinemode)
    if importonly:
        raise Exception(IMPORT_ONLY)
    if remove_sdm:
        casatools.post_to_log("Removing the SDM file...")
        rmtree(sdmname_IDC)

    # Hanning smooth the data
    # hifv_hanning(pipelinemode=pipelinemode)

    # Flag known bad data
    hifv_flagdata(pipelinemode=pipelinemode, scan=True, hm_tbuff='1.5int',
                  intents='*POINTING*,*FOCUS*,*ATMOSPHERE*,*SIDEBAND_RATIO*,' +
                  '*UNKNOWN*, *SYSTEM_CONFIGURATION*,' +
                  ' *UNSPECIFIED#UNSPECIFIED*')

    # My own quack, MW foreground, manual flagging
    min_chan_IDC, max_chan_IDC = MW_and_quack_IDC(msname=msname_IDC)
    execfile(pipepath + 'idchiang_HI_flagmanual_hifv.py')

    # Fill model columns for primary calibrators
    hifv_vlasetjy(pipelinemode=pipelinemode)

    # Gain curves, opacities, antenna position corrections,
    # requantizer gains (NB: requires CASA 4.1!)
    hifv_priorcals(pipelinemode=pipelinemode)

    # Initial test calibrations using bandpass and delay calibrators
    hifv_testBPdcals(pipelinemode=pipelinemode)

    # Identify and flag basebands with bad deformatters or rfi based on
    # bp table amps and phases
    hifv_flagbaddef(pipelinemode=pipelinemode)

    # Flag spws that have no calibration at this point
    # hifv_uncalspw(pipelinemode=pipelinemode, delaycaltable='testdelay.k',
    #               bpcaltable='testBPcal.b')

    # Flag possible RFI on BP calibrator using rflag
    hifv_checkflag(pipelinemode=pipelinemode)

    # DO SEMI-FINAL DELAY AND BANDPASS CALIBRATIONS
    # (semi-final because we have not yet determined the spectral index of the
    # bandpass calibrator)
    hifv_semiFinalBPdcals(pipelinemode=pipelinemode)

    # Use flagdata rflag mode again on calibrators
    hifv_checkflag(pipelinemode=pipelinemode, checkflagmode='semi')
    flagcali_IDC(msname=msname_IDC)

    # Re-run semi-final delay and bandpass calibrations
    hifv_semiFinalBPdcals(pipelinemode=pipelinemode)

    flagcali_IDC(msname=msname_IDC)

    # Flag spws that have no calibration at this point
    # hifv_uncalspw(pipelinemode=pipelinemode, delaycaltable='delay.k',
    #               bpcaltable='BPcal.b')

    # Determine solint for scan-average equivalent
    hifv_solint(pipelinemode=pipelinemode)

    # Do the flux density boostrapping -- fits spectral index of
    # calibrators with a power-law and puts fit in model column
    hifv_fluxboot(pipelinemode=pipelinemode)

    # Make the final calibration tables
    hifv_finalcals(pipelinemode=pipelinemode)

    # Polarization calibration
    # hifv_circfeedpolcal(pipelinemode=pipelinemode)

    # Apply all the calibrations and check the calibrated data
    hifv_applycals(pipelinemode=pipelinemode)

    # Flag spws that have no calibration at this point
    # hifv_uncalspw(pipelinemode=pipelinemode, delaycaltable='finaldelay.k',
    #               bpcaltable='finalBPcal.b')

    # Now run all calibrated data, including the target, through rflag
    # hifv_targetflag(pipelinemode=pipelinemode,
    #                 intents='*CALIBRATE*,*TARGET*')
    hifv_targetflag(pipelinemode=pipelinemode, intents='*CALIBRATE*')

    # Calculate data weights based on standard deviation within each spw
    # hifv_statwt(pipelinemode=pipelinemode)

    # Plotting Summary
    hifv_plotsummary(pipelinemode=pipelinemode)

    # Make a list of expected point source calibrators to be cleaned
    hif_makeimlist(intent='PHASE,BANDPASS', specmode='cont',
                   pipelinemode=pipelinemode)

    # Make clean images for the selected calibrators
    hif_makeimages(hm_masking='none')

    # Export the data
    # hifv_exportdata(pipelinemode=pipelinemode)

except Exception as e:
    if str(e) == IMPORT_ONLY:
        casatools.post_to_log("Exiting after import step ...",
                              echo_to_screen=echo_to_screen)
    else:
        casatools.post_to_log("Error in procedure execution ...",
                              echo_to_screen=echo_to_screen)
        errstr = traceback.format_exc()
        casatools.post_to_log(errstr, echo_to_screen=echo_to_screen)

finally:
    # Save the results to the context
    h_save()

    consub_and_split(msname=msname_IDC, min_chan=min_chan_IDC,
                     max_chan=max_chan_IDC)

    casatools.post_to_log("VLA CASA Pipeline finished." +
                          "  Terminating procedure execution ...",
                          echo_to_screen=echo_to_screen)
    # Restore previous state
    __rethrow_casa_exceptions = def_rethrow
