######################################################################
#
# Copyright (C) 2013
# Associated Universities, Inc. Washington DC, USA,
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Library General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public
# License for more details.
#
# You should have received a copy of the GNU Library General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 675 Massachusetts Ave, Cambridge, MA 02139, USA.
#
# Correspondence concerning VLA Pipelines should be addressed as follows:
#    Please register and submit helpdesk tickets via: https://help.nrao.edu
#    Postal address:
#              National Radio Astronomy Observatory
#              VLA Pipeline Support Office
#              PO Box O
#              Socorro, NM,  USA
#
######################################################################

# CALCULATE DATA WEIGHTS BASED ON ST. DEV. WITHIN EACH SPW
# use statwt
import numpy as np

if False:
    few_spw_ans, tb, ms_active, logprint, runtiming, default, statwt, \
        pipeline_save = 0

#
if few_spw_ans:
    fitspw = '*'
else:
    fitspw = '8'
#
"""
HIfreq = 1420405752  # Hz
# Read from galbase
target = os.getcwd().split('/')[-2]
gal_data = imp.load_source('gal_data',
                           '/home/idchiang/idc_lib/idc_lib/gal_data.py')
gdata = gal_data.gal_data(target)
vhel = gdata.field('VHEL_KMS')[0]
c = 299792458 / 1000
restfreq = HIfreq * c / (c + vhel)  # Hz
"""
tb.open(ms_active + '/SPECTRAL_WINDOW')
if few_spw_ans:
    freqs = tb.getcol('CHAN_FREQ').reshape(-1)
else:
    freqs = tb.getcell('CHAN_FREQ', 8)
restfreq = (freqs[0] + freqs[-1]) / 2  # Hz
min_freq = restfreq - 1953 * 200
max_freq = restfreq + 1953 * 200
mask = (freqs < max_freq) * (freqs > min_freq)
if np.sum(mask) > 0:
    fitspw += ':'
    temp = np.arange(len(mask), dtype=int)
    chan_ids = temp[mask]
    if not mask[0]:
        fitspw += '0~' + str(np.min(chan_ids) - 1)
    if not mask[4095]:
        if fitspw[-1] not in [':', ';']:
            fitspw += ';'
        fitspw += str(np.max(chan_ids) + 1) + '~4095'
tb.close()

logprint("Starting EVLA_pipe_statwt.py", logfileout='logs/statwt.log')
time_list = runtiming('checkflag', 'start')
QA2_statwt = 'Pass'

logprint("Calculate data weights per spw using statwt",
         logfileout='logs/statwt.log')

# Run on all calibrators

default(statwt)
vis = ms_active
dorms = False
fitspw = ''
fitcorr = ''
combine = ''
minsamp = 2
field = ''
spw = ''
intent = '*CALIBRATE*'
datacolumn = 'corrected'
statwt()

# Run on all targets
# set spw to exclude strong science spectral lines

default(statwt)
vis = ms_active
dorms = False
fitspw = fitspw
fitcorr = ''
combine = ''
minsamp = 2
field = ''
spw = fitspw
intent = '*TARGET*'
datacolumn = 'corrected'
statwt()

# Until we understand better the failure modes of this task, leave QA2
# score set to "Pass".

logprint("QA2 score: "+QA2_statwt, logfileout='logs/statwt.log')
logprint("Finished EVLA_pipe_statwt.py", logfileout='logs/statwt.log')
time_list = runtiming('targetflag', 'end')

pipeline_save()


######################################################################
