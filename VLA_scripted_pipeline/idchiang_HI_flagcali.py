# Remove bad data in some time ranges (low threshold)
# tfcrop: outliers on time-freq plane
if few_spw_ans:
    spw_line = ''
else:
    spw_cont = '0~7'
    spw_line = '8~12'

# Very strict flagging for continuum at bandpass after a normal flagging
# normal flagging at both
"""
flagdata(vis=ms_active, spw=spw_cont, intent='*CALI*', mode='tfcrop',
         datacolumn='corrected',
         extendflags=False, action='apply', display='',
         freqcutoff=3.0, timecutoff=3.0,
         combinescans=True,
         winsize=1, flagdimension='freqtime')
# flag too many things. gonna modify anyway.
# aggressive flagging in freq at bandpass calibrator
flagdata(vis=ms_active, spw=spw_cont, intent='*BANDPASS*', mode='tfcrop',
         datacolumn='corrected',
         extendflags=False, action='apply', display='',
         freqcutoff=2.0, freqfit='line', timebin='10000s',
         extendpols=True, combinescans=True,
         winsize=1, flagdimension='freqtime')
# grow flagging at both
flagdata(vis=ms_active, spw=spw_cont, intent='*CALI*', mode='extend',
         datacolumn='corrected',
         extendflags=True, action='apply', display='',
         freqcutoff=100.0, timecutoff=100.0,  # try only extending
         growtime=40.0, growfreq=80.0,  # strong time weak freq
         extendpols=True, combinescans=True,
         winsize=1, flagdimension='freqtime')
"""

# Narrow-time, broad-band RFI flag at both calibrators at non-continuum
# Extend flag is broken. So separate this to a two-stage flagging
flagdata(vis=ms_active, intent='*CALI*', mode='tfcrop',
         datacolumn='corrected', spw=spw_line,
         extendflags=False, action='apply', display='',
         freqcutoff=3.0, timecutoff=2.0,
         combinescans=True,
         winsize=1, flagdimension='timefreq')
flagdata(vis=ms_active, intent='*CALI*', mode='extend',
         datacolumn='corrected', spw=spw_line,
         extendflags=True, action='apply', display='',
         freqcutoff=100.0, timecutoff=100.0,
         growtime=70.0, growfreq=40.0,  # weak time due to quack
         extendpols=True, combinescans=False,
         winsize=1, flagdimension='timefreq')

"""
Test area

spw_line = '8'

flagdata(vis=ms_active, spw='8', intent='*CALI*', mode='tfcrop',
         datacolumn='corrected',
         extendflags=False, action='calculate', display='data',
         freqcutoff=3.0, timecutoff=2.0,
         combinescans=True,
         winsize=1, flagdimension='timefreq')

"""
