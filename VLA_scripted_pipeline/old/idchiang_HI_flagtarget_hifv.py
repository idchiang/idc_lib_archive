# After the flagcali & calibration, flag the weird points in target
# No need to calibrate again after this
# Must be careful not to flag data...

# Remove bad data in some time ranges (low threshold)
# tfcrop: outliers on time-freq plane


# time only
# grow
print('# Begin customized target flag -- get metadata')
msmd.open(ms_active)
idc_target_scans = msmd.scansforintent('*TARGET*')
idc_target_scans_str = [str(its) for its in idc_target_scans]
msmd.close()
print('# Begin customized target flag -- tfcrop')
for itss in idc_target_scans_str:
    flagdata(vis=ms_active, scan=itss, mode='tfcrop',
             datacolumn='corrected',
             extendflags=False, action='apply', display='',
             freqcutoff=3.0, timecutoff=2.0,
             combinescans=True,
             winsize=1, flagdimension='time')
print('# Begin customized target flag -- extend')
for itss in idc_target_scans_str:
    flagdata(vis=ms_active, scan=itss, mode='extend',
             datacolumn='corrected',
             extendflags=True, action='apply', display='',
             freqcutoff=100.0, timecutoff=100.0,
             growtime=50.0, growfreq=30.0,
             extendpols=False, combinescans=False,
             winsize=1, flagdimension='freq')
print('# End customized target flag')

"""
Test area

spw_line = '8'

flagdata(vis=ms_active, spw=spw_line, field='2', antenna='ea01',
         mode='tfcrop',
         datacolumn='corrected',
         extendflags=False, action='apply', display='',
         freqcutoff=3.0, timecutoff=3.0,
         combinescans=True,
         winsize=1, flagdimension='time')

flagdata(vis=ms_active, spw=spw_line, field='2', antenna='ea01',
         mode='extend',
         datacolumn='corrected',
         extendflags=True, action='calculate', display='data',
         freqcutoff=3.0, timecutoff=3.0,
         growtime=50.0, growfreq=40.0,
         extendpols=True, combinescans=True,
         winsize=1, flagdimension='freq')

"""
