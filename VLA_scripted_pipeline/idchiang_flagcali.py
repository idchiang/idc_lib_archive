# Remove bad data in some time ranges (low threshold)
flagdata(vis=ms_active, spw='', intent='CALI*', mode='tfcrop', extendflags=False, datacolumn='corrected', channelavg=True, timecutoff=1.75, growtime=0.0, extendpols=False, combinescans=True, winsize=1, flagdimension='time')
# Remove all other outliers (default threshold, time first)
flagdata(vis=ms_active, spw='', intent='CALI*', mode='rflag', extendflags=False, datacolumn='corrected', growtime=0.0, extendpols=False, combinescans=True, winsize=1, flagdimension='timefreq')

"""
For testing, add parameters:



flagdata(vis='test.ms', mode='tfcrop', extendflags=False, datacolumn='data', channelavg=True, timecutoff=1.75, growtime=0.0, extendpols=False, combinescans=True, winsize=1, flagdimension='time')
"""

