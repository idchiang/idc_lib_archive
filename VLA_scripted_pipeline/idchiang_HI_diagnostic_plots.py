import os
import imp
import numpy as np
from shutil import copyfile

# Just for HI
HIfreq = 1420.405752  # MHz
# Read from galbase
target = os.getcwd().split('/')[-2]
gal_data = imp.load_source('gal_data',
                           '/home/idchiang/idc_lib/idc_lib/gal_data.py')
gdata = gal_data.gal_data(target)
vhel = gdata.field('VHEL_KMS')[0]
c = 299792458 / 1000
restfreq = str(HIfreq * c / (c + vhel)) + 'MHz'

if False:
    plotms, plotcal, ms_active, weblog_dir, flagdata, plotants = 0
spath = weblog_dir + '/idchiang_extra/'

if not os.path.isdir(spath):
    os.mkdir(spath)

if few_spw_ans:
    plot_spw = ''
else:
    plot_spw = '8'

# amp-time
plotms(vis=ms_active,
       xaxis='time', yaxis='amp', avgchannel='4096', coloraxis='ant1',
       avgantenna=True, overwrite=True, ydatacolumn='corrected',
       plotfile=spath+'amp-time_target.png', avgtime='10',
       showgui=False, intent='*TARGET*', spw=plot_spw)

plotms(vis='calibrators.ms',
       xaxis='time', yaxis='amp', avgchannel='4096', coloraxis='ant1',
       avgantenna=True, overwrite=True, ydatacolumn='corrected',
       plotfile=spath+'amp-time_calibrators.png', avgtime='10',
       showgui=False, spw=plot_spw)

# amp-vel
plotms(vis=ms_active,
       xaxis='vel', yaxis='amp', avgchannel='', coloraxis='ant1',
       avgtime='100000', avgscan=True, restfreq=restfreq,
       veldef='RADIO',
       avgantenna=True, overwrite=True, ydatacolumn='corrected',
       plotfile=spath+'amp-vel_all.png',
       showgui=False, iteraxis='field', exprange='all', spw=plot_spw)

# phase-vel
plotms(vis=ms_active, intent='*CALI*',
       xaxis='vel', yaxis='phase', coloraxis='ant1',
       avgtime='100000', avgscan=True, restfreq=restfreq,
       veldef='RADIO',
       avgantenna=True, overwrite=True, ydatacolumn='corrected',
       plotfile=spath+'phase-vel.png',
       showgui=False, iteraxis='field', exprange='all',
       plotrange=[0, 0, -180, 180], spw=plot_spw)

# uvdist-amp
plotms(vis=ms_active,
       xaxis='uvdist', yaxis='amp', avgchannel='4096', coloraxis='ant1',
       avgtime='5000', avgscan=False,
       avgantenna=False, overwrite=True, ydatacolumn='corrected',
       plotfile=spath+'amp-uv_target.png',
       showgui=False, iteraxis='field', exprange='all', intent='*TARGET*',
       spw=plot_spw)

plotms(vis='calibrators.ms',
       xaxis='uvdist', yaxis='amp', avgchannel='4096', coloraxis='field',
       avgtime='5000', avgscan=False,
       avgantenna=False, overwrite=True, ydatacolumn='corrected',
       plotfile=spath+'amp-uv_calibrators.png',
       showgui=False, spw=plot_spw)

# uvdist-phase
plotms(vis='calibrators.ms',
       xaxis='uvdist', yaxis='phase', avgchannel='4096', coloraxis='ant1',
       avgtime='5000', avgscan=False,
       avgantenna=False, overwrite=True, ydatacolumn='corrected',
       plotfile=spath+'phase-uv.png',
       showgui=False, iteraxis='field', exprange='all',
       plotrange=[0, 0, -180, 180], spw=plot_spw)

# BPcal
plotcal(caltable='final_caltables/finalBPcal.b', xaxis='freq',
        yaxis='amp', iteration='antenna',
        subplot=651, overplot=False, clearpanel='Auto', showflags=False,
        showgui=False, figfile=spath+'finalBPcal.png', spw=plot_spw)

# UV coverage
plotms(vis=ms_active, intent='*TARGET*', xaxis='Uwave', yaxis='Vwave',
       avgchannel='4096', coloraxis='ant1', customflaggedsymbol=True,
       showgui=False, plotfile=spath+'uvcoverage.png', spw=plot_spw)

# flagdata summary
flagsummary = flagdata(ms_active, mode='summary', spw=plot_spw)
antsummary = flagsummary['antenna']
flaglist = []
for ant in antsummary.keys():
    flaglist.append([ant, round(antsummary[ant]['flagged'] /
                                antsummary[ant]['total'], 3)])
flaglist = np.sort(flaglist, axis=0)
np.savetxt(spath + 'flagsummary_all.txt', flaglist, fmt='%s')
#
flagsummary = flagdata(ms_active, mode='summary', intent='*CALI*',
                       spw=plot_spw)
antsummary = flagsummary['antenna']
flaglist = []
for ant in antsummary.keys():
    flaglist.append([ant, round(antsummary[ant]['flagged'] /
                                antsummary[ant]['total'], 3)])
flaglist = np.sort(flaglist, axis=0)
np.savetxt(spath+'flagsummary_cali.txt', flaglist, fmt='%s')

# copy from weblog
copyfile(weblog_dir+'/plotants.png', spath+'plotants.png')
copyfile(weblog_dir+'/onlineFlags.png', spath+'onlineFlags.png')
copyfile(weblog_dir+'/el_vs_time.png', spath+'el_vs_time.png')
