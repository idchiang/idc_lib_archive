if False:
    flagdata, ms_active = 0

spw_cont = '0~7'
spw_line = '8~12'

# QUACK EVERYTHING
flagdata(ms_active, mode='quack', action='apply', quackinterval=20,
         quackmode='endb', freqcutoff=100.0, timecutoff=100.0)
flagdata(ms_active, mode='quack', action='apply', quackinterval=20,
         quackmode='beg',  freqcutoff=100.0, timecutoff=100.0)
