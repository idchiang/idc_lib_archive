inp_list = {'14A-468.sb29639312.eb29693960.56920.90377532407spw_8.ms':
            ["antenna='ea19&ea28;ea11&ea25;ea04&ea19;ea16&ea24;ea11&ea22;ea04&ea28;ea13&ea25;ea13&ea24;ea04&ea18;ea11&ea19;ea17&ea18;ea22&ea25'",
             "scan='5' antenna='ea11&ea28;ea09&ea16;ea03&ea15;ea18&ea28;ea05&ea23;ea11&ea13;ea05&ea09'",
             "scan='8' antenna='ea24&ea28;ea18&ea28;ea05&ea09;ea05&ea23;ea11&ea13'",
             "scan='9,10' antenna='ea13&ea22'"]}

try:
    flagdata(vis=ms_active, mode='list', inpfile=inp_list[ms_active])
except KeyError:
    pass
