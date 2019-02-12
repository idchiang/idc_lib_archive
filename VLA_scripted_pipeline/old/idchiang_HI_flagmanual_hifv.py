# Flag data from known list.

if False:
    sdmname_IDC, msname_IDC, flagdata = 0

idchiang_manual_flags = \
    {'14B-396.sb29809142.eb30014592.56987.35765355324':  # IC342 01
     ["mode='manual' timerange='08:47:15~08:51:00' spw='8'"],
     '14B-396.sb29615895.eb29629907.56904.49060305556':  # IC342 02
     ["mode='manual' timerange='11:48:10~11:48:35' spw='8'",
      "mode='manual' timerange='12:02:15~12:02:25' spw='8'"],
     '14B-396.sb29809142.eb30006252.56984.355307210644':  # IC342 03
     ["mode='manual' timerange='08:44:00~08:48:00' spw='8'"],
     '14A-468.sb29806206.eb30086747.57019.45709732639':  # NGC2787 01
     ["mode='manual' antenna='ea05' spw='8'",
      "mode='manual' scan='2' spw='8'"],
     '14A-468.sb29806206.eb30092752.57025.430288171294':  # NGC2787 02
     ["mode='manual' scan='2' spw='8'"],
     '14A-468.sb29806909.eb30107700.57029.27375394676':  # NGC3227 01
     ["mode='manual' timerange='08:59:50~09:00:20' spw='8'",
      "mode='manual' antenna='ea13' spw='8'"],
     '16A-275.sb31548863.eb31985479.57502.09742939815':  # NGC3227 03
     ["mode='manual' timerange='04:15:55~04:16:10' spw='8'",
      "mode='manual' timerange='04:14:50~04:15:00' spw='8'",
      "mode='manual' timerange='04:15:50~04:17:10' antenna='ea03' spw='8'",
      "mode='manual' scan='15' antenna='ea03' spw='8'"],
     '14A-468.sb29807793.eb30084941.57014.33568722222':  # NGC3898 01
     ["mode='manual' timerange='09:48:00~09:51:00' spw='8'"],
     '14A-468.sb29807927.eb30085571.57015.5511071875':  # NGC4374 01
     ["mode='manual' antenna='ea05' spw='8'"],
     '17A-073.sb33225822.eb33498752.57797.37958828703':  # NGC4374 03
     ["mode='manual' antenna='ea22' spw='8'",
      "mode='manual' antenna='ea24' spw='8'"],
     '14A-468.sb29807994.eb30092753.57025.51342440972':  # NGC4477 01
     ["mode='manual' antenna='ea04' timerange='12:26:10~12:26:40' spw='8'"],
     '14B-396.sb29626618.eb31289487.57305.699423287035':  # NGC4477 02
     ["mode='manual' antenna='ea24' spw='8'",
      "mode='manual' timerange='16:50:20~16:50:55' spw='8'",
      "mode='manual' timerange='16:59:40~16:59:51' spw='8'",
      "mode='manual' timerange='17:03:15~17:03:25' spw='8'",
      "mode='manual' timerange='17:30:00~17:30:20' spw='8'"],
     '14B-396.sb29626723.eb31336959.57312.82546952546':  # NGC4494 01
     ["mode='manual' timerange='19:55:07~19:55:11' spw='8'"],
     '14B-396.sb29809236.eb30092754.57025.60658178241':  # NGC4494 03
     ["mode='manual' timerange='14:37:00~14:37:40' spw='8'"],
     '17A-073.sb33225356.eb33879561.57904.05461909722':  # NGC4496A 01
     ["mode='manual' timerange='01:22:30~01:22:50' spw='8'",
      "mode='manual' timerange='01:31:00~01:32:00' spw='8'"],
     '16A-275.sb31548796.eb31985714.57503.18479064815':  # NGC4496A 02
     ["mode='manual' timerange='04:38:30~04:39:30' spw='8'"],
     '14B-396.sb29626836.eb31333839.57311.65158101852A':  # NGC4496A 03
     ["mode='manual' scan='3' spw='8'",
      "mode='manual' antenna='ea02&ea11;ea02&ea12;ea02&ea17;ea04&ea11;" +
      "ea11&ea17;ea11&ea23;ea12&ea13;ea15&ea22;ea27&ea28;ea01&ea20;" +
      "ea01&ea28;ea02&ea04;ea02&ea13;ea03&ea19;ea04&ea23;ea06&ea17;" +
      "ea06&ea20;ea10&ea23;ea11&ea15;ea11&ea19;ea13&ea15;ea13&ea22;" +
      "ea15&ea23;ea17&ea20' spw='8'"],
     '14B-396.sb29626836.eb31333839.57311.65158101852B':  # NGC4496A 03
     ["mode='manual' antenna='ea02&ea11;ea02&ea12;ea02&ea17;ea04&ea11;" +
      "ea11&ea17;ea11&ea23;ea12&ea13;ea15&ea22;ea27&ea28;ea01&ea20;" +
      "ea01&ea28;ea02&ea04;ea02&ea13;ea03&ea19;ea04&ea23;ea06&ea17;" +
      "ea06&ea20;ea10&ea23;ea11&ea15;ea11&ea19;ea13&ea15;ea13&ea22;" +
      "ea15&ea23;ea17&ea20' spw='8'"],
     '14A-468.sb29808201.eb30086748.57019.54018327546':  # NGC4501 01
     ["mode='manual' intent='*CALI*' spw='8:1133~1137'",
      "mode='manual' antenna='ea13' timerange='13:00:00~13:01:00' spw='8'"],
     '14B-396.sb29627017.eb31291859.57306.722631412034':  # NGC4501 02
     ["mode='manual' intent='*CALI*' spw='8:1130~1140'"],
     '16A-275.sb31549065.eb31764282.57420.35559265046':  # NGC4501 03
     ["mode='manual' intent='*CALI*' spw='8:1130~1140'"],
     '14A-468.sb29808309.eb30089887.57022.66084362268':  # NGC4596 02
     ["mode='manual' antenna='ea18' timerange='15:56:20~15:56:30' spw='8'",
      "mode='manual' timerange='15:53:50~15:54:10' spw='8'",
      "mode='manual' antenna='ea27' spw='8'"],
     '14B-396.sb29627227.eb31367659.57318.65171790509':  # NGC4636 01
     ["mode='manual' timerange='17:36:30~17:36:50' spw='8'",
      "mode='manual' timerange='16:45:55~16:46:10' spw='8'",
      "mode='manual' timerange='15:45:15~15:45:25' spw='8'"],
     '14A-468.sb29639312.eb29693960.56920.90377532407':  # NGC5728 01
     ["mode='manual' antenna='ea09' spw='8'",
      "mode='manual' spw='8:2324~2328' intent='*CALI*'",
      "mode='manual' spw='8:2325~2327' intent='*TARGET*'",
      "mode='manual' scan='3' intent='*PHASE*' " +
      "spw='8:703,8:714,8:1011,8:1016',8:1038~1084"],
     '14B-396.sb30126560.eb30158093.57038.48830790509':  # NGC5728 02
     ["mode='manual' spw='8:2324~2328' intent='*CALI*'",
      "mode='manual' spw='8:2325~2327' intent='*TARGET*'"],
     '14B-396.sb30126560.eb30168231.57040.471906631945':  # NGC5728 03
     ["mode='manual' spw='8:2324~2328' intent='*CALI*'",
      "mode='manual' spw='8:2325~2327' intent='*TARGET*'"]}

if sdmname_IDC in idchiang_manual_flags.keys():
    flagdata(msname_IDC, mode='list',
             inpfile=idchiang_manual_flags[sdmname_IDC])

# ea02 is the center of NGC4496A
