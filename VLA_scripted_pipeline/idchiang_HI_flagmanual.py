# Flag data from known list.

if False:
    SDM_name, ms_active, flagdata = 0

idchiang_manual_flags = \
    {'14B-396.sb29939521.eb29987204.56975.4131616088':  # NGC1961 01
     ["mode='manual' antenna='ea02'"],
     '14B-396.sb29939521.eb29989968.56978.41638283565':  # NGC1961 02
     ["mode='manual' antenna='ea02,ea28'"],
     '14A-468.sb29806206.eb30086747.57019.45709732639':  # NGC2787 01
     ["mode='manual' antenna='ea05'",
      "mode='manual' scan='2'"],
     '14A-468.sb29806206.eb30092752.57025.430288171294':  # NGC2787 02
     ["mode='manual' scan='2'"],
     '14B-396.sb29895068.eb30086390.57017.618360092594':  # NGC3147 01
     ["mode='manual' spw='*:2463~2469' intent='*BANDPASS*'",
      "mode='manual' spw='*:2452~2478' intent='*PHASE*'",
      "mode='manual' spw='*:2462~2471' intent='*TARGET*'",
      "mode='manual' scan='9,10,12' spw='*:1560~1561' antenna='ea24,ea25' " +
      "intent='*TARGET*'",
      "mode='manual' spw='*:1501' intent='*TARGET*' antenna='ea13'",
      "mode='manual' spw='*:1437' intent='*TARGET*' antenna='ea13'",
      "mode='manual' antenna='ea27'"],
     '17A-073.sb33225557.eb33881364.57904.13970773148':  # NGC3147 02
     ["mode='manual' spw='*:2463~2469' intent='*BANDPASS*'",
      "mode='manual' spw='*:2452~2478' intent='*PHASE*'",
      "mode='manual' spw='*:2462~2471' intent='*TARGET*'",
      "mode='manual' antenna='ea11'"],
     '14A-468.sb29806909.eb30107700.57029.27375394676':  # NGC3227 01
     ["mode='manual' timerange='08:59:50~09:00:20'",
      "mode='manual' antenna='ea13'"],
     '16A-275.sb31548863.eb31985479.57502.09742939815':  # NGC3227 03
     ["mode='manual' timerange='04:15:55~04:16:10'",
      "mode='manual' timerange='04:14:50~04:15:00'",
      "mode='manual' timerange='04:15:50~04:17:10' antenna='ea03'",
      "mode='manual' scan='15' antenna='ea03'"],
     '17A-073.sb33229232.eb33908113.57907.00082898149':  # NGC3640 01
     ["mode='manual' scan='14'",
      "mode='manual' antenna='ea05'"],
     '17A-073.sb33229232.eb33912078.57908.97453717593':  # NGC3640 02
     ["mode='manual' scan='2' antenna='ea05'"],
     '14A-468.sb29807793.eb30084941.57014.33568722222':  # NGC3898 01
     ["mode='manual' timerange='09:48:00~09:51:00'"],
     '17A-073.sb33225688.eb33792405.57894.94687215278':  # NGC3953 02
     ["mode='manual' antenna='ea24'",
      "mode='manual' antenna='ea12&ea20'"],
     '16A-275.sb31549466.eb31985481.57502.18057351852':  # NGC4038 01
     ["mode='manual' antenna='ea24'"],
     '14A-468.sb29807927.eb30085571.57015.5511071875':  # NGC4374 01
     ["mode='manual' antenna='ea05'"],
     '17A-073.sb33225822.eb33498752.57797.37958828703':  # NGC4374 03
     ["mode='manual' antenna='ea22'", "mode='manual' antenna='ea24'"],
     '14A-468.sb29807994.eb30092753.57025.51342440972':  # NGC4477 01
     ["mode='manual' antenna='ea04' timerange='12:26:10~12:26:40'"],
     '14B-396.sb29626618.eb31289487.57305.699423287035':  # NGC4477 02
     ["mode='manual' antenna='ea24'",
      "mode='manual' timerange='16:50:20~16:50:55'",
      "mode='manual' timerange='16:59:40~16:59:51'",
      "mode='manual' timerange='17:03:15~17:03:25'",
      "mode='manual' timerange='17:30:00~17:30:20'"],
     '14B-396.sb29626723.eb31336959.57312.82546952546':  # NGC4494 01
     ["mode='manual' timerange='19:55:07~19:55:11'"],
     '14B-396.sb29809236.eb30092754.57025.60658178241':  # NGC4494 03
     ["mode='manual' timerange='14:37:00~14:37:40'"],
     '17A-073.sb33225356.eb33879561.57904.05461909722':  # NGC4496A 01
     ["mode='manual' antenna='ea08;ea18;ea22'",
      "mode='manual' antenna='ea03&ea07;ea06&ea14;" +
      "ea12&ea14;ea14&ea15;ea14&ea17' intent='*TARGET*'",
      "mode='manual' antenna='ea03&ea07;ea06&ea14;" +
      "ea12&ea14;ea14&ea15;ea14&ea17' intent='*PHASE*'",
      "mode='manual' antenna='ea01&ea09' intent='*TARGET*'",
      "mode='manual' antenna='ea01&ea09' intent='*PHASE*'",
      "mode='manual' antenna='ea01&ea25;ea03&ea09' intent='*TARGET*'",
      "mode='manual' antenna='ea01&ea25;ea03&ea09' intent='*PHASE*'",
      "mode='manual' antenna='ea01&ea13;ea01&ea27;ea07&ea13;ea07&ea14;" +
      "ea09&ea13;ea09&ea14;ea14&ea27;ea13&ea28' intent='*TARGET*'",
      "mode='manual' antenna='ea01&ea13;ea01&ea27;ea07&ea13;ea07&ea14;" +
      "ea09&ea13;ea09&ea14;ea14&ea27;ea13&ea28' intent='*PHASE*'"], 
     '16A-275.sb31548796.eb31985714.57503.18479064815':  # NGC4496A 02
     ["mode='manual' antenna='ea27;ea18'",
      "mode='manual' scan='17~19'",
      "mode='manual' antenna='ea02&ea05;ea02&ea20;ea02&ea26;ea05&ea07;" +
      "ea05&ea16;ea05&ea22;ea05&ea28;ea07&ea20;ea13&ea16;ea13&ea22;" +
      "ea16&ea20;ea16&ea23;ea16&ea26;ea20&ea22;ea22&ea23;ea22&ea26'"], 
     # '14B-396.sb29626836.eb31333839.57311.65158101852':  # NGC4496A 03
     # ["mode='manual' antenna='ea10&ea26;ea12&ea13;ea13&ea23;ea17&ea20;" +
     #  "ea01&ea20;ea06&ea20;ea11&ea19;ea27&ea28;ea01&ea28;ea10&ea23;" +
     #  "ea07&ea27;ea06&ea12;ea06&ea17;ea12&ea17;ea12&ea19;ea12&ea22;" +
     #  "ea13&ea15;ea13&ea22;ea15&ea22;ea16&ea24;ea16&ea25;ea18&ea24;" +
     #  "ea11&ea15;ea06&ea23;ea11&ea17;ea10&ea15;ea12&ea15;ea22&ea23;" +
     #  "ea17&ea25;ea11&ea23;ea02;ea04;ea03'"],
     '14A-468.sb29808201.eb30086748.57019.54018327546':  # NGC4501 01
     ["mode='manual' intent='*CALI*' spw='*:1133~1137'",
      "mode='manual' antenna='ea13' timerange='13:00:00~13:01:00'"],
     '14B-396.sb29627017.eb31291859.57306.722631412034':  # NGC4501 02
     ["mode='manual' intent='*CALI*' spw='*:1130~1140'"],
     '16A-275.sb31549065.eb31764282.57420.35559265046':  # NGC4501 03
     ["mode='manual' intent='*CALI*' spw='*:1130~1140'"],
     '14A-468.sb29808309.eb30089887.57022.66084362268':  # NGC4596 02
     ["mode='manual' antenna='ea18' timerange='15:56:20~15:56:30'",
      "mode='manual' timerange='15:53:50~15:54:10'",
      "mode='manual' antenna='ea27'"],
     '14B-396.sb29627227.eb31367659.57318.65171790509':  # NGC4636 01
     ["mode='manual' timerange='17:36:30~17:36:50'",
      "mode='manual' timerange='16:45:55~16:46:10'",
      "mode='manual' timerange='15:45:15~15:45:25'"],
     '14A-468.sb29639312.eb29693960.56920.90377532407':  # NGC5728 01
     ["mode='manual' antenna='ea09'",
      "mode='manual' spw='*:2324~2328' intent='*CALI*'",
      "mode='manual' spw='*:2325~2327' intent='*TARGET*'",
      "mode='manual' scan='3' intent='*PHASE*' " +
      "spw='*:703,*:714,*:1011,*:1016',*:1038~1084"],
     '14B-396.sb30126560.eb30158093.57038.48830790509':  # NGC5728 02
     ["mode='manual' spw='*:2324~2328' intent='*CALI*'",
      "mode='manual' spw='*:2325~2327' intent='*TARGET*'"],
     '14B-396.sb30126560.eb30168231.57040.471906631945':  # NGC5728 03
     ["mode='manual' spw='*:2324~2328' intent='*CALI*'",
      "mode='manual' spw='*:2325~2327' intent='*TARGET*'"],
     '14B-396.sb29939409.eb30006248.56984.02285145833':  # NGC7479 01
     ["mode='manual' spw='*:1468~1473'",
      "mode='manual' antenna='ea13'"],
     '14B-396.sb29939409.eb30008116.56986.017359618054':  # NGC7479 02
     ["mode='manual' spw='*:1468~1473'",
      "mode='manual' antenna='ea13'"]}

if SDM_name in idchiang_manual_flags.keys():
    flagdata(ms_active, mode='list', inpfile=idchiang_manual_flags[SDM_name])

# ea02 is the center of NGC4496A