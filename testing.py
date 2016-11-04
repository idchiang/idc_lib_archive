from __future__ import absolute_import, print_function, unicode_literals, \
                       division
range = xrange
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
# from astro_idchiang import Surveys, imshowid, read_dust_file
from h5py import File

################################
n, m = 2, 3
aset = [2, 4]
bset = [16, 32, 96]

ma, mb = max(aset), min(bset)
if ma > mb:
    print(0)
else:
    possible_set = [i for i in range(ma, mb//2 + 1)]
    possible_set.append(mb)
    possible_set = [i for i in range(2, mb + 1)]
    for i in range(len(possible_set) - 1, -1, -1):
        ja = [possible_set[i] % aset[j] for j in range(n)]
        jb = [bset[j] % possible_set[i] for j in range(m)]
        if sum(ja) + sum(jb):
            possible_set.remove(possible_set[i])
    print(len(possible_set))