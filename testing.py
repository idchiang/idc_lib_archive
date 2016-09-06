# log scale graph

import numpy as np
import pandas as pd

galaxy_data = pd.read_csv('galaxy_data.csv')
from astro_idchiang.io import Surveys
cmaps = Surveys(['NGC 628', 'NGC 3198'], ['THINGS', 'HERACLES', 'PACS_100', 'SPIRE_350'])