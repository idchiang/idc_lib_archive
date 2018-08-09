import os
import sys
from platform import system
# import multiprocessing as mp


# Path to pipeline and data. Change them when needed.
pipepath = '/home/idchiang/idc_lib/VLA_scripted_pipeline/'
datapath = '/home/idchiang/Data_EVLA/'


# SDM list. Should be read from outside in the final version
all_sdm = {'NGC5728': ['01', '02', '03']}
all_objects = ['NGC5728']
projectCode = '14A-468;14B-396'
piName = 'Karin Sandstrom'

# Just for first test.
all_sdm = {'NGC5728': ['01']}
all_objects = ['NGC5728']


# For easier debugging with PEP8 in Windows.
# Please don't execute this script in Windows even if you can.
just_for_debug = system() == 'Windows'
if just_for_debug:
    casa = 0
    execfile = 0
    casalogger = 0
######################################################################
#
# Definitions for the pipeline
#
######################################################################
version = "1.4.0"
svnrevision = '11nnn'
date = "2017Mar08"


def version_check():
    # Change version and date below with each svn commit.  Note changes in the
    # .../trunk/doc/CHANGELOG.txt and .../trunk/doc/bugs_features.txt files
    #
    # And prevent me running the pipeline with python directly...
    #
    print("Pipeline version " + version + " for use with CASA 5.0.0")
    #
    [major, minor, revision] = casa['build']['version'].split('.')
    casa_version = 100 * int(major) + 10 * int(minor) + int(revision[:1])
    if casa_version < 500:
        sys.exit("Your CASA version is " + casa['build']['version'] +
                 ", please re-start using CASA 5.0.0")
    if casa_version > 500:
        sys.exit("Your CASA version is " + casa['build']['version'] +
                 ", please re-start using CASA 5.0.0")


######################################################################
#
# Pipeline starts here
#
######################################################################

if __name__ == '__main__':
    version_check()
    for object_ in all_objects:
        for file_no in all_sdm[object_]:
            os.chdir(datapath + object_ + '/' + file_no)
            # execfile(pipepath + 'idchiang_Phase01_EVLA_pipeline.py')
            # execfile(pipepath + 'idchiang_phase2_mflag.py')
        os.chdir(datapath + object_)
        # execfile(pipepath + 'idchiang_Phase02_Imaging.py')
