#!/usr/bin/bash

# Create a list of obs blocks
#/local_data/0/utomo.6/casa-release-5.0.0-218.el6/bin/casa -c get_obs_block.py
filename="testBPdcals.txt"

# Iterate over each obs block
while read -r line
do
        # Create a directory for calibration
        obs_block=$line
        echo $obs_block
        rm -rf ../$obs_block
	mkdir ../$obs_block
	cd ../$obs_block
        # Create a script for calibration
	echo "code_dir = '/local_data/0/utomo.6/historic_vla/codes/'" >> $obs_block.py
	echo "pipe_dir = '/local_data/0/utomo.6/vla_pipeline/pipeline5.0.0/'" >> $obs_block.py
        echo "obs_block = '$obs_block'" >> $obs_block.py
	echo "SDM_name = '$obs_block.ms'" >> $obs_block.py
        echo "execfile(code_dir+'obs_block.py')" >> $obs_block.py # to concat and add intents
        echo "execfile(pipe_dir+'EVLA_pipeline.py')" >> $obs_block.py # calibration script
        # Execute calibration script in that directory
        /local_data/0/utomo.6/casa-release-5.0.0-218.el6/bin/casa -c $obs_block.py
done < "$filename"
