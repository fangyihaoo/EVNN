#!/bin/csh
#$ -M zhu4@nd.edu     # Email address for job notification
#$ -m abe               # Send mail when job begins, ends and aborts
#$ -q gpu             
#$ -l gpu_card=1
#$ -N EVNNToyPoi

module load python

python ./Tasks/toyEVNN.py train --type='poissoncycle' --exact=poiss2dcycleexact.pt --grid=poiss2dcyclegrid.pt
# python ./Tasks/toyEVNN.py train --type='poi' --exact=poiss2dexact.pt --grid=poiss2dgrid.pt
