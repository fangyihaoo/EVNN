#!/bin/csh
#$ -M zhu4@nd.edu     # Email address for job notification
#$ -m abe               # Send mail when job begins, ends and aborts
#$ -q gpu             
#$ -l gpu_card=1
#$ -N EVNNhPhaseField

module load python

python ./Tasks/toyEVNN.py train --type='poissoncycle' --exact=poiss2dcycleexact.pt --grid=poiss2dcyclegrid.pt
