#!/bin/csh
#$ -M zhu4@nd.edu     # Email address for job notification
#$ -m abe               # Send mail when job begins, ends and aborts
#$ -q gpu             
#$ -l gpu_card=1
#$ -N EVNNToyPoiCycleDritz

module load python


# python ./Tasks/toyEVNN.py train --type='poissoncycle' --exact=poiss2dcycleexact.pt --grid=poiss2dcyclegrid.pt

# python ./Tasks/toyEVNN.py train --type='poi' --exact=poiss2dexact.pt --grid=poiss2dgrid.pt

# python ./Tasks/toyDritz.py train --type='poi' --exact=poiss2dexact.pt --grid=poiss2dgrid.pt 

# python ./Tasks/toyDritz.py train --type='poissoncycle' --exact=poiss2dcycleexact.pt --grid=poiss2dcyclegrid.pt 

# python ./Tasks/toyPinn.py train --type='poissoncycle' --exact=poiss2dcycleexact.pt --grid=poiss2dcyclegrid.pt 

# python ./Tasks/toyPinn.py train --type='poi' --exact=poiss2dexact.pt --grid=poiss2dgrid.pt 

# python ./Tasks/Heat.py train 

# python ./Tasks/AllenCahn.py train 

python ./Tasks/Fokker.py train
