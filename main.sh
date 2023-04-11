#!/bin/csh
#$ -M yfang5@nd.edu     # Email address for job notification
#$ -m abe               # Send mail when job begins, ends and aborts
#$ -q gpu             
#$ -l gpu_card=1
#$ -N circleLamda10

echo "This job is running on the host $HOSTNAME"
echo "This job was assigned GPU: $CUDA_VISIBLE_DEVICES"
module load python/3.7.3

# python ./Tasks/toyEVNN.py train --type='poissoncycle' --exact=poiss2dcycleexact.pt --grid=poiss2dcyclegrid.pt

# python ./Tasks/toyEVNN.py train --type='poi' --exact=poiss2dexact.pt --grid=poiss2dgrid.pt

# python ./Tasks/toyDritz.py train --type='poi' --exact=poiss2dexact.pt --grid=poiss2dgrid.pt 

# python ./Tasks/toyDritz.py train --type='poissoncycle' --exact=poiss2dcycleexact.pt --grid=poiss2dcyclegrid.pt 

# python ./Tasks/toyPinn.py train --type='poissoncycle' --exact=poiss2dcycleexact.pt --grid=poiss2dcyclegrid.pt 

# python ./Tasks/toyPinn.py train --type='poi' --exact=poiss2dexact.pt --grid=poiss2dgrid.pt 

# python ./Tasks/Heat.py train 

#python3 ./Tasks/AllenCahn.py train

#python3 ./Tasks/MeanCur.py train --max_epoch=1000 --pretrain="ellipseInitilizationLFBGS.pt"

#python3 ./Tasks/MeanCur.py train --max_epoch=1000 --pretrain="dumbbellInitilizationLFBGS.pt"

#python3 ./Tasks/MeanCur.py train --max_epoch=1000 --pretrain="squareInitilizationLFBGS.pt"

python3 ./Tasks/MeanCur.py train --max_epoch=2000 --pretrain="circleInitilizationLFBGS.pt"

#python3 ./Tasks/MeanCur.py train --max_epoch=1000 --pretrain="twoEllipseInitilizationLFBGS.pt"

#python ./Tasks/Fokker2d.py train

#python3 utils/pretrain.py pretrain_Mean3d --max_epoch=20000 --lr_decay=0.7 --step_size=5000 --lr=1e-2

#python3 ./Tasks/Willmore3d.py train --max_epoch=200
