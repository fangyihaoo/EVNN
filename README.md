# Energetic Variational Neural Network (EVNN)
This repository contains the implementaion of the papers . To duplicate the results reported in the paper, follow the subsequent steps in order.

## Clone the repository and change your current directory:
```
git clone https://github.com/fangyihaoo/EVNN.git
cd EVNN
```

## Create a new conda environment using the default environment.yml:
```
conda env create
```
## Activate the default environment:
```
conda activate evnn
```

## An example for solving Allen-Cahn equation:
```
# Pretrain the model to fit the initial condition
python3 utils/pretrain.py pretrain_Allen --max_epoch=10000 --lr=1e-2 --lr_decay=0.7 --step_size=5000
# Train EVNN
python3 ./Tasks/AllenCahn.py train
# Plot the solution
python3 utils/visualizer.py 
```
## An example for solving Willmore flow: 

## Remark
Feel free to email fangyihao116@gmail.com if you have any questions and need help.
