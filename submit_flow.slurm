#!/bin/bash

#SBATCH --job-name=norm_flows 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00       
#SBATCH --constraint=gpu    
#SBATCH --qos=shared          
#SBATCH --account=m5058  
#SBATCH --mail-user=rwu4@stanford.edu
#SBATCH --mail-type=ALL     

module load pytorch

python train_norm_flow_conditional.py

