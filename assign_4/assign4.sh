#!/bin/bash
#SBATCH --job-name=assignment_4 # Job name
#SBATCH --output=assignment4_2.out # Name of output file
#SBATCH --cpus-per-task=2 # Schedule one core
#SBATCH --time=01:00:00 # Run time (hh:mm:ss)
#SBATCH --gres=gpu
#SBATCH --mail-type=BEGIN,END,FAIL # E-mail when status changes
#SBATCH --partition=brown

module load Anaconda3
source activate /opt/itu/condaenv/cs/rapu/synth_transformers

cd induction_heads_assignment  
python train_transformer.py --n_epoch=25  
python run_transformer.py  
