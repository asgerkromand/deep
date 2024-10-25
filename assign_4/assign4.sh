#!/bin/bash
#SBATCH --job-name=fine_tune_star_wars # Job name
#SBATCH --output=results_star_wars.out # Name of output file
#SBATCH --cpus-per-task=2 # Schedule one core
#SBATCH --time=01:00:00 # Run time (hh:mm:ss)
#SBATCH --gres=gpu:2
#SBATCH --mail-type=BEGIN,END,FAIL # E-mail when status changes
#SBATCH --partition=brown

module load Anaconda3
source activate /opt/itu/condaenv/cs/rapu/synth_transformers
