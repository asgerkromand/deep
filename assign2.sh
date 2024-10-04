#!/bin/bash
#SBATCH --job-name=fine_tune_star_wars # Job name
#SBATCH --output=results_star_wars.out # Name of output file
#SBATCH --cpus-per-task=2 # Schedule one core
#SBATCH --time=01:00:00 # Run time (hh:mm:ss)
#SBATCH --gres=gpu
#SBATCH --mail-type=BEGIN,END,FAIL # E-mail when status changes
#SBATCH --partition=brown

module load singularity
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
singularity exec --nv /opt/itu/containers/pytorchtransformers/pytorch-24.07-py3-transformers.sif python3 version.py
singularity exec --nv /opt/itu/containers/pytorchtransformers/pytorch-24.07-py3-transformers.sif pip install accelerate
singularity exec --nv /opt/itu/containers/pytorchtransformers/pytorch-24.07-py3-transformers.sif python3 -m torch.distributed.launch assign_2/ca/run_t5_mlm_torch.py --train_file wookiepedia_scrape.txt --output_dir ~ --validation_split_percentage 1 --model_name_or_path google/flan-t5-base --max_seq_length 512 --do_train --do_eva
