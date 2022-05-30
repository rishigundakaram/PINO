#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "CPINO 2D DARCY"   # job name
#SBATCH --mail-user=rgundaka@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

base_dir='/groups/tensorlab/rgundaka'
experiment_dir= 'code/PINO/CPINO/experiments/Darcy_pino/configs'
python $base_dir/code/PINO/train_operator.py --log --config_path $base_dir/$experiment_dir/train/Darcy-train.yaml
python $base_dir/code/PINO/eval_operator.py --log --config_path $base_dir/$experiment_dir/test/Darcy-test.yaml