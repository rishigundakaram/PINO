#!/bin/bash
#SBATCH --time=6:00:00  # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=6   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "CPINO 2D LR search"  # job name
#SBATCH --mail-user=rgundaka@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_pino/configs/train/Darcy-train-0.yaml &
wait
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_pino/configs/test/Darcy-test-0.yaml &
wait
