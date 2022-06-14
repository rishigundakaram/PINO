#!/bin/bash
#SBATCH --time=24:00:00  # walltime
#SBATCH --ntasks=7   # number of processor cores (i.e. tasks)
#SBATCH --nodes=7   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "Darcy_cpino_tolerance"  # job name
#SBATCH --mail-user=rgundaka@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/train/Darcy-train-0.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/train/Darcy-train-1.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/train/Darcy-train-2.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/train/Darcy-train-3.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/train/Darcy-train-4.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/train/Darcy-train-5.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/train/Darcy-train-6.yaml &
wait
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/test/Darcy-test-0.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/test/Darcy-test-1.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/test/Darcy-test-2.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/test/Darcy-test-3.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/test/Darcy-test-4.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/test/Darcy-test-5.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/Darcy_cpino_tolerance/configs/test/Darcy-test-6.yaml &
wait
