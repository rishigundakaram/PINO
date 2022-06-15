#!/bin/bash
#SBATCH --time=1:00:00  # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=2   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "NS_initial_test"  # job name
#SBATCH --mail-user=rgundaka@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
srun -n 1 --exclusive --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/NS_initial_test/configs/train/NS-train-0.yaml &
srun -n 1 --exclusive --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/train_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/NS_initial_test/configs/train/NS-train-1.yaml &
wait
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/NS_initial_test/configs/test/NS-test-0.yaml &
srun -n 1 --nodes=1 python /groups/tensorlab/rgundaka/code/PINO/eval_operator.py --log --config_path /groups/tensorlab/rgundaka/code/PINO/CPINO/experiments/NS_initial_test/configs/test/NS-test-1.yaml &
wait