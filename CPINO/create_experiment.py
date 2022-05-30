import itertools
from venv import create
import yaml
import os
from pprint import pprint

base_dir='/groups/tensorlab/rgundaka/code/PINO/'
experiment_dir='CPINO/experiments'
experiment_name='Darcy_cpino_batchsize_test'

base_config_train = os.path.join(base_dir, experiment_dir, 'Darcy_cpino/configs/train/Darcy-train.yaml')
base_config_test = os.path.join(base_dir, experiment_dir, 'Darcy_cpino/configs/test/Darcy-test.yaml')


def paths(cur_dict):
    all_paths = []
    for key, value in cur_dict.items(): 
        if type(value) == list: 
            all_paths.append([{key: val} for val in cur_dict[key]])
        else:
            found = paths(cur_dict[key])
            all_paths.append([{key: i} for i in found])
    all_paths = itertools.product(*all_paths)
    all_params = []
    for i in all_paths: 
        cur = {}
        for j in i: 
            cur.update(j)
        all_params.append(cur)
    return all_params

def update_config(config, params): 
    for key, val in params.items(): 
        if type(val) != dict: 
            config[key] = val
        else: 
            update_config(config[key], params[key])
    return config

def create_sh(path, params, nodes=1, time="24:00:00", name="CPINO"): 
    n_tasks = len(params)
    with open(path, 'w+') as file: 
        file.write(
f"""#!/bin/bash
#SBATCH --time={time}  # walltime
#SBATCH --ntasks={n_tasks}   # number of processor cores (i.e. tasks)
#SBATCH --nodes={nodes}   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "{name}"  # job name
#SBATCH --mail-user=rgundaka@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
"""
        )
        for idx in range(n_tasks):
            train_str =  f'python {os.path.join(base_dir, "train_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, experiment_name, "configs/train/Darcy-train")}-{idx}.yaml'
            file.write(f"srun -n 1 --nodes=1 {train_str} &\n")
        file.write('wait\n')
        for idx in range(n_tasks):
            test_str = f'python {os.path.join(base_dir, "eval_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, experiment_name, "configs/test/Darcy-test")}-{idx}.yaml'
            file.write(f"srun -n 1 --nodes=1 {test_str} &\n")
        file.write('wait\n')


if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name)):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name))
if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name, 'configs')):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name, 'configs'))
if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name, 'configs/train')):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name, 'configs/train'))
if not os.path.exists(os.path.join(base_dir, experiment_dir, experiment_name, 'configs/test')):
    os.mkdir(os.path.join(base_dir, experiment_dir, experiment_name, 'configs/test'))

param_grid = {
    'train': {
        'lr_min': [.01], 
        'lr_max': [.01],
        'epochs': [25], 
        'batchsize': [30, 64, 128, 256, 512, 1024]
    }
}
# param_grid = {
#     'train': {
#         'lr_min': [1, .1, .01, .001, .0001], 
#         'lr_max': [1, .1, .01, .001, .0001],
#         'epochs': [25],
#     }
# }

params = list(paths(param_grid))

with open(base_config_train, 'r') as stream: 
    config_train = yaml.load(stream, yaml.FullLoader)

with open(base_config_test, 'r') as stream: 
    config_test = yaml.load(stream, yaml.FullLoader)

for idx, param in enumerate(params): 
    cur_train_config = update_config(config_train, param)
    cur_train_config['train']['save_name'] = f'darcy-cpino-{idx}.pt'
    cur_path = os.path.join(base_dir, experiment_dir, experiment_name)
    with open(os.path.join(cur_path, f'configs/train/Darcy-train-{idx}.yaml'), 'w') as outfile:
            yaml.dump(cur_train_config, outfile)

    cur_test_config = config_test
    cur_test_config['test']['ckpt'] = os.path.join(cur_path, f'checkpoints/darcy-cpino-{idx}.pt')
    with open(os.path.join(cur_path, f'configs/test/Darcy-test-{idx}.yaml'), 'w') as outfile:
            yaml.dump(cur_test_config, outfile)
    

slurm_path = os.path.join(base_dir, experiment_dir, experiment_name, 'run.sh')
create_sh(slurm_path, params, nodes=6, time="6:00:00", name="CPINO 2D LR search")    
# print(f'python {os.path.join(base_dir, "train_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, "train/Darcy-train")}-{idx}.yaml')
# print(f'python {os.path.join(base_dir, "eval_operator.py")} --log --config_path {os.path.join(base_dir, experiment_dir, "train/Darcy-test")}-{idx}.yaml')




