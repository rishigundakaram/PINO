import yaml

config = '/groups/tensorlab/rgundaka/code/PINO/rishi/configs/Darcy/finetune/Darcy-finetune-normal-op-sa.yaml'

out_dir = '/groups/tensorlab/rgundaka/code/PINO/rishi/configs/Darcy/finetune/'
ckpt_dir = '/groups/tensorlab/rgundaka/scripts/checkpoints/darcy-FDM/'

prepend = 'Darcy-finetune'
ckpt = {'op-PINO': 'darcy-pino-nd.pt',
        'op-SA-PINO-10': 'darcy-pino-sa-lr-10-nd.pt'}
finetune = {
    'fine-PINO': [False, 0], 
    'fine-SA-PINO-100': [True, 10], 
    'fine-SA-PINO-10': [True, 1],
    'fine-SA-PINO-1': [True, .1],
    'fine-SA-PINO-01': [True, .01],
    'fine-SA-PINO-001': [True, .001],
}
sh_file = "python /groups/tensorlab/rgundaka/code/PINO/run_pino2d_sa.py --log --config_path "
start_stop = " --start 0 --stop 20"
total = ""
with open(config, 'r') as stream: 
    org = yaml.load(stream, yaml.FullLoader)

for key_op, val_op in ckpt.items():
    for key_fine, val_fine in finetune.items(): 
        out = out_dir + prepend + "-" + key_op + "-" + key_fine + ".yaml"
        org['train']['ckpt'] = ckpt_dir + val_op
        org['model']['self_adaptive'] = val_fine[0]
        org['train']['sa_lr'] = val_fine[1]
        with open(out, 'w') as outfile:
            yaml.dump(org, outfile)
        total += sh_file + out + start_stop + "\n" 
print(total)