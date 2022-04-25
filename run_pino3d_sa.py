from locale import normalize
import random
import yaml

import torch
from torch.utils.data import DataLoader

from train_utils import Adam, NSLoader, get_forcing
from train_utils.negadam import NAdam
from train_utils.train_3d import train_sa, train

from models.basics import SAWeightNS 
from models import FNN3d
from argparse import ArgumentParser
from train_utils.utils import requires_grad


def run_instance(loader, config, data_config):
    trainset = loader.make_dataset(data_config['n_sample'],
                                   start=data_config['offset'])
    train_loader = DataLoader(trainset, batch_size=config['train']['batchsize'])
    model = FNN3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers']).to(device)

    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    if 'twolayer' in config['train'] and config['train']['twolayer']:
        requires_grad(model, False)
        requires_grad(model.sp_convs[-1], True)
        requires_grad(model.ws[-1], True)
        requires_grad(model.fc1, True)
        requires_grad(model.fc2, True)
        params = []
        for param in model.parameters():
            if param.requires_grad == True:
                params.append(param)
    else:
        params = model.parameters()

    beta1 = config['train']['beta1'] if 'beta1' in config['train'] else 0.9
    beta2 = config['train']['beta2'] if 'beta2' in config['train'] else 0.999
    optimizer = Adam(params, betas=(beta1, beta2),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    forcing = get_forcing(loader.S).to(device)
    if 'self_adaptive' in config['model'].keys() and config['model']['self_adaptive'] is True: 
        weight_model = SAWeightNS(nx=data_config['nx'], ny=data_config['nx'], nt=data_config['nt'], 
                                  sub=data_config['sub'], sub_t=data_config['sub_t'], 
                                  normalize=config['train']['sa_normalize']).to(device)
        weight_optimizer = NAdam(weight_model.parameters(), betas=(0.9, 0.999),
                         lr=config['train']['sa_lr'])
        weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
        train_sa(model, weight_model,
            loader, train_loader,
            optimizer, scheduler,
            weight_optimizer, weight_scheduler, 
            forcing, config,
            rank=0,
            log=options.log,
            project=config['others']['project'],
            group=config['others']['group'],
            entity=config['others']['entity'],
            use_tqdm=True,
            profile=True)

    else: 
        train(model,
            loader, train_loader,
            optimizer, scheduler,
            forcing, config,
            rank=0,
            log=options.log,
            project=config['others']['project'],
            group=config['others']['group'],
            entity=config['others']['entity'],
            use_tqdm=True,
            profile=True)


if __name__ == '__main__':
    
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--start', type=int, help='Start index of test instance')
    parser.add_argument('--stop', type=int, help='Stop index of instances')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    options = parser.parse_args()

    config_file = options.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    print('loading data')
    loader = NSLoader(datapath1=data_config['datapath'],
                      nx=data_config['nx'], nt=data_config['nt'],
                      sub=data_config['sub'], sub_t=data_config['sub_t'],
                      N=data_config['total_num'],
                      t_interval=data_config['time_interval'])
    for i in range(options.start, options.stop):
        print('Start training on instance %d' % i)
        config['data']['offset'] = i
        data_config['offset'] = i
        # seed = random.randint(1, 10000)
        seed = 1000
        print(f'Random seed :{seed}')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        run_instance(loader, config, data_config)
    print('Done!')
