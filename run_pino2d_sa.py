import yaml
from argparse import ArgumentParser
import random

import torch

from models import FNN2d
from train_utils import Adam
from train_utils.negadam import NAdam
from torch.utils.data import DataLoader
from train_utils.datasets import DarcyFlow
from train_utils.train_2d import train_2d_operator, train_2d_operator_competitive
from models.basics import SAWeightDarcy


def train(args, config, dataloader):
    model = FNN2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  activation=config['model']['activation']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                         lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    if 'self_adaptive' in config['model'].keys() and config['model']['self_adaptive'] is True: 
        weight_model = SAWeightDarcy((dataset.S-4, dataset.S-4)).to(device)
        weight_optimizer = NAdam(weight_model.parameters(), betas=(0.9, 0.999),
                         lr=config['train']['sa_lr'])
        weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(weight_optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
        train_2d_operator_competitive(model, weight_model,
                                    dataloader,
                                    optimizer, weight_optimizer, 
                                    scheduler, weight_scheduler,
                                    config, rank=0, log=args.log,
                                    project=config['others']['project'],
                                    group=config['others']['group'], 
                                    entity=config['others']['entity'])
    else: 
        train_2d_operator(model,
                        dataloader,
                        optimizer, scheduler,
                        config, rank=0, log=args.log,
                        project=config['others']['project'],
                        group=config['others']['group'], 
                        entity=config['others']['entity'])


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--start', type=int, help='Start index of test instance')
    parser.add_argument('--stop', type=int, help='Stop index of instances')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    seed = random.randint(1, 10000)
    print(f'Random seed: {seed}')
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = DarcyFlow(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])

    for i in range(args.start, args.stop):
        print(f'Start solving instance {i}')
        config['data']['offset'] = i
        dataset.resample(i)
        dataloader = DataLoader(dataset, batch_size=config['train']['batchsize'])
        train(args, config, dataloader)
    print(f'{args.stop - args.start} instances are solved')