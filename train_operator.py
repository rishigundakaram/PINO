import yaml
from argparse import ArgumentParser
import math
import torch
from torch.utils.data import DataLoader
from CGDs import BCGD, ACGD

from solver.random_fields import GaussianRF
from train_utils import Adam
from train_utils.datasets import NSLoader, online_loader, DarcyFlow
from train_utils.train_3d import mixed_train
from train_utils.train_2d import train_2d_operator, train_2d_operator_cgd
from models import FNN3d, FNN2d


def train_3d(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    # prepare dataloader for training with data
    if 'datapath2' in data_config:
        loader = NSLoader(datapath1=data_config['datapath'], datapath2=data_config['datapath2'],
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])
    else:
        loader = NSLoader(datapath1=data_config['datapath'],
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])

    train_loader = loader.make_loader(data_config['n_sample'],
                                      batch_size=config['train']['batchsize'],
                                      start=data_config['offset'],
                                      train=data_config['shuffle'])
    # prepare dataloader for training with only equations
    gr_sampler = GaussianRF(2, data_config['S2'], 2 * math.pi, alpha=2.5, tau=7, device=device)
    a_loader = online_loader(gr_sampler,
                             S=data_config['S2'],
                             T=data_config['T2'],
                             time_scale=data_config['time_interval'],
                             batchsize=config['train']['batchsize'])
    # create model
    print(device)
    model = FNN3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    # create optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    mixed_train(model,
                train_loader,
                loader.S, loader.T,
                a_loader,
                data_config['S2'], data_config['T2'],
                optimizer,
                scheduler,
                config,
                device,
                log=args.log,
                project=config['others']['project'],
                group=config['others']['group'])


def train_2d(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    print('loading data')
    dataset = DarcyFlow(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])
    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'], shuffle=True)
    print('loaded data')
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

    if config['model']['competitive'] == False: 
        optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                            lr=config['train']['base_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=config['train']['milestones'],
                                                        gamma=config['train']['scheduler_gamma'])
        train_2d_operator(model,
                        train_loader,
                        optimizer, scheduler,
                        config, rank=0, log=args.log,
                        project=config['others']['project'],
                        group=config['others']['group'],
                        entity=config['others']['entity'])
    else: 
        Regressor = FNN2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  activation=config['model']['activation']).to(device)
        Discriminator = FNN2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  activation=config['model']['activation'],
                  in_dim=4).to(device)
        optimizer = ACGD(max_params=Discriminator.parameters(), 
                        min_params=Regressor.parameters(), 
                        tol=1e-4, lr_max=config['train']['lr_max'], lr_min=config['train']['lr_min'])
        # optimizer = BCGD(max_params=Discriminator.parameters(), 
        #                 min_params=Regressor.parameters(), 
        #                 lr_min=config['train']['lr_min'], 
        #                 lr_max=config['train']['lr_max'],
        #                 momentum=config['train']['momentum'])
        train_2d_operator_cgd(Regressor, Discriminator, optimizer, train_loader,
                      config, rank=0, log=args.log, entity=config['others']['entity'])


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if 'name' in config['data'] and config['data']['name'] == 'Darcy':
        train_2d(args, config)
    else:
        train_3d(args, config)
