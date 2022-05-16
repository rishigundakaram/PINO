from locale import normalize
import numpy as np
import torch
from tqdm import tqdm
from .utils import save_checkpoint
from .losses import LpLoss, darcy_loss, PINO_loss, weighted_darcy_loss

try:
    import wandb
except ImportError:
    wandb = None


def train_2d_operator(model,
                      train_loader,
                      optimizer, scheduler,
                      config,
                      rank=0, loss_fn = darcy_loss, 
                      log=False,
                      project='PINO-2d-default',
                      group='default',
                      tags=['default'],
                      entity='rishigundakaram',
                      use_tqdm=True,
                      profile=False):
    '''
    train PINO on Darcy Flow
    Args:
        model:
        train_loader:
        optimizer:
        scheduler:
        config:
        rank:
        log:
        project:
        group:
        tags:
        use_tqdm:
        profile:

    Returns:

    '''
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity=entity,
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    mesh = train_loader.dataset.mesh
    mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    mollifier = mollifier.to(rank)
    for e in pbar:
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)

            optimizer.zero_grad()

            pred = model(x).reshape(y.shape)
            pred = pred * mollifier

            data_loss = myloss(pred, y)

            a = x[..., 0]
            f_loss = loss_fn(pred, a)

            loss = data_weight * data_loss + f_weight * f_loss
            loss.backward()
            optimizer.step()

            loss_dict['train_loss'] += loss.item() * y.shape[0]
            loss_dict['f_loss'] += f_loss.item() * y.shape[0]
            loss_dict['data_loss'] += data_loss.item() * y.shape[0]

        scheduler.step()
        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val = loss_dict['f_loss'] / len(train_loader.dataset)
        data_loss_val = loss_dict['data_loss'] / len(train_loader.dataset)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                    f'data loss: {data_loss_val:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'train loss': train_loss_val,
                    'f loss': f_loss_val,
                    'data loss': data_loss_val
                }
            )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    if wandb and log:
        run.finish()
    print('Done!')

def train_2d_operator_cgd(regressor,
                      discriminator, 
                      optimizer,
                      train_loader,
                      config,
                      rank=0, 
                      log=False,
                      project='CGD-PINO',
                      group='Darcy',
                      tags=['default'],
                      use_tqdm=True,
                      entity='rishigundakaram'):
    '''
    train PINO on Darcy Flow
    Args:
        model:
        train_loader:
        optimizer:
        scheduler:
        config:
        rank:
        log:
        project:
        group:
        tags:
        use_tqdm:
        profile:

    Returns:

    '''
    if rank == 0 and wandb and log:
        print('here')
        run = wandb.init(project=project,
                         entity=entity,
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    regressor.train()
    discriminator.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    mesh = train_loader.dataset.mesh
    mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    mollifier = mollifier.to(rank)
    for e in pbar:
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss_w': 0.0,
                     'f_loss_uw': 0.0,
                     'test_error': 0.0}
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)

            optimizer.zero_grad()

            pred = regressor(x).reshape(y.shape)
            pred = pred * mollifier

            data_loss = myloss(pred, y)

            a = x[..., 0]
            x_w = x[:, 2:-2, 2:-2]
            pred_w = torch.unsqueeze(pred[:, 2:-2, 2:-2], -1)

            x_w = torch.cat((x_w,pred_w), dim=3)
            w = torch.squeeze(discriminator(x_w), dim=-1)
            f_loss_w, f_loss_uw = weighted_darcy_loss(pred, a, w)

            loss = data_weight * data_loss + f_weight * f_loss_w
            optimizer.step(loss)

            loss = data_weight * data_loss + f_weight * f_loss_uw

            loss_dict['train_loss'] += loss.item() * y.shape[0]
            loss_dict['f_loss_w'] += f_loss_w.item() * y.shape[0]
            loss_dict['f_loss_uw'] += f_loss_uw.item() * y.shape[0]
            loss_dict['data_loss'] += data_loss.item() * y.shape[0]

        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val_w = loss_dict['f_loss_w'] / len(train_loader.dataset)
        f_loss_val_uw = loss_dict['f_loss_uw'] / len(train_loader.dataset)
        data_loss_val = loss_dict['data_loss'] / len(train_loader.dataset)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss_w: {f_loss_val_w:.5f}, '
                    f'f_loss_uw: {f_loss_val_uw:.5f}, '
                    f'data loss: {data_loss_val:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'train loss': train_loss_val,
                    'f loss': f_loss_val_uw,
                    'f loss weighted': f_loss_val_w,
                    'data loss': data_loss_val
                }
            )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    regressor)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'][:-3] + "-weights.pt",
                    discriminator)
    if wandb and log:
        run.finish()
    print('Done!')

def train_2d_burger(model,
                    train_loader,
                    optimizer, scheduler,
                    config,
                    rank=0, log=False,
                    project='PINO-2d-default',
                    group='default',
                    tags=['default'],
                    use_tqdm=True):
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    for e in pbar:
        model.train()
        train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)
            out = model(x).reshape(y.shape)
            data_loss = myloss(out, y)

            loss_u, loss_f = PINO_loss(out, x[:, 0, :, 0])
            total_loss = loss_u * ic_weight + loss_f * f_weight + data_loss * data_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()
        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        train_loss /= len(train_loader)
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5f} '
                    f'train f error: {train_pino:.5f}; '
                    f'data l2 error: {data_l2:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': train_pino,
                    'Train L2 error': data_l2,
                    'Train loss': train_loss,
                }
            )

        if e % 100 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')