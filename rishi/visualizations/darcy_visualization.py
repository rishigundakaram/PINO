import sys
sys.path.append('/central/groups/tensorlab/rgundaka/code/PINO/')

from train_utils import DarcyFlow
from train_utils.losses import LpLoss
from models import FNN2d

import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# Data related params
base_dir = '/central/groups/tensorlab/rgundaka'
checkpoint = base_dir + '/scripts/checkpoints/darcy-FDM/darcy-pretrain-fno.pt'
data_path = base_dir + '/code/data/piececonst_r421_N1024_smooth2.mat'
total_num = 1024
offset = 0 
n_sample = 500 
nx = 421
sub = 7
shuffle = False
batch_size = 1

# model related params
modes_1 = [20, 20, 20, 20]
modes_2 = [20, 20, 20, 20]
fc_dim = 128
layers = [64, 64, 64, 64, 64]
activation = 'gelu'

device = 0 if torch.cuda.is_available() else 'cpu'

# Dataset and DataLoader
print("Loading Data")
dataset = DarcyFlow(data_path, nx=nx, sub=sub, offset=offset , num=total_num)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print("loading model")
model = FNN2d(modes1=modes_1, modes2=modes_2, fc_dim=fc_dim, layers=layers,
                activation=activation).to(device)
ckpt = torch.load(checkpoint)
model.load_state_dict(ckpt['model'])

# evaluating model
model.eval()
mesh = dataloader.dataset.mesh
mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
mollifier = mollifier.to(device)

# defining loss functions: 
def data_Loss(pred, y, p=2): 
    return torch.div(torch.abs(pred - y), y)

lploss = LpLoss(d=2, p=2, reduction=False)

data_loss = torch.zeros([1, dataset.S, dataset.S]).to(device)
print(mesh.size())
with torch.no_grad(): 
    for x,y in dataloader: 
        x, y = x.to(device), y.to(device)
        pred = model(x)
        pred = pred.reshape(y.shape)
        pred = pred * mollifier
        cur_data_loss = data_Loss(pred, y)
        data_loss = torch.add(data_loss, cur_data_loss)

print(data_loss.size())
data_loss = data_loss.reshape([dataset.S, dataset.S]).cpu().numpy()
data_loss = data_loss / total_num
[a, b] = np.shape(data_loss)
data_loss = data_loss[1:a-1, 1:b-1]
print(data_loss)
print(np.shape(data_loss))
plt.matshow(data_loss)
plt.savefig('darcy_loss_pino')


