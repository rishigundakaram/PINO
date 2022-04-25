import sys
sys.path.append('/central/groups/tensorlab/rgundaka/code/PINO/')

from train_utils import DarcyFlow
from train_utils.losses import LpLoss
from models import FNN2d

import torch
from torch.utils.data import DataLoader
from train_utils.losses import FDM_Darcy

import numpy as np
import matplotlib.pyplot as plt

# Data related params
base_dir = '/central/groups/tensorlab/rgundaka'
checkpoint = base_dir + '/scripts/checkpoints/darcy-FDM/darcy-pino-sa-lr-1000-nd.pt'
data_path = base_dir + '/code/data/piececonst_r421_N1024_smooth1.mat'
total_num = 1024
offset = 0 
n_sample = 1000 
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
def loss(pred, y): 
    return torch.div((pred - y)**2, y**2)

def darcy_loss(u, a, loss):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    return (Du-f)**2

def reject_outliers(data):
    m = 4
    u = np.mean(data)
    print(u)
    s = np.std(data)
    areas = np.where(np.abs(u-data)/s > m)
    data[areas] = u
    return data

lploss = LpLoss(d=2, p=2, size_average=False, reduction=False)


data_loss = torch.zeros([1, dataset.S, dataset.S]).to(device)
physics_loss = torch.zeros([1, dataset.S-4, dataset.S-4]).to(device)
print(mesh.size())
with torch.no_grad(): 
    for x,y in dataloader: 
        x, y = x.to(device), y.to(device)
        pred = model(x)
        pred = pred.reshape(y.shape)
        pred = pred * mollifier
        cur_data_loss = loss(pred, y)
        data_loss = torch.add(data_loss, cur_data_loss)
        
        a = x[..., 0]
        cur_physics_loss = darcy_loss(pred, a, lploss)
        physics_loss = torch.add(physics_loss, cur_physics_loss)


data_loss = data_loss.reshape([dataset.S, dataset.S]).cpu().numpy()
data_loss = data_loss / len(dataloader)

physics_loss = physics_loss.reshape([dataset.S-4, dataset.S-4]).cpu().numpy()
physics_loss = physics_loss / len(dataloader)


[a, b] = np.shape(data_loss)
data_loss = data_loss[1:a-1, 1:b-1]
physics_loss = physics_loss[1:a-1, 1:b-1]

physics_loss = reject_outliers(physics_loss)

plt.imshow(data_loss)
plt.colorbar()
plt.savefig('../bin/Darcy/no_data/darcy_data_loss_sapino_train_no_out')
plt.clf()

plt.imshow(physics_loss)
plt.colorbar()
plt.title('Darcy relative error')
plt.savefig('../bin/Darcy/no_data/darcy_physics_loss_sapino_train_no_out')



