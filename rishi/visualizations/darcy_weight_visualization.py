import sys
sys.path.append('/central/groups/tensorlab/rgundaka/code/PINO/')

from models.basics import SAWeightDarcy
import torch
import matplotlib.pyplot as plt
import numpy as np

# Parameters
base_dir = '/central/groups/tensorlab/rgundaka'
checkpoint = base_dir + '/scripts/checkpoints/darcy-FDM/darcy-pino-sa-lr-10-nd-weights.pt'
# data_path = base_dir + '/code/data/piececonst_r421_N1024_smooth2.mat'
mesh_size = (57,57)

device = 0 if torch.cuda.is_available() else 'cpu'
print("loading model")
model = SAWeightDarcy(mesh_size=mesh_size).to(device)
ckpt = torch.load(checkpoint)
print(ckpt['model'])
model.load_state_dict(ckpt['model'])

weights = model.params.detach().cpu().numpy()
plt.imshow(weights)
plt.colorbar()
plt.savefig('../bin/Darcy/no_data/sa_weights_darcy')