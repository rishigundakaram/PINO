import sys
sys.path.append('/central/groups/tensorlab/rgundaka/code/PINO/')

from models.basics import SAWeightNS
import torch
import matplotlib.pyplot as plt
import numpy as np

base_dir = '/central/groups/tensorlab/rgundaka'
device = 0 if torch.cuda.is_available() else 'cpu'
checkpoints = [
    "/scripts/checkpoints/NS-direct/SA-PINO-direct-lr-1-weights.pt"
]
items = []
for check in checkpoints: 
    w_checkpoint = base_dir + check
    model = SAWeightNS(64, 64, 64).to(device)
    ckpt = torch.load(w_checkpoint)
    model.load_state_dict(ckpt['model'])

    init_param_weights = model.init_param.detach().cpu().numpy()
    boundary_param_weights = model.domain_param.detach().cpu().numpy()
    quantiles = [0, .25, .5, .75, 1]
    ic_q = np.quantile(init_param_weights, quantiles)
    b_q = np.quantile(boundary_param_weights, quantiles)
    # items.append(b_q[0]/5/ic_q[0])
    # items.append(b_q[1]/5/ic_q[1])
    print(ic_q)
    print(b_q)
    # items.append(b_q[3]/5/ic_q[3])
    # items.append(b_q[4]/5/ic_q[4])
