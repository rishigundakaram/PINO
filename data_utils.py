import scipy.io
import numpy as np
from scipy.interpolate import griddata
from pyDOE import lhs
from torch.utils.data import Dataset


def load_data(datapath, N_f=10000):
    '''
    Parameters:
        - datapath: datapath of (x,t), u(x,t) data
        - N_f: number of (x, t) 
    Return:
        - X_u: (N, 2) ndarray
        - u: (N , 1) ndarray
        - X_f: (N_f + N, 2) ndarray

    '''
    data = scipy.io.loadmat(datapath)

    t = data['t'].flatten()[:, None]  # (100,1)
    x = data['x'].flatten()[:, None]  # (256, 1)
    Exact = np.real(data['usol']).T  # (100, 256)
    # print(x.shape)
    # print(t.shape)
    # print(Exact.shape)
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]

    X_u = np.vstack([xx1, xx2, xx3])
    X_f = lb + (ub-lb)*lhs(2, N_f)
    X_f = np.vstack((X_f, X_u))
    u = np.vstack([uu1, uu2, uu3])

    return X_u, u, X_f, X_star, u_star


def sample(X_u, u, N=100):
    '''
    Randomly sample N pairs  
    '''
    idx = np.random.choice(X_u.shape[0], N, replace=False)
    X_u = X_u[idx, :]
    u = u[idx, :]
    return X_u, u


class BurgerData(Dataset):
    '''
    members: 
        - t, x, Exact: raw data
        - X, T: meshgrid 
        - X_star, u_star: flattened (x, t), u array
        - lb, ub: lower bound and upper bound vector
        - X_u, u: boundary condition data (x, t), u
    '''

    def __init__(self, datapath):
        data = scipy.io.loadmat(datapath)

        # raw 2D data
        self.t = data['t'].flatten()[:, None]  # (100,1)
        self.x = data['x'].flatten()[:, None]  # (256, 1)
        self.Exact = np.real(data['usol']).T  # (100, 256)

        # Flattened sequence
        self.get_flatten_data()
        self.get_boundary_data()

    def __len__(self):
        return self.Exact.shape[0]

    def __getitem__(self, idx):
        return self.X_star[idx], self.u_star[idx]

    def get_flatten_data(self):
        X, T = np.meshgrid(self.x, self.t)
        self.X, self.T = X, T
        self.X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.u_star = self.Exact.flatten()[:, None]

        # lower bound of (x, t): 2-dimensional vector
        self.lb = self.X_star.min(0)
        # upper bound of (x, t): 2-dimensional vector
        self.ub = self.X_star.max(0)

    def get_boundary_data(self):
        xx1 = np.hstack((self.X[0:1, :].T, self.T[0:1, :].T))
        uu1 = self.Exact[0:1, :].T
        xx2 = np.hstack((self.X[:, 0:1], self.T[:, 0:1]))
        uu2 = self.Exact[:, 0:1]
        xx3 = np.hstack((self.X[:, -1:], self.T[:, -1:]))
        uu3 = self.Exact[:, -1:]
        self.X_u = np.vstack([xx1, xx2, xx3])
        self.u = np.vstack([uu1, uu2, uu3])

    def sample_xt(self, N=10000):
        '''
        Sample (x, t) pairs within the boundary
        Return:
            - X_f: (N, 2) array
        '''
        X_f = self.lb + (self.ub-self.lb)*lhs(2, N)
        X_f = np.vstack((X_f, self.X_u))
        return X_f

    def sample_xu(self, N=100):
        '''
        Sample N points from boundary data
        Return: 
            - X_u: (N, 2) array 
            - u: (N, 1) array
        '''
        idx = np.random.choice(self.X_u.shape[0], N, replace=False)
        X_u = self.X_u[idx, :]
        u = self.u[idx, :]
        return X_u, u