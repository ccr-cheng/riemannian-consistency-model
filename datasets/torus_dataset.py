import math

import pandas
import numpy as np
import torch
from torch.utils.data import Dataset


class BoardTorusDataset(Dataset):
    def __init__(self, N, seed=42, **kwargs):
        self.N = N
        self.seed = seed
        self.data = self.generate_all_samples(seed, N)
        self.data = self.wrap(self.data)
        self.data = self.data[:, None, :]

    @staticmethod
    def generate_all_samples(seed, N):
        generator = torch.Generator()
        generator.manual_seed(seed)

        x1 = torch.rand(N, generator=generator) * 4 - 2
        x2_ = (torch.rand(N, generator=generator) - torch.randint(high=2, size=(N,), generator=generator) * 2)
        x2 = x2_ + (torch.floor(x1) % 2)
        data = torch.cat([x1[:, None], x2[:, None]], dim=1)

        return data.float()

    def wrap(manifold, samples):
        return samples % (2 * torch.pi)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def dimension(self):
        return self.data.shape[-1]


class ProteinAngles(Dataset):
    def __init__(self, root, **kwargs):
        self.root = root
        self.data = torch.tensor(self.read_tsv(), dtype=torch.float32)
        self.data = self.wrap(self.data)
        self.data = self.data[:, None, :]

    def read_tsv(self):
        # 'source', 'phi', 'psi', 'amino'
        df = pandas.read_csv(self.root, sep='\t', header=None)
        col_1 = df[1] / 180 * math.pi + math.pi
        col_2 = df[2] / 180 * math.pi + math.pi
        return np.stack([col_1.to_numpy(), col_2.to_numpy()], axis=-1)

    def wrap(self, samples):
        return samples % (2 * torch.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def dimension(self):
        return self.data.shape[-1]


class RNAAngles(Dataset):
    def __init__(self, root, **kwargs):
        self.root = root
        self.data = torch.tensor(self.read_tsv(), dtype=torch.float32)
        self.data = self.wrap(self.data)
        self.data = self.data[:, None, :]

    def read_tsv(self):
        # 'source', 'base', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi'
        df = pandas.read_csv(self.root, sep='\t', header=None)
        cols = []
        for i in range(2, len(df.columns)):
            cols.append(df[i] / 180 * math.pi + math.pi)
        return np.stack(cols, axis=-1)

    def wrap(self, samples):
        return samples % (2 * torch.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def dimension(self):
        return self.data.shape[-1]
