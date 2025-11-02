import os

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RotationDataset(Dataset):
    def __init__(self, N, noise=1., proj_y=10., seed=42, **kwargs):
        self.N = N
        self.noise = noise
        self.proj_y = proj_y
        self.seed = seed
        self.data = self.generate_all_rots()
        self.data = self.data[:, None, :]

    def generate_all_rots(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        t = 3 * np.pi * (1 - torch.rand(self.N, generator=generator))
        x = t * torch.cos(t)
        z = t * torch.sin(t)
        x += self.noise * torch.randn(self.N, dtype=torch.float, generator=generator)
        z += self.noise * torch.randn(self.N, dtype=torch.float, generator=generator)

        target = F.normalize(torch.stack([x, torch.ones_like(x) * self.proj_y, z], dim=1), dim=1)
        source = torch.tensor([0, 1, 0], dtype=torch.float).unsqueeze(0)
        axis = torch.cross(source, target, dim=1)
        theta = torch.acos(torch.clamp(torch.sum(source * target, dim=1), -1.0, 1.0))
        return F.normalize(axis, dim=-1) * theta.unsqueeze(1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def dimension(self):
        return self.data.shape[-1]


class RawDataset(Dataset):
    def __init__(self, root, category, split='train', max_sample=100000, **kwargs):
        assert category in ['cone', 'fisher24', 'line', 'peak']
        self.root = root
        self.split = split
        self.category = category
        self.max_sample = max_sample
        data = np.load(os.path.join(self.root, f'{category}_{split}.npy'))[:max_sample]
        data = Rotation.from_matrix(data).as_rotvec()
        self.data = torch.from_numpy(data).float().unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def dimension(self):
        return self.data.shape[-1]
