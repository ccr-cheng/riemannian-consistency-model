import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def cartesian_from_latlon(x):
    lat, lon = x[..., 0], x[..., 1]
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def latlon_from_cartesian(x):
    x, y, z = x[..., 0], x[..., 1], x[..., 2]
    lat = torch.asin(z)
    lon = torch.atan2(y, x)
    return torch.stack([lat, lon], dim=-1) * 180 / np.pi


class EarthData(Dataset):
    def __init__(self, root, filename, **kwargs):
        self.root = root
        self.filename = filename

        fn = os.path.join(root, filename)
        with open(fn, 'r') as file:
            lines = csv.reader(file)
            dataset = np.array(list(lines)).astype(np.float32)
        self.latlon = torch.from_numpy(dataset)
        self.data = cartesian_from_latlon(self.latlon / 180 * np.pi)
        self.data = self.data[:, None, :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def dimension(self):
        return self.data.shape[-1]


class Volcano(EarthData):
    def __init__(self, root, **kwargs):
        super().__init__(root, 'volc.csv', **kwargs)


class Earthquake(EarthData):
    def __init__(self, root, **kwargs):
        super().__init__(root, 'quakes.csv', **kwargs)


class Fire(EarthData):
    def __init__(self, root, **kwargs):
        super().__init__(root, 'fire.csv', **kwargs)


class Flood(EarthData):
    def __init__(self, root, **kwargs):
        super().__init__(root, 'flood.csv', **kwargs)
