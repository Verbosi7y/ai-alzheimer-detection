'''
    AlzheimerDataset.py -- Custom PyTorch Dataset for reading .npz images.
    Authors: Darwin Xue
'''
import torch
import numpy as np


class AlzheimerDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path=None, samples=None):
        self.npz_path = npz_path
        self.images = None
        self.labels = None
        self.data_size = None
        
        self.init_constructor(samples);

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if idx >= self.data_size: return None, None
        return self.images[idx], self.labels[idx]
    
    def init_constructor(self, samples):
        if samples is None:
            xy = np.load(self.npz_path)

            self.images = torch.from_numpy(np.expand_dims(xy["images"], axis=1)).float() / 255.0
            self.labels = torch.from_numpy(xy["labels"])
        
        else:
            self.images = torch.from_numpy(np.expand_dims(samples["X"], axis=1)).float() / 255.0
            self.labels = torch.from_numpy(samples["y"])

        self.data_size = self.images.shape[0]