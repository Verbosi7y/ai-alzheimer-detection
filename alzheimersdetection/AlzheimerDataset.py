'''
    AlzheimerDataset.py -- Custom PyTorch Dataset for reading .npz images.
    Authors: Darwin Xue
'''
import torch
import cv2
import numpy as np


class AlzheimerDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, resize=False):
        self.npz_path = npz_path
        self.xy = np.load(self.npz_path)
        self.images = torch.from_numpy(np.expand_dims(self.xy["images"], axis=1)).float() / 255.0
        self.labels = torch.from_numpy(self.xy["labels"])
        self.data_size = self.images.shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if idx >= self.data_size: return None, None
        return self.images[idx], self.labels[idx]