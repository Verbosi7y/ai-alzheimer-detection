'''
    AlzheimerDataset.py -- Custom PyTorch Dataset for reading .npz images.
    Authors: Darwin Xue
'''
import torch
import numpy as np

from . import Dataset 


class AlzheimerDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, transform=None):
        self.npz_path = npz_path
        self.transform = transform

    def __len__(self):
        with np.load(self.npz_path) as data:
            return len(data["images"])

    def __getitem__(self, idx):
        with np.load(self.npz_path) as data:
            image = data["images"][idx]
            label = data["labels"][idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def setTransform(self, transform): self.transform = transform

    def load_dataset_npz(self, npz_path):
        np_dataset = np.load(npz_path)
        return np_dataset["images"], np_dataset["labels"]