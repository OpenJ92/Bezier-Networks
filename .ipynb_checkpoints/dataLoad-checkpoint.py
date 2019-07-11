import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from dataTransform import *

class NavBotData(Dataset):

    def __init__(self, npy_file, root_dir, transform=transforms.Compose([NGRAMDomain(32), MINMAX()])):
        self.series_frame = np.load(root_dir + '/' + npy_file)
        self.root_dir = root_dir
        self.npy_file = npy_file
        self.transform = transform
        self.train, self.validataion = self.train_validation_indices(.8)

    def __len__(self):
        return len(self.series_frame)

    def __getitem__(self, idx):
        assert isinstance(idx, (int, tuple, slice, np.ndarray))
        def getter(element):
            sample = np.load(self.root_dir + '/' + str(element) + '.npy')
            return sample

        if isinstance(idx, int):
           sample = getter(idx)

        if isinstance(idx, tuple):
            sample = np.stack([getter(element) for element in idx], axis=0)

        if isinstance(idx, slice):
            start = idx.start if idx.start else 0
            step = idx.step if idx.step else 1
            stop = idx.stop if idx.stop else len(self)
            sample = np.stack([getter(element) for element in range(start, stop, step)], axis=0)

        if isinstance(idx, np.ndarray):
            idx = idx.flatten()
            sample = np.stack([getter(element) for element in idx], axis=0)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def train_validation_indices(self, per):
        len_ = len(self)
        train_val_per = int(len_*per)
        samples = np.arange(len_)
        random.shuffle(samples)
        return samples[:train_val_per], samples[train_val_per:]

    def batch_indices(self):
        return np.random.choice(self.train, 100)


if __name__ == '__main__':
    Xs = NavBotData('seriesID.npy', 'train_X/standard')
    X1 = NavBotData('seriesID.npy', 'train_X/tensor_order_1')
    XT = NavBotData('seriesID.npy', 'train_X/tensor_order_T')
    y = NavBotData('seriesID.npy', 'train_y/standard', transform = transforms.Compose([labelEncoder('labelEncodery.joblib'),NGRAMRange(32, 128)]))

