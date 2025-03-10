import json
import os

from matplotlib import pyplot as plt
import numpy as np
from data.base_dataset import BaseDataset, compute_mean_std, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

import torch
from cnn_framework.utils.readers.tiff_reader import TiffReader
from cnn_framework.utils.enum import ProjectMethods
from cnn_framework.utils.readers.utils.projection import Projection


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        
        # Compute mean & std
        self.mean_std_A = compute_mean_std(os.path.join(opt.dataroot, "trainA"), [opt.channel_A], list(range(opt.input_nc))) # SiR-DNA channel
        self.mean_std_B = compute_mean_std(os.path.join(opt.dataroot, "trainB"), [opt.channel_B], list(range(opt.output_nc))) # DAPI channel
        
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1), mean_std=self.mean_std_A, pad_size=opt.pad_size_A)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), mean_std=self.mean_std_B, pad_size=opt.pad_size_B)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        # New code
        A_img = torch.from_numpy(TiffReader(A_path, project=[Projection(
                        method=ProjectMethods.Channel,
                        channels=[self.opt.channel_A], # SiR-DNA channel
                        axis=1,  # channels
                    )]).get_processed_image().squeeze()) # SiR-DNA
        B_img = torch.from_numpy(TiffReader(B_path, project=[Projection(
                        method=ProjectMethods.Channel,
                        channels=[self.opt.channel_B], # DAPI channel
                        axis=1,  # channels
                    )]).get_processed_image().squeeze())  # DAPI
        
        # Original
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
