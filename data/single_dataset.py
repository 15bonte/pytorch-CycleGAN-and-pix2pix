import json
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

import torch
from cnn_framework.utils.readers.tiff_reader import TiffReader
from cnn_framework.utils.enum import ProjectMethods
from cnn_framework.utils.readers.utils.projection import Projection

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        
        if os.path.isfile(opt.mean_std_path):
            with open(opt.mean_std_path, "r") as mean_std_file:
                mean_std = json.load(mean_std_file)
        else:
            mean_std = {}

        self.transform = get_transform(opt, grayscale=(input_nc == 1), mean_std=mean_std, pad_size=opt.pad_size)

        print("Careful: dedicated to generate SiR-DNA images from DAPI.")

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]

        # New code
        A_img = torch.from_numpy(TiffReader(A_path, project=[Projection(
                        method=ProjectMethods.Channel,
                        channels=[3], # DAPI channel
                        axis=1,  # channels
                    )]).get_processed_image().squeeze())
        
        # Original
        # A_img = Image.open(A_path).convert('RGB')
        
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
