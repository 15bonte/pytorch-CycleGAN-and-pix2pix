"""This module contains simple helper functions """
from __future__ import print_function
from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image
import os
from skimage.transform import resize

from cnn_framework.utils.tools import save_tiff



def tensor2im(input_image, mean_std, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        
        # Apply reverse normalization
        if imtype == np.uint16:
            factor = 65535.0
        elif imtype == np.uint8:
            factor = 255.0
        else:
            raise NotImplementedError("Only uint8 and uint16 are supported for now")
        image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) * mean_std["std"] + mean_std["mean"]) * factor, 0, factor)

        # # Original code
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    save_tiff(image_numpy, image_path.replace('.png', '.tiff'), original_order="YXZ")
    return

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

# Function to detect padding
def detect_padding(padded_image):
    def find_edge(values):
        first_diff = np.where(values != values[0])[0]
        assert first_diff.size > 0
        return first_diff[0] - 1

    # Detect padding on each side
    top_pad = find_edge(padded_image[:, padded_image.shape[1] // 2]) + 1
    bottom_pad = find_edge(padded_image[::-1, padded_image.shape[1] // 2]) + 1
    left_pad = find_edge(padded_image[padded_image.shape[0] // 2, :]) + 1
    right_pad = find_edge(padded_image[padded_image.shape[0] // 2, ::-1]) + 1

    return top_pad, bottom_pad, left_pad, right_pad


def adapt_image(image, test_opt, pad_size, padding):
    image = np.moveaxis(image, -1, 0) # CYX
    
    # Get the padding widths
    if padding is None:
        top_pad, bottom_pad, left_pad, right_pad = detect_padding(image[0])
        padding = (top_pad, bottom_pad, left_pad, right_pad)
    else:
        top_pad, bottom_pad, left_pad, right_pad = padding

    # Slice the original image from the padded array
    unpad_img = image[
        ..., top_pad : -bottom_pad or None, left_pad : -right_pad or None
    ]
    
    # Resize
    resized_img = (resize(unpad_img, (unpad_img.shape[0], int(unpad_img.shape[1] * pad_size / test_opt.crop_size), int(unpad_img.shape[2] * pad_size / test_opt.crop_size))) * 65535).astype(np.uint16)

    # plt.subplot(2, 2, 1)
    # plt.imshow(image[0])
    # plt.subplot(2, 2, 2)
    # plt.imshow(unpad_img[0])
    # plt.subplot(2, 2, 3)
    # plt.imshow(resized_img[0])
    # plt.show()

    # Back to YXC
    return np.moveaxis(resized_img, 0, -1), padding