import torch as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as T
import torchattacks
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import copy
import imageio
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"