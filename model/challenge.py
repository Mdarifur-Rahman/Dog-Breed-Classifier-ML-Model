"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Challenge CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed



__all__ = ["Challenge"]


class Challenge(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # define each layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 8, kernel_size=5, stride=2, padding=2)
        self.fc_1 = nn.Linear(32, 2)

        # self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        # self.fc_1 = nn.Linear(256, 2)

        self.init_weights()


    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            # initialize the parameters for the convolutional layers
            std = sqrt(1/(5 * 5 * conv.in_channels))
            nn.init.normal_(conv.weight, mean=0.0, std=std)
            nn.init.constant_(conv.bias, 0.0)

        # initialize the parameters for [self.fc_1]
        std =  sqrt(1/32)
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc_1.bias, 0.0)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        # forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = x.view(N, -1)

        x = self.fc_1(x)

        return x



        
    # def __init__(self) -> None:
    #     """Define model architecture."""
    #     super().__init__()

    #     self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
    #     self.conv2 = nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2)
    #     self.conv3 = nn.Conv2d(64, 8, kernel_size=5, stride=2, padding=2)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    #     self.fc1 = nn.Linear(32, 2)


    #     self.init_weights()


    # def init_weights(self) -> None:
    #     """Initialize model weights."""
    #     set_random_seed()

    #     for conv in [self.conv1, self.conv2, self.conv3]:
    #         # initialize the parameters for the convolutional layers
    #         std = sqrt(1/(conv.kernel_size[0] * conv.kernel_size[1] *conv.in_channels))
    #         nn.init.normal_(conv.weight, mean=0.0, std=std)
    #         nn.init.constant_(conv.bias, 0.0)
        
    #     # initialize the parameters for [self.fc1]
    #     # fc1: input 32 -> output 16
    #     std= sqrt(1.0/32.0)
    #     nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
    #     nn.init.constant_(self.fc1.bias, 0.0)


    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Perform forward propagation for a batch of input examples. Pass the input array
    #     through layers of the model and return the output after the final layer.

    #     Args:
    #         x: array of shape (N, C, H, W) 
    #             N = number of samples
    #             C = number of channels
    #             H = height
    #             W = width

    #     Returns:
    #         z: array of shape (1, # output classes)
    #     """
    #     N, C, H, W = x.shape

    #     x = F.relu(self.conv1(x))
    #     x = self.pool(x)

    #     x = F.relu(self.conv2(x))
    #     x = self.pool(x)

    #     x = F.relu(self.conv3(x))
    #     # shape is (N, 8, 2, 2) => flatten to size (N, 8*2*2) = (N, 32)

    #     x = x.view(N, -1)  # (N, 32)

    #     x = self.fc1(x)
    #     # shape: (N, 2)
    #     return x



