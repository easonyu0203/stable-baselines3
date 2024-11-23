import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-7, shape: Tuple[int, ...] = ()) -> None:
        """
        Initializes the RunningMeanStd module.

        Args:
            epsilon (float): A small value to initialize the count for numerical stability.
            shape (Tuple[int, ...]): The shape of the mean and variance tensors.
        """
        super(RunningMeanStd, self).__init__()
        # Non-trainable parameters for mean, variance, and count
        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float64))
        self.register_buffer('count', torch.tensor(epsilon, dtype=torch.float64))
        self.epsilon = epsilon

    def update(self, x: torch.Tensor) -> None:
        """
        Update the running mean and variance with a batch of data.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, ...).
        """
        # Ensure input is a PyTorch tensor and matches device and dtype
        x = x.to(dtype=torch.float64, device=self.mean.device)

        # Compute batch statistics
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        # Update the statistics using batch moments
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int
    ) -> None:
        """
        Update the running statistics from batch moments.

        Args:
            batch_mean (torch.Tensor): Mean of the batch.
            batch_var (torch.Tensor): Variance of the batch.
            batch_count (int): Number of samples in the batch.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Update mean
        self.mean += delta * batch_count / total_count

        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count

        # Update count
        self.count = total_count

    def normalize(self, x: torch.Tensor, baise=0.0, n_std: float = 1.0) -> torch.Tensor:
        """
        Normalize the input using the running mean and standard deviation.

        Args:
            x (torch.Tensor): Input data to be normalized.
            baise (float): Baise to add to the normalized data.
            n_std (int): Number of standard deviations to normalize.

        Returns:
            torch.Tensor: Normalized data.
        """
        return (x - self.mean) / (torch.sqrt(self.var) * n_std + self.epsilon) + baise



class FeatureCNN(nn.Module):
    def __init__(self, input_shape, convfeat=32, rep_size=512):
        super(FeatureCNN, self).__init__()
        c, h, w = input_shape

        # Using nn.Sequential for better readability
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, convfeat, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(convfeat, convfeat * 2, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(convfeat * 2, convfeat * 2, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        
        # Calculate output size after the convolutional layers
        conv_out_size = self._get_conv_out((c, h, w))
        
        # Fully connected layer to generate feature representation
        self.fc = nn.Linear(conv_out_size, rep_size)


    def _get_conv_out(self, shape):
        """Helper to compute the size after convolution layers."""
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



    

    