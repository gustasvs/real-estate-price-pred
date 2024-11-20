import numpy as np
import torch
from torch import nn

from collections import Counter

def compute_bin_weights(targets, num_bins=20):
    """
    Compute weights for each target based on the inverse frequency of the target
    
    Args:
        targets (list): List of targets
        num_bins (int): Number of bins to divide the targets into
    
    Returns:
        torch.Tensor: Weights corresponding to each target
        np.array: Weights corresponding to each bin
    """
    epsilon = 1e-9  # using epsilon because np.digitize is one side inclusive
    bins = np.linspace(min(targets) - epsilon, max(targets) + epsilon, num_bins + 1)
    
    print("bins", bins)

    bin_indices = np.digitize(targets, bins, right=True) - 1

    print("bin_indices", bin_indices)

    
    unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)
    bin_weights = 1.0 / bin_counts  # inverse frequency
    # bin_weights = bin_weights / np.sum(bin_weights)  # normalize

    # scale weights so that the max weight is 1
    bin_weights = bin_weights / np.max(bin_weights)

    print("bin weights: ", bin_weights)

    
    full_bin_weights = np.zeros(num_bins)
    full_bin_weights[unique_bins] = bin_weights
    
    return bins, full_bin_weights


class WeightedMSELoss(nn.Module):
    def __init__(self, bins, bin_weights, device='cpu'):
        super(WeightedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')  # No reduction for manual weighting
        self.bins = torch.tensor(bins, device=device)  # Ensure bins are a torch tensor on the correct device
        self.bin_weights = torch.tensor(bin_weights, device=device)  # Ensure bin weights are a torch tensor on the correct device
        self.device = device
    
    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss.
        
        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): True target values.
        
        Returns:
            torch.Tensor: Weighted MSE loss.
        """

        loss = self.mse(predictions, targets) * 10
        # print("loss", loss)
        # print("pred", predictions)
        # print("targets", targets)

        bin_indices = torch.bucketize(targets, self.bins, right=True) - 1  
        batch_weights = self.bin_weights[bin_indices]
        # print("batch_weights", batch_weights)

        weighted_loss = (loss * batch_weights).mean()
        # print("weighted_loss", weighted_loss)
        # exit(0)
        return weighted_loss