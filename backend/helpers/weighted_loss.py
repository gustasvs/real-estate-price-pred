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
    sample_weights = torch.tensor([full_bin_weights[bin_idx] for bin_idx in bin_indices], dtype=torch.float32)
    return sample_weights, full_bin_weights


class WeightedMSELoss(nn.Module):
    def __init__(self, weights, device='cpu'):
        super(WeightedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')  # No reduction for manual weighting
        self.weights = weights.to(device)
    
    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss.
        
        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): True target values.
        
        Returns:
            torch.Tensor: Weighted MSE loss.
        """

        loss = self.mse(predictions, targets)
        # print("pred", predictions)
        # print("targets", targets)
        batch_weights = self.weights[targets.long()]
        # print("batch_weights", batch_weights)
        weighted_loss = (loss * batch_weights).mean()
        return weighted_loss