import numpy as np
import torch
from torch import nn

from collections import Counter

import matplotlib.pyplot as plt

from config.settings import DEMO_MODE

def plot_loss_demonstration(self):
    plt.figure(figsize=(18, 6))

    # First plot: MSE Loss by difference from 0 to 1
    differences = np.linspace(0, 1, 100)
    mse_losses = differences ** 2
    plt.subplot(1, 3, 1)
    plt.plot(differences, mse_losses, 'o-', label='MSE Loss')
    plt.title('MSE Loss by Difference')
    plt.xlabel('Difference (Predictions - Targets)')
    plt.ylabel('Loss')

    # Second plot: Bin weights
    weights = self.bin_weights.detach().cpu().numpy()
    plt.subplot(1, 3, 2)
    plt.bar(range(len(weights)), weights)
    plt.title('Weight Bins Distribution')
    plt.xlabel('Bin Index')
    plt.ylabel('Weight')
    plt.annotate('Least weight has the highest count of values', (np.argmin(weights), min(weights)), 
                 textcoords="offset points", xytext=(0,90), ha='center', fontsize=12, color='g', backgroundcolor='w')
    # a line at the least weight
    plt.axvline(x=np.argmin(weights), color='r', linestyle='--')

    # Third plot: Example final losses for difference = 0.3
    difference_example = 0.3
    example_losses = difference_example ** 2 * weights
    example_targets = [0.1, 0.9]  # Assuming these map to certain bins
    example_bin_indices = [int(example_targets[0] * len(weights)), int(example_targets[1] * len(weights))]
    final_losses = example_losses[example_bin_indices]

    plt.subplot(1, 3, 3)
    plt.bar(['Target: 0.1', 'Target: 0.9'], final_losses)
    plt.title('Example Final Losses for Difference = 0.3')
    plt.xlabel('Target Values')
    plt.ylabel('Final Weighted Loss')

    plt.tight_layout()
    plt.show()



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
    
    # print("bins", bins)

    bin_indices = np.digitize(targets, bins, right=True) - 1

    # print("bin_indices", bin_indices)

    
    unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)
    bin_weights = 1.0 / bin_counts  # inverse frequency
    # bin_weights = bin_weights / np.sum(bin_weights)  # normalize

    # scale weights so that the max weight is 1
    bin_weights = bin_weights / np.max(bin_weights)

    # print("bin weights: ", bin_weights)

    
    full_bin_weights = np.zeros(num_bins)
    full_bin_weights[unique_bins] = bin_weights

    if DEMO_MODE:
        plt.figure(figsize=(10, 6))
        plt.bar(range(num_bins), full_bin_weights, alpha=0.6, label='Bin Weights')
        plt.xlabel('Bin Index')
        plt.ylabel('Weight')
        plt.title('Bin Weights Distribution')
        plt.legend()
        plt.show()
    
    return bins, full_bin_weights


class WeightedMSELoss(nn.Module):
    def __init__(self, bins, bin_weights, device='cpu'):
        super(WeightedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bins = torch.tensor(bins, device=device)
        self.bin_weights = torch.tensor(bin_weights, device=device)
        self.device = device

        # if DEMO_MODE:
        #     plot_loss_demonstration(self)
    
    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss.
        
        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): True target values.
        
        Returns:
            torch.Tensor: Weighted MSE loss.
        """

        loss = self.mse(predictions, targets) * 100
        # print("loss", loss)
        # print("pred", predictions)
        # print("targets", targets)

        bin_indices = torch.bucketize(targets, self.bins, right=True) - 1  
        batch_weights = self.bin_weights[bin_indices]

        # print("targets", targets)
        # print("bin_indices", bin_indices)
        # print("batch_weights", batch_weights)

        weighted_loss = (loss * batch_weights).mean()
        # print("weighted_loss", weighted_loss)
        # exit(0)
        return weighted_loss