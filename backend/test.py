import numpy as np
import matplotlib.pyplot as plt
import torch

def compute_bin_weights(targets, num_bins=20):
    bins = np.linspace(min(targets), max(targets), num_bins + 1)
    bin_indices = np.digitize(targets, bins, right=True) - 1
    unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)
    bin_weights = 1.0 / bin_counts  # Inverse frequency
    bin_weights = bin_weights / bin_weights.sum()  # Normalize weights
    full_bin_weights = np.zeros(num_bins)
    full_bin_weights[unique_bins] = bin_weights
    sample_weights = torch.tensor([full_bin_weights[bin_idx] for bin_idx in bin_indices], dtype=torch.float32)
    return sample_weights, full_bin_weights

# Example data
prices = np.random.normal(50, 20, size=1000)  # Simulated price data
weights, weight_bins = compute_bin_weights(prices, num_bins=20)

# Bin edges from linspace for consistent plotting
bins = np.linspace(min(prices), max(prices), 21)

fig, ax = plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(prices, bins=bins)  # Histogram with specific bins
ax[1].bar(bins[:-1], weight_bins, align='edge', width=np.diff(bins))  # Bar plot for weights

ax[1].set_xlabel('Price')
ax[1].set_ylabel('Normalized Inverse Frequency Weights')
plt.show()
