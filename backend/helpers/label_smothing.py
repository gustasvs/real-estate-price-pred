import numpy as np
from scipy.ndimage import gaussian_filter1d

def apply_lds(results_array, sigma=1.0):
    """
    Apply Label Distribution Smoothing (LDS) on the results array.
    Args:
        results_array (numpy array): The array of target values.
        sigma (float): The standard deviation for the Gaussian kernel.
    Returns:
        numpy array: Smoothed results array.
    """
    smoothed_labels = gaussian_filter1d(results_array, sigma=sigma)
    return smoothed_labels

def apply_fds(features_array, results_array, sigma=1.0):
    """
    Apply Feature Distribution Smoothing (FDS) by aligning features to smoothed labels.
    Args:
        features_array (numpy array): The array of features.
        results_array (numpy array): The array of target values.
        sigma (float): The standard deviation for the Gaussian kernel.
    Returns:
        numpy array: Adjusted features array.
    """
    smoothed_labels = apply_lds(results_array, sigma)
    # Adjust features proportionally to the change in labels
    adjusted_features = features_array * (smoothed_labels / (results_array + 1e-9))
    return adjusted_features


# TODO
def adaptive_lds(results_array, density_threshold=0.1, max_sigma=10, min_sigma=0.5):
    # Compute density
    density = np.histogram(results_array, bins=30, density=True)[0]
    sample_density = np.interp(results_array, np.linspace(min(results_array), max(results_array), 30), density)

    # Adjust sigma inversely to density
    sigmas = np.clip(1 / (sample_density + 1e-9), min_sigma, max_sigma)
    smoothed_labels = np.array([gaussian_filter1d(results_array, sigma=s) for s in sigmas])
    return np.mean(smoothed_labels, axis=0)
