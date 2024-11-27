import torch

def r2_score(outputs, prices):
    # Calculate the total sum of squares
    total_variance = torch.sum((prices - prices.mean())**2)
    
    # Calculate the residual sum of squares
    residual_variance = torch.sum((prices - outputs)**2)
    
    # Compute R2 score
    r2 = 1 - (residual_variance / total_variance)
    
    return r2.item()