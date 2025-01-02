import torch

def r2_score(outputs, prices):

    outputs = outputs.detach()  # ensure no gradients are involved
    prices = prices.detach()  # ensure no gradients are involved
    

    total_variance = torch.sum((prices - prices.mean())**2)
    
    residual_variance = torch.sum((prices - outputs)**2)
    
    r2 = 1 - (residual_variance / total_variance)
    
    return r2.item()