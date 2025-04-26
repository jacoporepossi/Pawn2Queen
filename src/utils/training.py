import torch.nn as nn
import torch

def init_weights(m):
    """
    Initialize the weights of the model.
    This function applies Kaiming normal initialization to the weights of linear layers
    and sets the biases to zero.

    Args:
        m (nn.Module): The module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def topk_acc(logits, targets, k=5):
    """
    Calculate the top-k accuracy of the model predictions.
    This function computes the top-k accuracy by checking if the true label is among the top-k predicted labels.

    Args:
        logits (torch.Tensor): The model's output logits.
        targets (torch.Tensor): The true labels.
        k (int): The number of top predictions to consider.

    Returns:
        acc (float): The top-k accuracy.
    """
    with torch.no_grad():
        _, pred = torch.topk(logits, k, dim=1)              
        correct = pred.eq(targets.unsqueeze(1)).any(dim=1) 
        acc = correct.float().mean().item()
    return acc