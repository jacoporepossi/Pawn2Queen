import torch.nn as nn
import torch
import math
import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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

def init_transformer_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

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


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)