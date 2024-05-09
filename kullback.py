import torch.nn.functional as F
import torch


def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence between two tensors p and q.
    """
    return torch.sum(p * (torch.log(p) - torch.log(q))) / p.shape[0]

def calc_kl(source, target):
    p_sm = F.softmax(source, dim=1)
    q_sm = F.softmax(target, dim=1)
    
    kl = kl_divergence(p_sm, q_sm)
        
    return kl