import torch
import torch.nn.functional as F

def gelu(x):
    return F.gelu(x)

def swish(x):
    return x * torch.sigmoid(x)

def gelu_new(x):
    import math
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
