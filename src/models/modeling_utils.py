from transformers.modeling_utils import PreTrainedModel
try:
    from transformers.modeling_utils import prune_linear_layer
except ImportError:
    def prune_linear_layer(layer, index, dim=0):
        import torch
        import torch.nn as nn
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        new_layer = nn.Linear(W.shape[1], W.shape[0], bias=layer.bias is not None)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            b = layer.bias.index_select(0, index).clone().detach()
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer
