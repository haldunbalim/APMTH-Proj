
from typing import Optional
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        feature_sizes: list,
        activation: type = nn.ReLU,
        use_layer_norm: bool = False,
    ):
        def block(i, o, final=False):
            ls = [nn.Linear(i, o)]
            if use_layer_norm and not final:
                ls.append(nn.LayerNorm(o))
            if not final:
                ls.append(activation())
            return nn.Sequential(*ls)
        super().__init__()
        self.layers = nn.ModuleList([block(i, o, final=l==len(feature_sizes)-2) 
                                     for l, (i, o) in enumerate(zip(feature_sizes[:-1], feature_sizes[1:]))])
        self.use_layer_norm = use_layer_norm
        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
