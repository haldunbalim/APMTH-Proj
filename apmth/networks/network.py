from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from gymnasium import spaces

from typing import Optional, List
from apmth.env.env_adapter import Grid2OpEnvAdapter

class ResidualGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.SiLU()

        # Project input to match output shape if needed
        self.residual_proj = nn.Linear(
            in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        out = self.gcn(x, edge_index)
        out = self.norm(out)
        res = self.residual_proj(x)
        return self.activation(out + res)
    
class GNNEncoder(nn.Module):
    def __init__(self, env: Grid2OpEnvAdapter, gcn_dims: List[int], residual: Optional[bool] = True):
        super().__init__()

        self.graph_fn = env.get_obs_graph

        gcn_dims = [env.node_dim] + list(gcn_dims)
        if residual:
            self.gcn_blocks = nn.ModuleList(
                [ResidualGCNBlock(i, o) for i, o in zip(gcn_dims[:-1], gcn_dims[1:])])
        else:
            self.gcn_blocks = nn.ModuleList(
                [GCNConv(i, o) for i, o in zip(gcn_dims[:-1], gcn_dims[1:])])

    def batched_graph_fn(self, obs):
        from torch_geometric.data import Data, Batch
        node_feats, edge_indices = self.graph_fn(obs)
        data_list = []
        batch_size = node_feats.size(0)

        for i in range(batch_size):
            x = node_feats[i]                          # [14, 7]
            edge_index = edge_indices[i]               # [2, 20]

            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        return Batch.from_data_list(data_list)

    def forward(self, obs):
        batch_data = self.batched_graph_fn(obs)
        x = batch_data.x
        edge_index = batch_data.edge_index
        batch = batch_data.batch

        for block in self.gcn_blocks:
            x = block(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, out_dim]
        return x
    

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
        self.layers = nn.ModuleList([block(i, o, final=l == len(feature_sizes)-2)
                                     for l, (i, o) in enumerate(zip(feature_sizes[:-1], feature_sizes[1:]))])
        self.use_layer_norm = use_layer_norm

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
