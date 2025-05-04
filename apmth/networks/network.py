from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from .utils import MLP
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
    
class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, env: Grid2OpEnvAdapter, 
                 gcn_dims: List[int], residual: Optional[bool] = True):
        super().__init__(observation_space, features_dim=gcn_dims[-1])

        self.graph_fn = env.get_obs_graph

        gcn_dims = [env.node_dim] + list(gcn_dims)
        if residual:
            self.gcn_blocks = nn.ModuleList([ResidualGCNBlock(i, o) for i, o in zip(gcn_dims[:-1], gcn_dims[1:])])
        else:
            self.gcn_blocks = nn.ModuleList([GCNConv(i, o) for i, o in zip(gcn_dims[:-1], gcn_dims[1:])])
        

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

class MLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, env: Grid2OpEnvAdapter, mlp_dims: List[int]):
        super().__init__(observation_space, features_dim=mlp_dims[-1])

        mlp_dims = [observation_space.shape[0]] + list(mlp_dims)
        self.mlp = MLP(mlp_dims, activation=nn.SiLU)

    def forward(self, obs):
        return self.mlp(obs)