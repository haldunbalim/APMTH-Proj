from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from typing import Optional, List
from apmth.env.env_adapter import Grid2OpEnvAdapter
from apmth.networks import MLP, GNNEncoder
import torch
import torch.nn as nn

class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, env: Grid2OpEnvAdapter,
                 gcn_dims: List[int], residual: Optional[bool] = True):
        super().__init__(observation_space, features_dim=gcn_dims[-1])
        self.encoder = GNNEncoder(env, gcn_dims, residual)

    def forward(self, obs):
        return self.encoder(obs)


class MLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, env: Grid2OpEnvAdapter, mlp_dims: List[int]):
        super().__init__(observation_space, features_dim=mlp_dims[-1])

        mlp_dims = [observation_space.shape[0]] + list(mlp_dims)
        self.mlp = MLP(mlp_dims, activation=nn.SiLU)

    def forward(self, obs):
        return self.mlp(obs)
