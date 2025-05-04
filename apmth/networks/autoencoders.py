from .base import BaseModel
from apmth.networks import MLP
from torch import nn
import torch.nn.functional as F
from typing import List, Optional
import torch
from apmth.env import Grid2OpEnvAdapter
from apmth.networks import GNNEncoder


class GNNAutoEncoder(BaseModel):

    def __init__(self, env: Grid2OpEnvAdapter, gcn_dims: List[int], residual: Optional[bool] = True):
        super().__init__()
        nx = env.observation_space.shape[0]
        nu = env.action_space.shape[0]

        self.u_encoder = nn.Linear(nu, gcn_dims[-1])
        self.decoder = MLP([gcn_dims[-1]*2] + gcn_dims[:-1]
                           [::-1] + [nx], activation=nn.SiLU)

        self.x_encoder = GNNEncoder(env, gcn_dims, residual)

    def compute_losses(self, input_dict, intermediate_dict):
        ae_loss = F.mse_loss(
            intermediate_dict['pred'], input_dict["next_state"])
        output_dict = {"optimized_loss": ae_loss}
        return output_dict

    def predict_step(self, input_dict, batch_idx=None):
        x_encoded = self.x_encoder(input_dict["state"])
        u_encoded = self.u_encoder(input_dict["action"])
        z = torch.cat([x_encoded, u_encoded], dim=-1)
        pred = self.decoder(z)
        return {'pred': pred}


class MLPAutoEncoder(BaseModel):

    def __init__(self, env: Grid2OpEnvAdapter, mlp_dims: List[int]):
        super().__init__()
        nx = env.observation_space.shape[0]
        nu = env.action_space.shape[0]
        self.x_encoder = MLP([nx] + mlp_dims, activation=nn.SiLU)
        self.u_encoder = nn.Linear(nu, mlp_dims[-1])
        self.decoder = MLP([mlp_dims[-1]*2] + mlp_dims[:-1]
                           [::-1] + [nx], activation=nn.SiLU)

    def compute_losses(self, input_dict, intermediate_dict):
        ae_loss = F.mse_loss(
            intermediate_dict['pred'], input_dict["next_state"])
        output_dict = {"optimized_loss": ae_loss}
        return output_dict

    def predict_step(self, input_dict, batch_idx=None):
        x, u = input_dict["state"], input_dict["action"]
        x_encoded = self.x_encoder(x)
        u_encoded = self.u_encoder(u)
        pred = self.decoder(torch.cat([x_encoded, u_encoded], dim=-1))
        return {'pred': pred}
