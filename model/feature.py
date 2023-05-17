from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from agent_constants import MAP_SIZE
from model.teacher_network import StateNetwork

class feature_net(nn.Module):
    def __init__(self, in_channels: int, feature_size: int, layers: int):
        super().__init__()
        self.state_net = StateNetwork(in_channels, feature_size, layers)

    def forward(self, states: torch.Tensor, maskings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.get_feature(states, maskings)

        return x

    def get_feature(self, states: torch.Tensor, maskings: torch.Tensor) -> torch.Tensor:
        return self.state_net(states, maskings)