from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from agent_constants import MAP_SIZE

class PolicyNetwork(nn.Module):
    def __init__(self,  feature_size: int, num_unit_actions: int, num_citytile_actions: int,map_size: int = MAP_SIZE):
        super().__init__()
        self.feature_size = feature_size
        self.num_actions = num_unit_actions + num_citytile_actions
        self.num_unit_actions = num_unit_actions
        self.num_citytile_actions = num_citytile_actions
        self.fc1 = torch.nn.Sequential(
            nn.Linear(self.feature_size, self.num_actions)
        )
    def forward(self,feature) -> Tuple[torch.Tensor, torch.Tensor]:
        x = feature
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # x.shape == (B, H, W, C)
        x = x.reshape(-1, c)  # x.shape == (B * H * W, C)
        
        out = self.fc1(x)  # out.shape == (B * H * W, out_dim)

        out = out.reshape(b, h, w, -1)  # out.shape == (B, H, W, out_dim)
        out1, out2 = torch.split(out, [self.num_unit_actions, self.num_citytile_actions], dim=-1)
        #critic_value=self.out(critic_value)
        return out1,out2


    def act(self, feature: torch.Tensor, targets: torch.Tensor):
        def _get_action_index_and_probs(x: torch.Tensor):
            probs = F.softmax(x, dim=-1)  # x.shape == (B, H, W, out_dim)
            actions = x.argmax(dim=-1)  # actions.shape == (B, H, W)

            log_prob = self.get_log_prob(probs, actions, targets)
            return actions, probs, log_prob
        # targets.shape == (B, H, W)
        x, y = self.forward(feature)
        return _get_action_index_and_probs(x), _get_action_index_and_probs(y)

    def get_log_prob(self, probs: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor):
        probs = torch.gather(probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # probs.shape == (B, H, W)

        referenced = torch.ones_like(probs)
        probs = torch.where(targets == 1, probs, referenced)

        prob = torch.prod(probs)
        log_prob = torch.log(prob)

        return log_prob
class Critic(nn.Module):
    def __init__(self,feature_size: int,map_size: int = MAP_SIZE,):
        super().__init__()
        self.actions_states=torch.nn.Sequential(
            nn.Conv2d(in_channels=feature_size, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(map_size*map_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )
    def forward(self, states): 
        states=states.detach()  

        x=self.actions_states(states)
        b,c,h,w=x.shape
        x=x.reshape(b,-1)
        x=self.out(x)

        return x
class Discriminator(nn.Module):
    def __init__(self,feature_size: int,num_actions,map_size: int = MAP_SIZE):
        super().__init__()
        self.actions_states=torch.nn.Sequential(
            nn.Conv2d(in_channels=num_actions+feature_size, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(map_size*map_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )
    def forward(self, action_out,states):
        action_out=action_out.detach()
        action_out=action_out.permute(0, 3, 1, 2)
        states=states.detach()

        cat = torch.cat([action_out, states], dim=1)
        x=self.actions_states(cat)
        b,c,h,w=x.shape
        x=x.reshape(b,-1)
        x=self.out(x)

        return x
