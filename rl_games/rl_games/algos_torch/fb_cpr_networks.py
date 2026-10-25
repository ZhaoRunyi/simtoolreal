# BFM-Zero / Meta FB-CPR architecture and loss adaptation.
# Reference components: CC BY-NC 4.0; see docs/licenses/bfm-zero-LICENSE.
"""BFM-Zero residual networks and transition replay for SimToolReal.

Architecture follows LeCAR-Lab/BFM-Zero 05ad611, nn_models.py. The sensor
interface remains unchanged; a four-frame internal history replaces the LSTM.
"""

import math

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn import functional

from rl_games.algos_torch.fb_networks import project_task


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.Mish())

    def forward(self, inputs):
        return inputs + self.layers(inputs)


def block(input_size, output_size, activate=True):
    layers = [nn.LayerNorm(input_size), nn.Linear(input_size, output_size)]
    if activate:
        layers.append(nn.Mish())
    return nn.Sequential(*layers)


def embedding(input_size, width, depth):
    return nn.Sequential(
        block(input_size, width),
        *[ResidualBlock(width) for _ in range(depth - 2)],
        block(width, width // 2),
    )


class ResidualMap(nn.Module):
    def __init__(self, state_size, action_size, latent_size, output_size, width, depth):
        super().__init__()
        # The official ResidualForwardMap uses hidden_layers for both embeddings.
        self.task_embedding = embedding(state_size + latent_size, width, depth)
        self.action_embedding = embedding(state_size + action_size, width, depth)
        self.output = nn.Sequential(
            *[ResidualBlock(width) for _ in range(depth)],
            block(width, output_size, activate=False),
        )

    def forward(self, state, action, command):
        task = self.task_embedding(torch.cat((state, command), dim=-1))
        control = self.action_embedding(torch.cat((state, action), dim=-1))
        return self.output(torch.cat((control, task), dim=-1))


class HistoryActor(nn.Module):
    def __init__(self, observation_size, action_size, latent_size, config):
        super().__init__()
        self.observation_size = observation_size
        self.history_steps = config['history_steps']
        self.history_size = observation_size * self.history_steps
        input_size = self.history_size + action_size
        width, depth = config['residual_width'], config['residual_layers']
        self.task_embedding = embedding(input_size + latent_size, width, 2)
        self.observation_embedding = embedding(input_size, width, 2)
        self.policy = nn.Sequential(
            *[ResidualBlock(width) for _ in range(depth)],
            block(width, action_size, activate=False),
        )

    def forward(self, history, command):
        task = self.task_embedding(torch.cat((history, command), dim=-1))
        observation = self.observation_embedding(history)
        return self.policy(torch.cat((observation, task), dim=-1)).tanh()

    def sample(self, history, command, std, clip=None):
        action = self(history, command)
        noise = torch.randn_like(action) * std
        if clip is not None:
            noise = noise.clamp(-clip, clip)
        action = action + noise
        # TruncatedNormal's straight-through clipping, as used by BFM-Zero.
        return action + (action.clamp(-1 + 1e-6, 1 - 1e-6) - action).detach()


class BackwardMap(nn.Module):
    def __init__(self, state_size, latent_size):
        super().__init__()
        self.state_size = state_size
        self.layers = nn.Sequential(
            nn.Linear(state_size, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, latent_size),
        )

    def forward(self, state):
        return project_task(self.layers(state[..., :self.state_size]))


class CPRModel(nn.Module):
    def __init__(self, observation_size, state_size, action_size, config, network):
        super().__init__()
        latent_size = config['latent_size']
        self.actor = HistoryActor(observation_size, action_size, latent_size, config)
        self.backward_state_size = config.get('backward_state_size', state_size)
        self.backward = BackwardMap(self.backward_state_size, latent_size)
        critic_size = state_size + self.actor.history_size + action_size
        arguments = (critic_size, action_size, latent_size)
        width, depth = config['residual_width'], config['residual_layers']
        self.forward_maps = nn.ModuleList([
            ResidualMap(*arguments, latent_size, width, depth) for _ in range(2)
        ])
        self.auxiliary_maps = nn.ModuleList([
            ResidualMap(*arguments, 1, width, depth) for _ in range(2)
        ])
        self.style_maps = nn.ModuleList([
            ResidualMap(*arguments, 1, width, depth) for _ in range(2)
        ])
        self.discriminator = nn.Sequential(
            nn.Linear(self.backward_state_size + latent_size, 1024), nn.LayerNorm(1024), nn.Tanh(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1),
        )

    def initialize(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                parallel_map = name.startswith(('forward_maps.', 'style_maps.', 'auxiliary_maps.'))
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2) if parallel_map else 1.)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state, action, command):
        return torch.stack([model(state, action, command) for model in self.forward_maps])

    def auxiliary(self, state, action, command):
        return torch.stack([model(state, action, command).squeeze(-1) for model in self.auxiliary_maps])

    def style(self, state, action, command):
        return torch.stack([model(state, action, command).squeeze(-1) for model in self.style_maps])

    def discriminate(self, state, command):
        return self.discriminator(torch.cat((command, state[..., :self.backward_state_size]), dim=-1)).squeeze(-1)


class BatchNormalizer(nn.Module):
    """BN running statistics synchronized across ranks, without affine weights."""
    def __init__(self, size, momentum=.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(size))
        self.register_buffer('running_var', torch.ones(size))

    def forward(self, inputs):
        if self.training:
            with torch.no_grad():
                count = inputs.new_tensor([inputs.shape[0]])
                moments = torch.cat((inputs.sum(0), inputs.square().sum(0), count))
                if distributed.is_initialized():
                    distributed.all_reduce(moments)
                mean, square = moments[:-1].chunk(2)
                total = moments[-1]
                mean = mean / total
                variance = ((square - total * mean.square()) / (total - 1).clamp_min(1)).clamp_min(0)
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(variance, self.momentum)
        return (inputs - self.running_mean) / (self.running_var + 1e-5).sqrt()


class RewardNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(()))
        self.register_buffer('square', torch.zeros(()))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def forward(self, reward):
        moments = torch.stack((reward.sum(), reward.square().sum(), reward.new_tensor(reward.numel())))
        if distributed.is_initialized():
            distributed.all_reduce(moments)
        self.mean.lerp_(moments[0] / moments[2], .01)
        self.square.lerp_(moments[1] / moments[2], .01)
        self.count.add_(1)
        correction = 1 - .99 ** self.count
        variance = (self.square / correction - (self.mean / correction).square()).clamp_min(1e-8)
        return reward / variance.sqrt()


class TransitionReplay:
    """Uniform valid single-step transitions; time blocks are storage only."""
    def __init__(self, capacity, actors, horizon, device):
        self.capacity = max(1, capacity // (actors * horizon))
        self.device = device
        self.storage = None
        self.position = 0
        self.size = 0
        self.valid_indices = [None] * self.capacity

    def add(self, block):
        if self.storage is None:
            self.storage = {
                name: torch.empty((self.capacity,) + value.shape,
                                  dtype=torch.float16 if value.is_floating_point() and name not in ('reward', 'discount', 'reward_components') else value.dtype,
                                  device=self.device)
                for name, value in block.items()
            }
        for name, value in block.items():
            self.storage[name][self.position].copy_(value)
        self.valid_indices[self.position] = block['valid'].flatten().nonzero().flatten()
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, generator=None):
        lengths = torch.tensor([len(indices) for indices in self.valid_indices[:self.size]], device=self.device)
        offsets = lengths.cumsum(0)
        selected = torch.randint(int(offsets[-1]), (count,), device=self.device, generator=generator)
        blocks = torch.searchsorted(offsets, selected, right=True)
        starts = offsets - lengths
        local = selected - starts[blocks]
        rows = torch.empty_like(selected)
        for index in range(self.size):
            mask = blocks == index
            rows[mask] = self.valid_indices[index][local[mask]]
        return {name: value.flatten(1, 2)[blocks, rows].float() for name, value in self.storage.items()}

    def state_dict(self):
        return {'storage': self.storage, 'position': self.position, 'size': self.size, 'valid_indices': self.valid_indices}

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value)


class ExpertSequences:
    def __init__(self, states, valid):
        self.states = states
        self.valid = valid
        self.starts = {}

    def sample(self, count, length):
        if length not in self.starts:
            # A sequence contains only transitions from one physical episode.
            invalid = (~self.valid.bool()).long()
            prefix = torch.cat((torch.zeros_like(invalid[:1]), invalid.cumsum(0)))
            windows = prefix[length:] - prefix[:-length]
            self.starts[length] = (windows == 0).nonzero()
        starts = self.starts[length]
        if not len(starts):
            raise ValueError('Expert collection has no valid contiguous sequence of length ' + str(length))
        selected = starts[torch.randint(len(starts), (count,), device=starts.device)]
        times = selected[:, :1] + torch.arange(length, device=starts.device)
        return self.states[times, selected[:, 1:2]].float()
