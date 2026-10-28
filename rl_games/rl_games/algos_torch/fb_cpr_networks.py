# BFM-Zero / Meta FB-CPR architecture and loss adaptation.
# Reference components: CC BY-NC 4.0; see docs/licenses/bfm-zero-LICENSE.
# Sphere-log / LOGO reference: Roboparty/I-BFM-Code b4547263 (CC BY-NC 4.0).
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


def sphere_log_prompt(base, target, epsilon=1e-7):
    """I-BFM sphere logarithm, including a deterministic antipodal direction."""
    current = functional.normalize(base, dim=-1, eps=epsilon)
    goal = functional.normalize(target, dim=-1, eps=epsilon)
    cosine = (current * goal).sum(-1, keepdim=True).clamp(-1, 1)
    tangent = goal - cosine * current
    sine = tangent.norm(dim=-1, keepdim=True)
    angle = torch.atan2(sine, cosine)
    basis = torch.zeros_like(current).scatter_(-1, current.abs().argmin(-1, keepdim=True), 1)
    antipode = functional.normalize(basis - (basis * current).sum(-1, keepdim=True) * current,
                                   dim=-1, eps=epsilon)
    direction = torch.where((sine <= epsilon) & (cosine < 0), antipode, tangent / sine.clamp_min(epsilon))
    return math.sqrt(base.shape[-1]) * angle * direction


class BackwardMap(nn.Module):
    def __init__(self, state_size, latent_size, state_indices=None):
        super().__init__()
        self.state_size = state_size
        indices = list(range(state_size)) if state_indices is None else list(state_indices)
        if len(indices) != state_size or len(set(indices)) != state_size:
            raise ValueError('Backward input indices must select each field exactly once')
        self.register_buffer('state_indices', torch.tensor(indices, dtype=torch.long), persistent=False)
        self.layers = nn.Sequential(
            nn.Linear(state_size, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, latent_size),
        )

    def forward(self, state):
        return project_task(self.layers(self.select_state(state)))

    def select_state(self, state):
        return state.index_select(-1, self.state_indices)


class CPRModel(nn.Module):
    def __init__(self, observation_size, state_size, action_size, config, network):
        super().__init__()
        latent_size = config['latent_size']
        self.actor = HistoryActor(observation_size, action_size, latent_size, config)
        self.backward_state_size = config.get('backward_state_size', state_size)
        self.backward = BackwardMap(self.backward_state_size, latent_size, config.get('backward_state_indices'))
        self.local_prompt = config.get('local_prompt', False)
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
        return self.discriminator(torch.cat((command, self.backward.select_state(state)), dim=-1)).squeeze(-1)

    @torch.no_grad()
    def actor_prompt(self, state, command):
        if self.local_prompt:
            return sphere_log_prompt(self.backward(state), project_task(command))
        return command


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
                                  dtype=torch.float16 if value.is_floating_point() and name not in ('reward', 'discount', 'reward_components', 'actor_diagnostic') else value.dtype,
                                  device=self.device)
                for name, value in block.items()
            }
        # Optional observability fields may appear after restoring an older ring.
        # Unknown historical rows stay unknown; insertion and sampling use no RNG.
        defaults = {'command_source': -1, 'actor_diagnostic': 0, 'actor_diagnostic_valid': False}
        for name, default in defaults.items():
            if name in block and name not in self.storage:
                self.storage[name] = torch.full((self.capacity,) + block[name].shape, default,
                                                dtype=block[name].dtype, device=self.device)
            if name in self.storage and name not in block:
                self.storage[name][self.position].fill_(default)
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


class TemporalReplay(TransitionReplay):
    """CPU single-step ring with vectorized valid sampling and episode windows.

    Payload keeps the production transition schema. Time ids prevent windows
    crossing the newest ring boundary; discount/valid prevent reset crossings.
    """
    def __init__(self, capacity, actors, horizon, output_device):
        if horizon != 1:
            raise ValueError('Temporal replay requires one vector step per insertion')
        super().__init__(capacity, actors, horizon, 'cpu')
        self.output_device = output_device
        self.row_indices = torch.zeros(self.capacity, actors, dtype=torch.long)
        self.lengths = torch.zeros(self.capacity, dtype=torch.long)
        self.time_ids = torch.full((self.capacity,), -1, dtype=torch.long)
        self.next_time = 0

    def add(self, block):
        block = {name: value.detach().cpu() for name, value in block.items()}
        position = self.position
        super().add(block)
        rows = self.valid_indices[position]
        self.row_indices[position, :len(rows)] = rows
        self.lengths[position] = len(rows)
        self.time_ids[position] = self.next_time
        self.next_time += 1

    def _indices(self, count, generator=None):
        offsets = self.lengths[:self.size].cumsum(0)
        if not self.size or offsets[-1] == 0:
            raise ValueError('No valid transitions in replay')
        selected = torch.randint(int(offsets[-1]), (count,), generator=generator)
        blocks = torch.searchsorted(offsets, selected, right=True)
        rows = self.row_indices[blocks, selected - (offsets - self.lengths[:self.size])[blocks]]
        return blocks, rows

    def _gather(self, blocks, rows):
        return {name: value[:, 0][blocks, rows].float().to(self.output_device)
                for name, value in self.storage.items()}

    def sample(self, count, generator=None):
        return self._gather(*self._indices(count, generator))

    def sample_window(self, count, horizon):
        if self.size < horizon:
            return None
        chosen_blocks, chosen_rows = [], []
        remaining = count
        offsets = torch.arange(horizon)
        for _ in range(32):
            blocks, rows = self._indices(max(remaining * 2, 64))
            future = (blocks[:, None] + offsets) % self.capacity
            valid = (self.time_ids[future] == self.time_ids[blocks, None] + offsets).all(-1)
            valid &= self.storage['valid'][future, 0, rows[:, None], 0].all(-1)
            valid &= (self.storage['discount'][future[:, :-1], 0, rows[:, None], 0] > 0).all(-1)
            blocks, rows = blocks[valid][:remaining], rows[valid][:remaining]
            chosen_blocks.append(blocks)
            chosen_rows.append(rows)
            remaining -= len(blocks)
            if remaining == 0:
                blocks, rows = torch.cat(chosen_blocks), torch.cat(chosen_rows)
                batch = self._gather(blocks, rows)
                batch['local_state'] = self.storage['next_state'][blocks, 0, rows].float().to(self.output_device)
                batch['goal_state'] = self.storage['next_state'][(blocks + horizon - 1) % self.capacity, 0, rows].float().to(self.output_device)
                return batch
        return None

    def state_dict(self):
        return {**super().state_dict(), 'time_ids': self.time_ids, 'next_time': self.next_time}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.capacity = len(self.valid_indices)
        actors = self.storage['valid'].shape[2]
        self.row_indices = torch.zeros(self.capacity, actors, dtype=torch.long)
        self.lengths = torch.zeros(self.capacity, dtype=torch.long)
        for index, rows in enumerate(self.valid_indices[:self.size]):
            self.row_indices[index, :len(rows)] = rows
            self.lengths[index] = len(rows)


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
