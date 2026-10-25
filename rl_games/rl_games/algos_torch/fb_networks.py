"""Forward-backward models using the existing rl_games network building blocks."""

import math

import torch
from torch import nn

from rl_games.algos_torch.network_builder import NetworkBuilder


def project_task(task):
    return math.sqrt(task.shape[-1]) * torch.nn.functional.normalize(task, dim=-1)


def successor_loss(forward, backward, target_forward, target_backward, discount, valid):
    """Exact off-diagonal squared FB loss, without materializing an N x N matrix.

    Targets use ensemble mean. The identity is tested against dense values and
    gradients. Sampling weights exclude the simulator's post-reset boundary row.
    """
    weight = valid.float().reshape(-1, 1)
    count = weight.sum()
    denominator = (count * (count - 1)).clamp_min(1)
    backward = backward.float()
    target_backward = target_backward.float().detach()
    backward_gram = backward.T @ (weight * backward)
    target_gram = target_backward.T @ (weight * target_backward)
    cross_gram = backward.T @ (weight * target_backward)
    continuation = target_forward.float().mean(0).detach() * discount.reshape(-1, 1)
    continuation_gram = continuation.T @ (weight * continuation)
    loss = backward.new_zeros(())
    paired = backward.new_zeros(())
    for prediction in forward.float():
        prediction_gram = prediction.T @ (weight * prediction)
        prediction_cross = prediction.T @ (weight * continuation)
        squared_sum = (prediction_gram * backward_gram).sum()
        squared_sum = squared_sum + (continuation_gram * target_gram).sum()
        squared_sum = squared_sum - 2 * (prediction_cross * cross_gram).sum()
        diagonal = (prediction * backward).sum(-1)
        target_diagonal = (continuation * target_backward).sum(-1)
        diagonal_square = (weight[:, 0] * (diagonal - target_diagonal).square()).sum()
        loss = loss + 0.5 * (squared_sum - diagonal_square) / denominator
        paired = paired - (weight[:, 0] * diagonal).sum() / count.clamp_min(1)
    gram_diagonal = backward.square().sum(-1)
    ortho_square = backward_gram.square().sum()
    ortho_square = ortho_square - (weight[:, 0] * gram_diagonal.square()).sum()
    ortho = ortho_square / denominator
    ortho = ortho - 2 * (weight[:, 0] * gram_diagonal).sum() / count.clamp_min(1)
    return loss + paired, ortho


class DenseMap(NetworkBuilder.BaseNetwork):
    def __init__(self, input_size, output_size, units, activation="elu"):
        super().__init__()
        self.features = self._build_mlp(input_size, units, activation, nn.Linear)
        self.output = nn.Linear(units[-1], output_size)

    def forward(self, inputs):
        return self.output(self.features(inputs))


class RecurrentTaskActor(NetworkBuilder.BaseNetwork):
    """Memory encodes observations; task conditions the action head after memory.

    This placement makes stored memory independent of the relabeled command.
    It retains SimToolReal's LSTM-before-MLP structure and layer normalization.
    """
    def __init__(self, observation_size, action_size, latent_size, network):
        super().__init__()
        self.hidden_size = network["rnn"]["units"]
        self.memory = self._build_rnn("lstm", observation_size, self.hidden_size, 1)
        self.normalizer = nn.LayerNorm(self.hidden_size)
        self.policy = DenseMap(
            self.hidden_size + latent_size,
            action_size,
            network["mlp"]["units"],
            network["mlp"].get("activation", "elu"),
        )

    def initial_state(self, batch_size, device):
        shape = (1, batch_size, self.hidden_size)
        return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))

    def encode(self, observations, states, resets=None):
        if resets is not None and resets.ndim == 2:
            resets = resets.unsqueeze(-1)
        memory, states = self.memory(observations, states, resets)
        return self.normalizer(memory), states

    def act(self, memory, task):
        return self.action_logits(memory, task).tanh()

    def action_logits(self, memory, task):
        return self.policy(torch.cat((memory, task), dim=-1))


class FBModel(nn.Module):
    def __init__(self, observation_size, state_size, action_size, config, network):
        super().__init__()
        latent_size = config["latent_size"]
        self.actor = RecurrentTaskActor(observation_size, action_size, latent_size, network)
        self.backward = DenseMap(observation_size, latent_size, config["backward_units"])
        forward_size = state_size + action_size + latent_size
        self.forward_maps = nn.ModuleList([
            DenseMap(forward_size, latent_size, config["critic_units"])
            for _ in range(2)
        ])
        self.auxiliary_maps = nn.ModuleList([
            DenseMap(forward_size, 1, config["critic_units"])
            for _ in range(2)
        ])

    def forward(self, state, action, task):
        inputs = torch.cat((state, action, task), dim=-1)
        return torch.stack([network(inputs) for network in self.forward_maps])

    def auxiliary(self, state, action, task):
        inputs = torch.cat((state, action, task), dim=-1)
        return torch.stack([network(inputs).squeeze(-1) for network in self.auxiliary_maps])


class SequenceReplay:
    """Ring of contiguous vector-environment blocks, preserving recurrent context.

    Each block stores initial actor states, current/next observations separately,
    reset masks and actual rollout commands. Burn-in and loss suffix never cross
    an unrecorded gap, and episode boundaries reset memory inside a block.
    """
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.storage = None
        self.position = 0
        self.size = 0

    def add(self, block):
        if self.storage is None:
            self.storage = {}
            for key, value in block.items():
                dtype = value.dtype
                if value.is_floating_point() and key not in ("reward", "discount"):
                    dtype = torch.float16
                self.storage[key] = torch.empty(
                    (self.capacity,) + value.shape, dtype=dtype, device=self.device
                )
        for key, value in block.items():
            self.storage[key][self.position].copy_(value)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, sequence_count):
        block_indices = torch.randint(self.size, (sequence_count,), device=self.device)
        environment_count = self.storage["observation"].shape[2]
        environment_indices = torch.randint(environment_count, (sequence_count,), device=self.device)
        result = {}
        for key, value in self.storage.items():
            sampled = value[block_indices, :, environment_indices]
            result[key] = sampled.transpose(0, 1).float()
        return result

    def state_dict(self):
        return {"storage": self.storage, "position": self.position, "size": self.size}

    def load_state_dict(self, state):
        self.storage = state["storage"]
        self.position = state["position"]
        self.size = state["size"]
