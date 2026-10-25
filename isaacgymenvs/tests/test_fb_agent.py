"""Tests of FB identities, recurrent relabeling and replay/reset alignment."""

import torch
import pytest

from rl_games.algos_torch.fb_networks import RecurrentTaskActor, SequenceReplay, successor_loss


def test_factorized_loss_and_gradients_match_dense():
    torch.manual_seed(5)
    count, dimension = 13, 5
    forward = torch.randn(2, count, dimension, requires_grad=True)
    backward = torch.randn(count, dimension, requires_grad=True)
    target_forward = torch.randn_like(forward)
    target_backward = torch.randn_like(backward)
    discount = .99 * torch.randint(0, 2, (count, 1)).float()
    valid = torch.ones(count)
    valid[2] = 0
    fast_fb, fast_ortho = successor_loss(forward, backward, target_forward, target_backward, discount, valid)
    prediction = forward @ backward.T
    target = (target_forward.mean(0) @ target_backward.T) * discount
    off_diagonal = valid[:, None].bool() & valid[None, :].bool() & ~torch.eye(count).bool()
    diagonal = valid.bool()
    dense_fb = .5 * sum((head - target)[off_diagonal].square().mean() for head in prediction)
    dense_fb = dense_fb - sum(head.diag()[diagonal].mean() for head in prediction)
    gram = backward @ backward.T
    dense_ortho = gram[off_diagonal].square().mean() - 2 * gram.diag()[diagonal].mean()
    torch.testing.assert_close(fast_fb, dense_fb)
    torch.testing.assert_close(fast_ortho, dense_ortho)
    fast_gradients = torch.autograd.grad(fast_fb + 3 * fast_ortho, (forward, backward), retain_graph=True)
    dense_gradients = torch.autograd.grad(dense_fb + 3 * dense_ortho, (forward, backward))
    for actual, expected in zip(fast_gradients, dense_gradients):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_task_relabeling_keeps_observation_memory_and_reset(batch_size):
    network = {"rnn": {"units": 12}, "mlp": {"units": [16, 8], "activation": "elu"}}
    actor = RecurrentTaskActor(5, 3, 4, network)
    observation = torch.randn(7, batch_size, 5)
    reset = torch.zeros(7, batch_size)
    reset[3] = 1
    initial = actor.initial_state(batch_size, "cpu")
    memory, _ = actor.encode(observation, initial, reset)
    after_reset, _ = actor.encode(observation[3:], initial)
    torch.testing.assert_close(memory[3:], after_reset)
    first_task = torch.randn(7, batch_size, 4)
    second_task = torch.randn(7, batch_size, 4)
    first_action = actor.act(memory, first_task)
    second_action = actor.act(memory, second_task)
    assert not torch.allclose(first_action, second_action)
    one_step_states = initial
    outputs = []
    for index in range(len(observation)):
        output, one_step_states = actor.encode(observation[index:index + 1], one_step_states, reset[index:index + 1])
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs), memory)


def test_replay_keeps_contiguous_environment_sequences_and_hidden_states():
    replay = SequenceReplay(2, "cpu")
    environment_count = 3
    for block_index in range(3):
        observation = torch.arange(8)[:, None, None] + 10 * torch.arange(environment_count)[None, :, None]
        observation = observation + 100 * block_index
        replay.add({
            "observation": observation.float(),
            "next_observation": observation.float() + 1,
            "hidden": torch.ones(1, environment_count, 4) * block_index,
            "reset": torch.zeros(8, environment_count, 1, dtype=torch.bool),
        })
    sampled = replay.sample(20)
    assert sampled["observation"].shape == (8, 20, 1)
    assert sampled["hidden"].shape == (1, 20, 4)
    torch.testing.assert_close(sampled["observation"][1:] - sampled["observation"][:-1], torch.ones(7, 20, 1))
    torch.testing.assert_close(sampled["next_observation"], sampled["observation"] + 1)
    assert sampled["hidden"].min() >= 1


@pytest.mark.parametrize("auxiliary_coefficient", [0.0, 1.0])
def test_production_learner_updates_all_enabled_networks(tmp_path, auxiliary_coefficient):
    from types import SimpleNamespace
    import gym
    from rl_games.algos_torch.fb_agent import FBAgent

    metadata = SimpleNamespace(obs_list=["keypoints_rel_goal", "joint_pos"], obs_type_size_dict={"keypoints_rel_goal": 12, "joint_pos": 4})
    vector = SimpleNamespace(env=metadata, get_env_info=lambda: {
        "observation_space": gym.spaces.Box(-10, 10, (16,)),
        "state_space": gym.spaces.Box(-10, 10, (20,)),
        "action_space": gym.spaces.Box(-1, 1, (3,)),
    })
    options = {
        "latent_size": 4, "critic_units": [16, 8], "backward_units": [8, 8],
        "backward_learning_rate": 1e-4, "replay_blocks": 2,
        "goal_mix_probability": .5, "task_mix_probability": .2,
        "relabel_probability": .8, "exploration_std": .2,
        "target_noise_clip": .3, "orthogonal_coefficient": 1.,
        "auxiliary_coefficient": auxiliary_coefficient, "reward_scale": .01,
        "task_reward_temperature": .05, "target_tau": .01,
    }
    agent = FBAgent("test", {
        "config": {"fb": options, "device": "cpu", "num_actors": 3, "vec_env": vector,
                   "seq_length": 2, "horizon_length": 2, "minibatch_size": 6,
                   "learning_rate": 1e-3, "gamma": .99, "train_dir": str(tmp_path), "name": "test"},
        "network": {"rnn": {"units": 8}, "mlp": {"units": [16, 8]}},
    })
    agent.replay.add({
        "observation": torch.randn(4, 3, 16), "next_observation": torch.randn(4, 3, 16),
        "state": torch.randn(4, 3, 20), "next_state": torch.randn(4, 3, 20),
        "action": torch.randn(4, 3, 3).tanh(), "command": torch.randn(4, 3, 4),
        "reward": torch.randn(4, 3, 1), "discount": torch.ones(4, 3, 1) * .99,
        "reset": torch.zeros(4, 3, 1, dtype=torch.bool),
        "valid": torch.ones(4, 3, 1, dtype=torch.bool),
        "hidden": torch.zeros(1, 3, 8), "cell": torch.zeros(1, 3, 8),
    })
    before = {name: parameter.detach().clone() for name, parameter in agent.model.named_parameters()}
    metrics = agent.update()
    assert all(torch.isfinite(value) for value in metrics.values())
    changes = {name: not torch.equal(parameter, before[name]) for name, parameter in agent.model.named_parameters()}
    assert any(changed for name, changed in changes.items() if name.startswith("actor."))
    assert any(changed for name, changed in changes.items() if name.startswith("forward_maps."))
    assert any(changed for name, changed in changes.items() if name.startswith("backward."))
    assert any(changed for name, changed in changes.items() if name.startswith("auxiliary_maps.")) == bool(auxiliary_coefficient)
    assert all(parameter.grad is None for parameter in agent.target.parameters())


def test_original_bounds_penalty_can_recover_saturated_action_logits():
    from rl_games.algos_torch.a2c_continuous import A2CAgent
    from types import SimpleNamespace

    logits = torch.tensor([[-20., 0., 20.]], requires_grad=True)
    baseline = A2CAgent.bound_loss(SimpleNamespace(bounds_loss_coef=.0001), logits)
    penalty = (logits.abs() - 1.1).clamp_min(0).square().sum(-1)
    torch.testing.assert_close(penalty, baseline)
    gradient = torch.autograd.grad(penalty.sum(), logits)[0]
    assert gradient[0, 0] < 0 and gradient[0, 2] > 0
    assert gradient[0, 1] == 0
