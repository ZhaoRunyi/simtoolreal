"""Scalar FB Bellman identities and production update isolation."""
import copy
import ast
import os
import random
import subprocess
import types
from pathlib import Path

import numpy as np

import pytest
import torch

from rl_games.algos_torch.fb_networks import scalar_successor_loss
from test_fb_cpr import make_agent


def inputs():
    generator = torch.Generator().manual_seed(573)
    forward = torch.randn(2, 11, 4, generator=generator, requires_grad=True)
    backward = torch.randn(11, 4, generator=generator, requires_grad=True)
    target = torch.randn(2, 11, 4, generator=generator, requires_grad=True)
    command = torch.randn(11, 4, generator=generator, requires_grad=True)
    discount = torch.full((11, 1), .98)
    discount[2] = 0
    valid = torch.ones(11, dtype=torch.bool)
    valid[0] = False
    return forward, backward, target, command, discount, valid


def test_scalar_bellman_dense_values_gradients_and_detach():
    forward, backward, target, command, discount, valid = inputs()
    loss, metrics = scalar_successor_loss(forward, backward, target, command, discount, valid, True)
    with torch.no_grad():
        features = backward[valid].double()
        covariance = features.T @ features / len(features)
        reward = ((features @ covariance.inverse()) * command[valid].double()).sum(-1).float()
        next_q = (target[:, valid] * command[valid]).sum(-1).mean(0)
        expected_target = reward + discount[valid, 0] * next_q
    prediction = (forward[:, valid] * command.detach()[valid]).sum(-1)
    expected = .5 * (prediction - expected_target).square().mean(1).sum()
    torch.testing.assert_close(loss, expected)
    actual_gradient = torch.autograd.grad(loss, forward, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, forward)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual_gradient[:, 0].count_nonzero() == 0
    loss.backward()
    assert backward.grad is None and target.grad is None and command.grad is None
    assert metrics['q_td/solve_relative_residual'] < 1e-12


def test_masked_rows_and_terminal_do_not_contaminate_target():
    values = inputs()
    original, _ = scalar_successor_loss(*values)
    altered = [item.detach().clone() for item in values]
    altered[0][:, 0] = float('nan')
    altered[1][0] = float('nan')
    altered[2][:, 0] = float('nan')
    altered[3][0] = float('nan')
    altered[2][:, 2] += 1e6
    result, _ = scalar_successor_loss(*altered)
    torch.testing.assert_close(result, original)


def test_singular_coordinates_fail_without_fallback():
    values = list(inputs())
    values[1] = torch.zeros_like(values[1])
    with pytest.raises(RuntimeError):
        scalar_successor_loss(*values)


def fill_replay(agent):
    agent.replay.add({
        'observation': torch.randn(4, 3, 16), 'next_observation': torch.randn(4, 3, 16),
        'state': torch.randn(4, 3, 20), 'next_state': torch.randn(4, 3, 20),
        'history': torch.randn(4, 3, 64), 'next_history': torch.randn(4, 3, 64),
        'previous_action': torch.randn(4, 3, 3), 'action': torch.randn(4, 3, 3).tanh(),
        'command': torch.randn(4, 3, 4), 'reward': torch.randn(4, 3, 1),
        'auxiliary_reward': -torch.rand(4, 3, 1),
        'discount': torch.ones(4, 3, 1) * .98, 'valid': torch.ones(4, 3, 1, dtype=torch.bool),
    })


def equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_nested(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for one, two in zip(left, right):
            equal_nested(one, two)
    else:
        assert left == right


def test_zero_coefficient_keeps_complete_update_and_rng(tmp_path):
    from rl_games.algos_torch import fb_cpr_agent
    from rl_games.algos_torch.fb_cpr_networks import TemporalReplay
    torch.manual_seed(108)
    agent = make_agent(tmp_path / 'base')
    agent.model.local_prompt = True
    agent.options.update(logo_coefficient=.005, logo_horizon=3,
                         logo_local_weight=4 / 7, logo_goal_weight=3 / 7,
                         actor_gradient_probe_every=1)
    agent.replay = TemporalReplay(60, 3, 1, 'cpu')
    for _ in range(5):
        staging = make_agent(tmp_path / 'staging')
        fill_replay(staging)
        block = {name: value[0] for name, value in staging.replay.storage.items()}
        for step in range(4):
            agent.replay.add({name: value[step:step + 1] for name, value in block.items()})
    disabled = copy.deepcopy(agent)
    disabled.options['q_loss_coefficient'] = 0.
    # Execute the actual, pinned v4 production method, not a second new-code call.
    repository = Path(__file__).resolve().parents[2]
    source = subprocess.check_output([
        'git', 'show', '3bd73c66ae78c8ce5334bcf37bcc46cdb0e9eec2:rl_games/rl_games/algos_torch/fb_cpr_agent.py',
    ], cwd=repository, text=True)
    tree = ast.parse(source)
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FBcprAgent')
    update = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'update')
    namespace = dict(vars(fb_cpr_agent))
    exec(compile(ast.Module(body=[update], type_ignores=[]), '<pinned-v4-update>', 'exec'), namespace)
    agent.update = types.MethodType(namespace['update'], agent)
    rng = torch.get_rng_state()
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    metrics = agent.update()
    final_rng = torch.get_rng_state()
    final_numpy = np.random.get_state()
    final_python = random.getstate()
    torch.set_rng_state(rng)
    np.random.set_state(numpy_rng)
    random.setstate(python_rng)
    observed = disabled.update()
    assert torch.equal(final_rng, torch.get_rng_state())
    assert final_python == random.getstate()
    assert all(np.array_equal(left, right) for left, right in zip(final_numpy, np.random.get_state()))
    for name in ('model', 'target', 'actor_optimizer', 'representation_optimizer', 'auxiliary_optimizer',
                 'style_optimizer', 'discriminator_optimizer', 'observation_normalizer', 'state_normalizer',
                 'action_normalizer', 'reward_normalizer', 'replay'):
        equal_nested(getattr(agent, name).state_dict(), getattr(disabled, name).state_dict())
    equal_nested(metrics, observed)
    assert observed['logo/available'] == 1
    assert 'loss/logo_scaled' in observed
    equal_nested(agent.z_buffer[:agent.z_size], disabled.z_buffer[:disabled.z_size])
    equal_nested(agent.z_source_buffer, disabled.z_source_buffer)
    for name in ('optimizer_step', 'frame', 'epoch_num', 'z_size', 'z_position', 'task_command'):
        equal_nested(getattr(agent, name), getattr(disabled, name))


def test_enabled_production_update_finite_and_no_extra_rng(tmp_path):
    torch.manual_seed(109)
    agent = make_agent(tmp_path / 'base')
    fill_replay(agent)
    enabled = copy.deepcopy(agent)
    enabled.options['q_loss_coefficient'] = .1
    enabled.options['actor_gradient_probe_every'] = 1
    rng = torch.get_rng_state()
    agent.update()
    final_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    metrics = enabled.update()
    assert torch.equal(final_rng, torch.get_rng_state())
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert metrics['q_grad/weighted_q_norm'] > 0
    assert enabled.optimizer_step == 1
    assert not torch.equal(next(agent.model.forward_maps.parameters()), next(enabled.model.forward_maps.parameters()))
    # No additional B derivative, and the test has a non-binding clipping bound.
    equal_nested(agent.model.backward.state_dict(), enabled.model.backward.state_dict())
    equal_nested(agent.observation_normalizer.state_dict(), enabled.observation_normalizer.state_dict())
    equal_nested(agent.state_normalizer.state_dict(), enabled.state_normalizer.state_dict())


def test_combined_representation_gradient_is_sum():
    from rl_games.algos_torch.fb_networks import successor_loss
    forward, backward, target, command, discount, valid = inputs()
    target_backward = backward.detach().clone()
    fb, orthogonal = successor_loss(forward, backward, target, target_backward, discount, valid)
    q_loss, _ = scalar_successor_loss(forward, backward, target, command, discount, valid)
    original = fb + 50 * orthogonal
    original_grad = torch.autograd.grad(original, (forward, backward), retain_graph=True)
    q_grad = torch.autograd.grad(.1 * q_loss, forward, retain_graph=True)[0]
    actual = torch.autograd.grad(original + .1 * q_loss, (forward, backward))
    torch.testing.assert_close(actual[0], original_grad[0] + q_grad)
    torch.testing.assert_close(actual[1], original_grad[1])


@pytest.mark.parametrize('kind', ['insufficient', 'nonfinite'])
def test_invalid_scalar_inputs_fail(kind):
    values = list(inputs())
    if kind == 'insufficient':
        values[-1] = torch.zeros_like(values[-1])
    else:
        values[3] = torch.full_like(values[3], float('inf'))
    with pytest.raises((ValueError, FloatingPointError)):
        scalar_successor_loss(*values)
