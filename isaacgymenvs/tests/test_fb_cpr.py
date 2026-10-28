"""CPR update, replay and expert-sequence contracts without simulator dependencies."""

from types import SimpleNamespace

import gym
import torch

from rl_games.algos_torch.fb_cpr_agent import FBcprAgent
from rl_games.algos_torch.fb_cpr_networks import ExpertSequences, TransitionReplay


def test_transition_replay_excludes_reset_rows_and_keeps_pairs():
    replay = TransitionReplay(24, 3, 4, 'cpu')
    for offset in range(3):
        state = torch.arange(12).reshape(4, 3, 1).float() + 100 * offset
        valid = torch.ones_like(state, dtype=torch.bool)
        valid[0] = False
        replay.add({'state': state, 'next_state': state + 1, 'valid': valid})
    sample = replay.sample(1000)
    assert sample['valid'].all()
    assert sample['state'].min() >= 103
    torch.testing.assert_close(sample['next_state'], sample['state'] + 1)


def test_expert_slices_never_cross_episode_or_environment():
    state = torch.arange(30)[:, None, None].expand(30, 3, 1).float()
    state = state + 100 * torch.arange(3)[None, :, None]
    valid = torch.ones(30, 3, dtype=torch.bool)
    valid[15] = False
    experts = ExpertSequences(state, valid)
    sample = experts.sample(1000, 8)
    torch.testing.assert_close(sample[:, 1:] - sample[:, :-1], torch.ones(1000, 7, 1))
    assert not ((sample % 100) == 15).any()


def make_agent(tmp_path, **overrides):
    metadata = SimpleNamespace(obs_list=['keypoints_rel_goal', 'joint_pos'],
                               obs_type_size_dict={'keypoints_rel_goal': 12, 'joint_pos': 4})
    vector = SimpleNamespace(env=metadata, get_env_info=lambda: {
        'observation_space': gym.spaces.Box(-10, 10, (16,)),
        'state_space': gym.spaces.Box(-10, 10, (20,)),
        'action_space': gym.spaces.Box(-1, 1, (3,)),
    })
    options = dict(latent_size=4, residual_width=16, residual_layers=2, history_steps=4,
                   backward_learning_rate=1e-5, discriminator_learning_rate=1e-5,
                   discriminator_penalty=10., replay_blocks=2, replay_capacity=48, z_buffer_size=32,
                   expert_sequence_length=2, exploration_std=.05, target_noise_clip=.3,
                   orthogonal_coefficient=100., relabel_probability=.8,
                   target_tau=.01, critic_target_tau=.005, auxiliary_coefficient=.02,
                   style_coefficient=.05, task_reward_temperature=.05, cpu_threads=2)
    options.update(overrides)
    agent = FBcprAgent('test', {
        'config': {'fb': options, 'device': 'cpu', 'num_actors': 3, 'vec_env': vector,
                   'seq_length': 4, 'horizon_length': 4, 'minibatch_size': 8,
                   'learning_rate': 3e-4, 'gamma': .98, 'grad_norm': 1e30,
                   'train_dir': str(tmp_path), 'name': 'test'},
        'network': {},
    })
    agent.experts = ExpertSequences(torch.randn(30, 3, 20), torch.ones(30, 3, dtype=torch.bool))
    return agent


def test_production_cpr_update_and_checkpoint_restore(tmp_path):
    torch.manual_seed(12)
    agent = make_agent(tmp_path)
    agent.replay.add({
        'observation': torch.randn(4, 3, 16), 'next_observation': torch.randn(4, 3, 16),
        'state': torch.randn(4, 3, 20), 'next_state': torch.randn(4, 3, 20),
        'history': torch.randn(4, 3, 64), 'next_history': torch.randn(4, 3, 64),
        'previous_action': torch.randn(4, 3, 3), 'action': torch.randn(4, 3, 3).tanh(),
        'command': torch.randn(4, 3, 4), 'reward': torch.randn(4, 3, 1), 'auxiliary_reward': -torch.rand(4, 3, 1),
        'discount': torch.ones(4, 3, 1) * .98, 'valid': torch.ones(4, 3, 1, dtype=torch.bool),
    })
    before = {name: value.clone() for name, value in agent.model.named_parameters()}
    metrics = agent.update()
    assert all(torch.isfinite(value) for value in metrics.values())
    for prefix in ('forward_maps', 'backward', 'actor', 'style_maps', 'auxiliary_maps', 'discriminator'):
        assert any(not torch.equal(value, before[name]) for name, value in agent.model.named_parameters() if name.startswith(prefix))
    assert all(parameter.grad is None for parameter in agent.target.parameters())
    assert agent.z_size == 8
    assert agent.optimizer_step == 1
    assert agent._sample_commands(20).shape == (20, 4)
    assert metrics['replay/valid_fraction'] == 1
    agent.run_started = __import__('time').monotonic()
    path = tmp_path / 'checkpoint.pt'
    agent.save(path)
    restored = make_agent(tmp_path / 'restored')
    restored.restore(path)
    assert restored.optimizer_step == 1
    torch.testing.assert_close(restored.task_command, agent.task_command)
    torch.testing.assert_close(restored.reward_normalizer.square, agent.reward_normalizer.square)
    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, agent.model.state_dict()[name])


def test_reference_reset_updates_lifting_height_reference():
    import ast
    from pathlib import Path
    source = Path(__file__).parents[1] / 'tasks/simtoolreal/env.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_apply_reference_reset')
    namespace = {'torch': torch}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    env = SimpleNamespace(
        device='cpu', cfg={'env': {'referenceResetProbability': 1., 'tableObjectZOffset': .1}},
        arm_hand_dof_pos=torch.zeros(2, 3), arm_hand_dof_vel=torch.zeros(2, 3),
        prev_targets=torch.zeros(2, 3), cur_targets=torch.zeros(2, 3),
        root_state_tensor=torch.zeros(4, 13), object_indices=torch.tensor([0, 2]),
        table_indices=torch.tensor([1, 3]), lifted_object=torch.zeros(2, dtype=torch.bool),
        table_init_state=torch.zeros(2, 13), object_init_state=torch.zeros(2, 13),
        deferred_set_actor_root_state_tensor_indexed=lambda indices: None,
    )
    table = torch.zeros(1, 2, 13)
    table[:, :, 2] = torch.tensor([.35, .37])
    env.reference_reset_bank = dict(valid=torch.ones(1, 2, dtype=torch.bool),
        joint_position=torch.ones(1, 2, 3), joint_velocity=torch.zeros(1, 2, 3),
        target=torch.ones(1, 2, 3) * 2, object=torch.ones(1, 2, 13), table=table,
        lifted=torch.ones(1, 2, dtype=torch.bool))
    namespace['_apply_reference_reset'](env, torch.tensor([1]))
    torch.testing.assert_close(env.object_init_state[1, 2], torch.tensor(.47))
    torch.testing.assert_close(env.table_init_state[1, 2], env.root_state_tensor[3, 2])
    assert env.object_init_state[0, 2] == 0
    torch.testing.assert_close(env.prev_targets[1], torch.ones(3) * 2)
    env.record_reference_reset_statistics = True
    env.reference_reset_bank['valid'][0, 0] = False
    env.reference_reset_bank['object'][0, 1, 2] = .48
    namespace['_apply_reference_reset'](env, torch.tensor([0, 1]))
    statistics = env.reference_reset_statistics
    assert statistics['attempts'] == 2 and statistics['restored'] == 1
    assert statistics['fallback'] == 1 and statistics['unavailable'] == 1
    assert statistics['lifted_history'] == 1
    assert statistics['physically_at_or_below_5cm'] == 1
    assert statistics['physically_above_15cm'] == 0
    torch.testing.assert_close(statistics['physical_height_sum'], torch.tensor(.01))
    torch.testing.assert_close(statistics['object_linear_speed_sum'], torch.tensor(3.).sqrt())
    assert env.arm_hand_dof_pos[0].eq(0).all()


def test_actor_gradient_monitor_preserves_update_and_rng(tmp_path):
    import copy
    torch.manual_seed(127)
    agent = make_agent(tmp_path / 'plain')
    agent.replay.add({
        'observation': torch.randn(4, 3, 16), 'next_observation': torch.randn(4, 3, 16),
        'state': torch.randn(4, 3, 20), 'next_state': torch.randn(4, 3, 20),
        'history': torch.randn(4, 3, 64), 'next_history': torch.randn(4, 3, 64),
        'previous_action': torch.randn(4, 3, 3), 'action': torch.randn(4, 3, 3).tanh(),
        'command': torch.randn(4, 3, 4), 'reward': torch.randn(4, 3, 1),
        'auxiliary_reward': -torch.rand(4, 3, 1),
        'discount': torch.ones(4, 3, 1) * .98, 'valid': torch.ones(4, 3, 1, dtype=torch.bool),
    })
    monitored = copy.deepcopy(agent)
    monitored.options['actor_gradient_probe_every'] = 1
    initial_rng = torch.get_rng_state()
    baseline_metrics = agent.update()
    final_rng = torch.get_rng_state()
    torch.set_rng_state(initial_rng)
    metrics = monitored.update()
    assert torch.equal(final_rng, torch.get_rng_state())
    for name, parameter in agent.model.state_dict().items():
        torch.testing.assert_close(parameter, monitored.model.state_dict()[name], rtol=0, atol=0)
    for name, value in baseline_metrics.items():
        torch.testing.assert_close(value, metrics[name], rtol=0, atol=0)
    for name in ['fb_norm', 'style_norm', 'auxiliary_norm', 'fb_style_cosine',
                 'fb_auxiliary_cosine', 'fb_total_cosine', 'fb_update_cosine', 'update_norm']:
        assert torch.isfinite(metrics['actor_grad/' + name])
    assert metrics['actor_grad/update_norm'] > 0


def test_actor_gradient_probe_survives_sparse_log_interval(tmp_path, monkeypatch):
    from rl_games.algos_torch.fb_agent import FBAgent
    agent = make_agent(tmp_path)
    expected = {'actor_grad/probe_optimizer_step': torch.tensor(1025.),
                'actor_grad/fb_norm': torch.tensor(3.)}
    agent.pending_actor_gradient_metrics = expected.copy()
    captured = []
    monkeypatch.setattr(FBAgent, '_log', lambda self, metrics: captured.append(metrics) or metrics)
    first = agent._log({'optimizer_step': torch.tensor(1040.)})
    assert first['actor_grad/probe_optimizer_step'] == 1025
    assert first['optimizer_step'] == 1040
    agent._log({'optimizer_step': torch.tensor(1200.)})
    assert 'actor_grad/fb_norm' not in captured[1]


def test_auxiliary_soft_limits_preserve_original_reward_and_have_correct_units(tmp_path):
    agent = make_agent(tmp_path)
    agent.options['auxiliary_reward_terms'] = ['velocity']
    components = {'velocity': torch.tensor([-2., -3., -4.])}
    environment = SimpleNamespace(
        arm_hand_dof_lower_limits=torch.tensor([-1., -2.]),
        arm_hand_dof_upper_limits=torch.tensor([1., 2.]),
        arm_hand_dof_pos=torch.tensor([[0., 0.], [.95, -1.9], [1., -2.]]),
    )
    legacy, penalty = agent._auxiliary_reward(environment, components)
    torch.testing.assert_close(legacy, components['velocity'])
    assert not penalty.any()
    agent.options.update(joint_limit_coefficient=10., soft_joint_limit_fraction=.95)
    reward, penalty = agent._auxiliary_reward(environment, components)
    torch.testing.assert_close(penalty, torch.tensor([0., 0., -1.5]))
    torch.testing.assert_close(reward, torch.tensor([-2., -3., -5.5]))
    torch.testing.assert_close(components['velocity'], torch.tensor([-2., -3., -4.]))


def test_reset_only_update_has_no_reference_or_style_path(tmp_path, monkeypatch):
    from rl_games.algos_torch.fb_cpr_networks import TemporalReplay
    torch.manual_seed(29)
    agent = make_agent(tmp_path, reference_reset_only=True, style_coefficient=0.,
                       tracking_fraction=0., actor_gradient_probe_every=1,
                       logo_coefficient=.005, logo_horizon=8,
                       logo_local_weight=4 / 7, logo_goal_weight=3 / 7)
    agent.model.local_prompt = True
    agent.experts = None
    agent.replay = TemporalReplay(60, 3, 1, 'cpu')
    for step in range(16):
        state = torch.randn(1, 3, 20)
        agent.replay.add({
            'observation': torch.randn(1, 3, 16), 'next_observation': torch.randn(1, 3, 16),
            'state': state, 'next_state': state + .1,
            'history': torch.randn(1, 3, 64), 'next_history': torch.randn(1, 3, 64),
            'previous_action': torch.randn(1, 3, 3), 'action': torch.randn(1, 3, 3).tanh(),
            'command': torch.randn(1, 3, 4), 'reward': torch.randn(1, 3, 1),
            'auxiliary_reward': -torch.rand(1, 3, 1),
            'discount': torch.ones(1, 3, 1) * .98, 'valid': torch.ones(1, 3, 1, dtype=torch.bool),
        })
    def forbidden(*args, **kwargs):
        raise AssertionError('Reset-only training called a disabled expert/style path')
    for target, name in [(agent, '_discriminator_update'), (agent.model, 'discriminate'),
                         (agent.model, 'style'), (agent.target, 'style'),
                         (agent.style_optimizer, 'step'), (agent.discriminator_optimizer, 'step')]:
        monkeypatch.setattr(target, name, forbidden)
    before = {name: value.clone() for name, value in agent.model.state_dict().items()}
    target_before = {name: value.clone() for name, value in agent.target.state_dict().items()}
    agent.config['minibatch_size'] = 1024
    agent.z_buffer = torch.empty(2048, 4)
    agent.z_source_buffer = torch.zeros(2048, dtype=torch.long)
    metrics = agent.update()
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics['logo/available'] == 1
    assert metrics['actor_grad/style_norm'] == 0
    assert metrics['value/style'] == 0
    assert metrics['supervision/style_enabled'] == 0
    assert 'loss/style' not in metrics and 'loss/discriminator' not in metrics
    assert not agent.style_optimizer.state and not agent.discriminator_optimizer.state
    for name, value in agent.model.state_dict().items():
        if name.startswith(('style_maps', 'discriminator')):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            torch.testing.assert_close(agent.target.state_dict()[name], target_before[name], rtol=0, atol=0)
    for prefix in ('forward_maps', 'backward', 'actor', 'auxiliary_maps'):
        assert any(not torch.equal(value, before[name]) for name, value in agent.model.state_dict().items()
                   if name.startswith(prefix))
    assert all(not parameter.requires_grad for parameter in agent.model.style_maps.parameters())
    assert all(not parameter.requires_grad for parameter in agent.model.discriminator.parameters())
    sources = agent.z_source_buffer[:agent.z_size]
    assert set(sources.tolist()) == {0, 1}
    assert .4 < sources.float().mean() < .6
    assert .7 < metrics['task/relabel_fraction'] < .9
    torch.testing.assert_close(agent.z_buffer[:agent.z_size].norm(dim=-1), torch.full((1024,), 2.))


def test_reset_only_cache_loading_and_rollout_refresh(tmp_path):
    agent = make_agent(tmp_path, reference_reset_only=True, style_coefficient=0., tracking_fraction=0., command_steps=100)
    agent.experts = None
    agent.global_rank = 0
    agent.output_dir.mkdir(parents=True, exist_ok=True)
    # No states/valid keys: the reset-only loader must not depend on reference trajectories.
    bank = {'valid': torch.ones(2, 3, dtype=torch.bool), 'joint_position': torch.randn(2, 3, 4)}
    torch.save({'resets': bank}, agent.output_dir / 'expert.rank0.pt')
    agent._prepare_experts()
    assert agent.experts is None
    for name, value in bank.items():
        torch.testing.assert_close(agent.vec_env.env.reference_reset_bank[name], value, rtol=0, atol=0)
    agent.rollout_command = torch.zeros(3, 4)
    agent.rollout_command_source = torch.full((3,), 7, dtype=torch.long)
    agent.reset_before = torch.tensor([True, False, False])
    agent.command_age = torch.tensor([1, 99, 100])
    agent._rollout_prompts()
    assert agent.rollout_command[1].eq(0).all()
    torch.testing.assert_close(agent.rollout_command[[0, 2]].norm(dim=-1), torch.full((2,), 2.))
    assert agent.rollout_command_source.tolist() == [0, 7, 0]
    assert agent.command_age.tolist() == [0, 99, 0]


def test_reset_only_rejects_incompatible_resume_and_configuration(tmp_path):
    import pytest
    with pytest.raises(ValueError, match='zero style'):
        make_agent(tmp_path / 'bad', reference_reset_only=True, tracking_fraction=0.)
    agent = make_agent(tmp_path / 'valid', reference_reset_only=True, style_coefficient=0., tracking_fraction=0.)
    with pytest.raises(ValueError, match='different reference-use'):
        agent._restore_extra({})
    restored = make_agent(tmp_path / 'restored', reference_reset_only=True, style_coefficient=0., tracking_fraction=0.)
    agent._cache_commands(torch.ones(3, 4), torch.tensor([0, 1, 0]))
    restored._restore_extra(agent._checkpoint_extra())
    torch.testing.assert_close(restored.z_buffer, agent.z_buffer)
    torch.testing.assert_close(restored.z_source_buffer, agent.z_source_buffer)
    assert restored.z_size == 3


def test_reset_only_production_collect_and_rank_log(tmp_path, monkeypatch):
    import json
    from rl_games.algos_torch.fb_agent import FBAgent
    agent = make_agent(tmp_path, reference_reset_only=True, style_coefficient=0.,
                       tracking_fraction=0., command_steps=100,
                       auxiliary_reward_terms=['action'], log_command_sources=True,
                       updates_per_iteration=16)
    environment = agent.vec_env.env
    environment.reset_buf = torch.zeros(3, dtype=torch.long)
    environment.prev_episode_successes = torch.zeros(3)
    environment.prev_episode_closest_keypoint_max_dist = torch.zeros(3)
    environment.success_tolerance = .05
    environment.lifted_object = torch.zeros(3)
    environment.reference_reset_statistics = {'attempts': torch.tensor(9.), 'restored': torch.tensor(7.)}
    observations = {'obs': torch.randn(3, 16), 'states': torch.randn(3, 20)}
    agent.vec_env.reset = lambda: observations
    agent.vec_env.step = lambda action: (
        observations, torch.ones(3), torch.tensor([False, True, False]),
        {'episode_cumulative': {'action': -action.square().sum(-1)}},
    )
    def forbidden(*args, **kwargs):
        raise AssertionError('Collection read the reference sequence or style network')
    monkeypatch.setattr(agent.experts, 'sample', forbidden)
    monkeypatch.setattr(agent.model, 'style', forbidden)
    monkeypatch.setattr(agent.model, 'discriminate', forbidden)
    agent._cache_commands(torch.ones(2, 4), torch.tensor([0, 1]))
    normalizers_before = {name: {key: value.clone() for key, value in getattr(agent, name).state_dict().items()}
                          for name in ('observation_normalizer', 'state_normalizer', 'action_normalizer')}
    agent._initialize_rollout()
    metrics = agent.collect()
    assert agent.frame == 12
    assert agent.replay.size == 1
    assert metrics['command_count/expert_mean'] == 0
    assert metrics['command_count/expert_sequence'] == 0
    assert metrics['reference_reset/attempts'] == 9
    for name, before in normalizers_before.items():
        for key, value in getattr(agent, name).state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    monkeypatch.setattr(FBAgent, '_log', lambda self, metrics: metrics)
    agent.global_rank = 2
    agent._log(metrics)
    record = json.loads((agent.output_dir / 'reference_reset.rank2.jsonl').read_text())
    assert record['rank'] == 2 and record['attempts'] == 9 and record['restored'] == 7
    assert record['environment_steps'] == 12
    assert record['counters_since_process_start']


def test_reset_only_full_checkpoint_restore_preserves_frozen_branches(tmp_path):
    import time
    import pytest
    agent = make_agent(tmp_path / 'source', reference_reset_only=True, style_coefficient=0., tracking_fraction=0.)
    agent._cache_commands(torch.ones(3, 4), torch.tensor([0, 1, 0]))
    agent.run_started = time.monotonic()
    checkpoint = tmp_path / 'reset_only.pt'
    agent.save(checkpoint)
    restored = make_agent(tmp_path / 'restored', reference_reset_only=True, style_coefficient=0., tracking_fraction=0.)
    restored.restore(checkpoint)
    assert restored.z_size == 3
    assert not restored.style_optimizer.state and not restored.discriminator_optimizer.state
    for name in ('style_maps', 'discriminator'):
        assert all(not parameter.requires_grad for parameter in getattr(restored.model, name).parameters())
        for actual, expected in zip(getattr(restored.model, name).parameters(), getattr(agent.model, name).parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    old = make_agent(tmp_path / 'old')
    old.run_started = time.monotonic()
    old.save(tmp_path / 'old.pt')
    with pytest.raises(ValueError, match='different reference-use'):
        restored.restore(tmp_path / 'old.pt')
