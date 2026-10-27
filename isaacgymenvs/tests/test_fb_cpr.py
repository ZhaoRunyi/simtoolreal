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


def make_agent(tmp_path):
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
