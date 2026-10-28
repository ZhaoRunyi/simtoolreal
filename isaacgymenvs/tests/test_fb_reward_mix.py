"""Reward command interfaces tested through the real learner and replay."""
import copy
from pathlib import Path

import pytest
import torch

from test_fb_cpr import make_agent
from rl_games.algos_torch.fb_task_inference import mixed_reward_commands, stage_rewards
from rl_games.algos_torch.fb_cpr_networks import TemporalReplay


def make_mixed_agent(tmp_path):
    agent = make_agent(tmp_path)
    agent.replay = TemporalReplay(60, 3, 1, 'cpu')
    agent.options.update(mixed_reward_commands=True, reward_samples_per_source=4,
                         tracking_length=250, command_steps=100, lifted_state_index=19,
                         logo_coefficient=.005, logo_horizon=8, logo_local_weight=4/7,
                         logo_goal_weight=3/7)
    agent.model.local_prompt = True
    agent.reward_generator = torch.Generator().manual_seed(15)
    agent.reward_sidecar = {'next_state': torch.randn(20, 20), 'reward_components': torch.ones(20, 10)}
    agent.reward_sidecar_identity = {'sha256': 'fixture', 'population': 20}
    for step in range(16):
        state = torch.randn(1, 3, 20)
        agent.replay.add({
            'observation': torch.randn(1, 3, 16), 'next_observation': torch.randn(1, 3, 16),
            'state': state, 'next_state': state + .1,
            'history': torch.randn(1, 3, 64), 'next_history': torch.randn(1, 3, 64),
            'previous_action': torch.randn(1, 3, 3), 'action': torch.randn(1, 3, 3).tanh(),
            'command': torch.randn(1, 3, 4), 'reward': torch.ones(1, 3, 1),
            'command_source': torch.tensor([[[0], [2], [3]]]),
            'reward_components': torch.ones(1, 3, 10), 'auxiliary_reward': -torch.rand(1, 3, 1),
            'discount': torch.ones(1, 3, 1) * .98, 'valid': torch.ones(1, 3, 1, dtype=torch.bool),
        })
    return agent


def test_signed_mean_and_unsupported_rewards():
    features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., 2.]])
    components = torch.zeros(4, 10)
    components[:, 0] = torch.tensor([1., 2., 3., 4.])
    components[:, 4] = torch.tensor([3., 4., 5., 6.])
    components[:, 5] = -1.
    commands, metrics, valid = mixed_reward_commands(features, components, 2)
    expected = stage_rewards(components).T @ features / 4
    expected *= 2**.5 / expected.norm(dim=-1, keepdim=True)
    assert valid
    torch.testing.assert_close(commands, expected)
    for name in ('grasp_lift', 'track', 'full'):
        torch.testing.assert_close(metrics[name+'/ppo_direction_contribution'] + metrics[name+'/online_direction_contribution'], torch.tensor(1.))
    components.zero_()
    components[:, 5] = -1
    assert not mixed_reward_commands(features, components, 2)[2]
    components[:, 0] = float('nan')
    assert not mixed_reward_commands(features, components, 2)[2]


def test_refresh_is_detached_uses_local_generator_and_freezes_normalizer(tmp_path):
    torch.manual_seed(23)
    agent = make_mixed_agent(tmp_path)
    before = {name: value.clone() for name, value in agent.state_normalizer.state_dict().items()}
    global_rng = torch.get_rng_state().clone()
    sample_rng = agent.reward_generator.get_state().clone()
    metrics = agent._refresh_mixed_reward_bank()
    assert torch.equal(global_rng, torch.get_rng_state())
    assert not torch.equal(sample_rng, agent.reward_generator.get_state())
    for name, value in before.items():
        torch.testing.assert_close(value, agent.state_normalizer.state_dict()[name], rtol=0, atol=0)
    assert not agent.mixed_reward_bank['commands'].requires_grad
    assert all(parameter.grad is None for parameter in agent.model.parameters())
    assert agent.mixed_reward_bank['before_optimizer_step'] == 0
    assert all(torch.isfinite(value) for value in metrics.values())


def test_real_update_relabels_to_reward_and_preserves_style_and_logo(tmp_path):
    agent = make_mixed_agent(tmp_path)
    before = {name: value.clone() for name, value in agent.model.named_parameters()}
    metrics = agent.update()
    assert agent.optimizer_step == 1
    assert agent.mixed_reward_bank['before_optimizer_step'] == 0
    assert metrics['logo/available'] == 1
    assert all(torch.isfinite(value) for value in metrics.values())
    for prefix in ('forward_maps', 'backward', 'actor', 'style_maps', 'auxiliary_maps', 'discriminator'):
        assert any(not torch.equal(value, before[name]) for name, value in agent.model.named_parameters() if name.startswith(prefix))
    reward_rows = agent.z_source_buffer[:agent.z_size] == 2
    assert reward_rows.any()
    torch.testing.assert_close(agent.z_buffer[:agent.z_size][reward_rows],
                               agent.mixed_reward_bank['commands'][2].expand(int(reward_rows.sum()), -1))
    assert agent._command_source_names()[2:] == ('reward_full', 'reward_phased')


def test_rollout_warmup_then_phase_uses_observed_lifted_and_no_expert(tmp_path):
    agent = make_mixed_agent(tmp_path)
    agent.experts.sample = lambda *args: (_ for _ in ()).throw(AssertionError('Unexpected PPO command sampling'))
    agent.reset_before = torch.ones(3, dtype=torch.bool)
    agent.command_age = torch.zeros(3, dtype=torch.long)
    agent.rollout_command = torch.zeros(3, 4)
    agent.rollout_command_source = torch.zeros(3, dtype=torch.long)
    agent.rollout_step = 0
    agent.tracking_indices = torch.arange(3)
    agent.tracking_prompts = None
    agent.observations = {'states': torch.zeros(3, 20)}
    agent.observations['states'][1, 19] = 1
    agent._rollout_prompts()
    assert (agent.rollout_command_source == 0).all()
    agent._refresh_mixed_reward_bank()
    agent._rollout_prompts()
    torch.testing.assert_close(agent.rollout_command,
                               agent.mixed_reward_bank['commands'][torch.tensor([0, 1, 0])])
    assert (agent.rollout_command_source == 3).all()


def test_failure_keeps_batch_and_does_not_update_model(tmp_path):
    agent = make_mixed_agent(tmp_path)
    agent.reward_sidecar['reward_components'].zero_()
    agent.replay.storage['reward_components'].zero_()
    before = {name: value.clone() for name, value in agent.model.state_dict().items()}
    with pytest.raises(FloatingPointError):
        agent.update()
    assert agent.optimizer_step == 0
    assert (agent.output_dir / 'reward_bank_failure.rank0.pt').exists()
    for name, value in before.items():
        torch.testing.assert_close(value, agent.model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('feature_scale', [0., 1e-12, float('inf')])
def test_gate_rejects_zero_tiny_and_nonfinite_directions(feature_scale):
    components = torch.ones(4, 10)
    features = torch.ones(4, 2) * feature_scale
    assert not mixed_reward_commands(features, components, 2)[2]


def test_gate_rejects_cancellation_and_projects_in_float32():
    components = torch.ones(4, 10, dtype=torch.float64)
    features = torch.tensor([[1., 2.], [3., 4.], [-1., -2.], [-3., -4.]], dtype=torch.float64)
    assert not mixed_reward_commands(features, components, 2)[2]
    commands, _, valid = mixed_reward_commands(features.abs(), components, 2)
    assert valid and commands.dtype == torch.float32
    torch.testing.assert_close(commands.norm(dim=-1), torch.full((3,), 2**.5))


def test_uniform_valid_sampling_records_each_source_identity(tmp_path):
    agent = make_mixed_agent(tmp_path)
    agent.options['reward_samples_per_source'] = 1024
    # A terminal row stays eligible, while a physical-reset row is excluded.
    agent.replay.storage['discount'][0, 0, 0] = 0
    agent.replay.valid_indices[0] = torch.tensor([0, 2])
    state = agent.replay.state_dict()
    agent.replay.load_state_dict(state)
    agent._refresh_mixed_reward_bank()
    bank = agent.mixed_reward_bank
    assert bank['ppo_indices'].shape == bank['online_rows'].shape == (1024,)
    assert not ((bank['online_blocks'] == 0) & (bank['online_rows'] == 1)).any()
    assert ((bank['online_blocks'] == 0) & (bank['online_rows'] == 0)).any()
    assert bank['ppo_indices'].unique().numel() == len(agent.reward_sidecar['next_state'])
    raw = bank['ppo_contribution'] + bank['online_contribution']
    torch.testing.assert_close(bank['commands'], raw / raw.norm(dim=-1, keepdim=True) * 2)


def test_phase_reselection_restores_original_held_commands(tmp_path, monkeypatch):
    agent = make_mixed_agent(tmp_path)
    agent._refresh_mixed_reward_bank()
    agent.reset_before = torch.ones(3, dtype=torch.bool)
    agent.command_age = torch.zeros(3, dtype=torch.long)
    agent.rollout_command = torch.zeros(3, 4)
    agent.rollout_command_source = torch.zeros(3, dtype=torch.long)
    agent.rollout_step = 0
    agent.tracking_indices = torch.tensor([0])
    agent.tracking_prompts = None
    agent.observations = {'states': torch.zeros(3, 20)}
    choices = iter((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))
    monkeypatch.setattr(torch, 'randperm', lambda *args, **kwargs: next(choices))
    agent._rollout_prompts()
    held = agent.held_command.clone()
    assert agent.rollout_command_source.tolist() == [3, 0, 0]
    agent.reset_before.zero_()
    agent.command_age.fill_(50)
    agent.rollout_step = 250
    agent.observations['states'][1, 19] = 1
    agent._rollout_prompts()
    assert agent.rollout_command_source.tolist() == [0, 3, 0]
    torch.testing.assert_close(agent.rollout_command[0], held[0])
    torch.testing.assert_close(agent.rollout_command[1], agent.mixed_reward_bank['commands'][1])


def test_remote_bank_failure_stops_before_discriminator(tmp_path, monkeypatch):
    from rl_games.algos_torch import fb_cpr_agent
    agent = make_mixed_agent(tmp_path)
    agent.world_size = 2
    monkeypatch.setattr(fb_cpr_agent.distributed, 'all_reduce', lambda tensor, **kwargs: tensor.zero_())
    agent._discriminator_update = lambda *args: pytest.fail('Entered a later training collective')
    with pytest.raises(FloatingPointError, match='another rank'):
        agent.update()
    saved = torch.load(agent.output_dir / 'reward_bank_failure.rank0.pt')
    assert saved['local_valid']
    assert len(saved['online_rows']) == len(saved['ppo_indices']) == 4
    assert agent.optimizer_step == 0


def test_source_gradient_diagnostics_do_not_change_rng_or_update(tmp_path):
    agent = make_mixed_agent(tmp_path)
    generator = torch.Generator()
    generator.set_state(agent.reward_generator.get_state())
    observed = copy.deepcopy(agent, {id(agent.reward_generator): generator})
    observed.options['actor_gradient_probe_every'] = 1
    rng = torch.get_rng_state()
    agent.update()
    after_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    metrics = observed.update()
    assert torch.equal(torch.get_rng_state(), after_rng)
    for name, value in agent.model.state_dict().items():
        torch.testing.assert_close(value, observed.model.state_dict()[name], rtol=0, atol=0)
    for source in observed._command_source_names():
        assert 'actor_grad/' + source + '_sample_count' in metrics
        assert 'actor_grad/' + source + '_fb_style_cosine' in metrics
        assert 'discriminator_source/' + source + '/count' in metrics
    assert 'discriminator_fixed_state_swap/reward_full/delta_from_behavior' in metrics


def test_rank_checkpoint_binds_model_replay_and_restores_sampling_rng(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    agent = make_mixed_agent(tmp_path)
    agent.options.update(warmup_blocks=2, tracking_fraction=.5)
    agent.update()
    agent.run_started = time.monotonic()
    agent.rollout_step = 16
    agent.observations = {'obs': torch.zeros(3, 16), 'states': torch.zeros(3, 20)}
    agent.actor_state = ()
    agent.rollout_command = torch.zeros(3, 4)
    agent.command_age = torch.zeros(3, dtype=torch.long)
    agent.reset_before = torch.zeros(3, dtype=torch.bool)
    agent.episode_reward = torch.zeros(3)
    agent.previous_block = agent.previous_initial_state = None
    agent.vec_env.get_env_state = lambda: {}
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: torch.zeros(8, dtype=torch.uint8))
    path = tmp_path / 'resume.pt'
    agent.save(path, replay=True)
    command_path = tmp_path / 'resume.rank0.commands.pt'
    saved = torch.load(command_path)
    assert saved['warmup_complete'] and saved['replay_state'] == agent._reward_replay_state()
    generator = torch.Generator()
    generator.set_state(agent.reward_generator.get_state())
    expected = copy.deepcopy(agent, {id(agent.reward_generator): generator})
    expected._refresh_mixed_reward_bank()
    restored = make_mixed_agent(tmp_path / 'restored')
    restored.options.update(warmup_blocks=2, tracking_fraction=.5)
    restored.vec_env.env.reset_buf = torch.ones(3, dtype=torch.long)
    restored.vec_env.reset = lambda: {'obs': torch.zeros(3, 16), 'states': torch.zeros(3, 20)}
    restored.restore(path)
    restored._initialize_rollout()
    replay = torch.load(tmp_path / 'resume.rank0.replay.pt')['replay']
    restored.replay.load_state_dict(replay)
    assert restored._reward_replay_state() == restored.reward_resume_replay_state
    # The sidecar fixture is an independent data source, not part of the model.
    restored.reward_sidecar = copy.deepcopy(agent.reward_sidecar)
    restored._refresh_mixed_reward_bank()
    torch.testing.assert_close(restored.mixed_reward_bank['commands'], expected.mixed_reward_bank['commands'], rtol=0, atol=0)
    assert restored.rollout_step == 16
    with (tmp_path / 'resume.rank0.replay.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='snapshot'):
        restored._initialize_rollout()


def test_production_sidecar_collection_timing_terminal_and_identity(tmp_path):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from rl_games.algos_torch.fb_task_inference import checkpoint_digest, reward_components
    agent = make_mixed_agent(tmp_path)
    agent.num_actors = 6144
    teacher_root = tmp_path / 'teacher'
    teacher_root.mkdir()
    (teacher_root / 'model.pth').write_bytes(b'CPU teacher fixture')
    (teacher_root / 'config.yaml').write_text('fixture: true\n')
    agent.options.update(expert_steps=1500, expert_environments=256, teacher_path=str(teacher_root))
    environment = SimpleNamespace(
        cfg=omegaconf_to_dict(OmegaConf.create({
            'test': False, 'env': {'successTolerance': .1, 'obsList': ['fixture']},
            'sim': {'gravity': [0., 0., -9.81], 'nested': [{'values': [1, 2]}]},
        })),
        success_tolerance=.1, initial_tolerance=.1, frame_since_restart=0, last_curriculum_update=0,
        object_asset_files=['fixture-a.urdf', 'fixture-b.urdf'],
        object_scales=torch.ones(6144, 3), reset_goal_buf=torch.zeros(6144, dtype=torch.long),
        goal_states=torch.zeros(6144, 13), root_state_tensor=torch.zeros(12288, 13),
        object_indices=torch.arange(6144), table_indices=torch.arange(6144, 12288),
        object_init_state=torch.zeros(6144, 13), lifted_object=torch.zeros(6144, dtype=torch.bool),
        successes=torch.zeros(6144),
    )
    observations = {'obs': torch.zeros(6144, 16), 'states': torch.zeros(6144, 20)}
    step_number = 0

    def step(action):
        nonlocal step_number
        environment.goal_states.add_(environment.reset_goal_buf[:, None])
        observations['states'].fill_(step_number + 1)
        done = torch.full((6144,), step_number % 3 == 1, dtype=torch.bool)
        environment.reset_goal_buf.fill_(step_number % 5 == 1)
        components = {name: torch.full((6144,), float(step_number + 1)) for name in reward_components}
        reward = torch.full((6144,), float(10 * (step_number + 1)))
        step_number += 1
        return observations, reward, done, {'episode_cumulative': components}

    class Teacher:
        calls = 0

        def get_default_rnn_state(self):
            return [torch.zeros(1, 6144, 1)]

        def __call__(self, inputs):
            if self.calls > 0:
                expected = 0. if (self.calls - 1) % 3 == 1 else 1.
                assert (inputs['rnn_states'][0] == expected).all()
            self.calls += 1
            return {'mus': torch.zeros(6144, 3), 'rnn_states': [torch.ones(1, 6144, 1)]}

    agent.vec_env = SimpleNamespace(env=environment, reset=lambda: observations, step=step)
    agent._load_reference_policy = lambda: (Teacher(), {}, teacher_root)
    agent._status = lambda *args, **kwargs: None
    agent._collect_reward_sidecar()
    path = agent.output_dir / 'reward_sidecar.rank0.pt'
    data = torch.load(path)
    assert isinstance(data['collection_environment_config']['env']['obsList'], list)
    assert isinstance(data['collection_environment_config']['sim']['nested'][0], dict)
    rows = data['rows']
    assert data['simulation_transitions'] == 9216000
    assert data['stored_transitions'] == 384000
    assert rows['valid'][1].all() and rows['done'][1].all()
    assert not rows['valid'][2].any()
    torch.testing.assert_close(rows['next_state'][:, :, 0], rows['source_step'].float() + 1)
    torch.testing.assert_close(rows['state'][:, :, 0], rows['source_step'].float())
    torch.testing.assert_close(rows['reward_components'][:, :, 0], rows['next_state'][:, :, 0])
    assert rows['goal_reset_after'][1].all() and rows['goal_reset_before'][2].all()
    assert rows['goal'][1, 0, 0] == 0 and rows['goal'][2, 0, 0] == 1
    agent.options.update(reward_sidecar_path=str(path), reward_sidecar_sha256=[checkpoint_digest(path)],
                         reward_sidecar_environment_sha256=[data['collection_environment_sha256']],
                         reward_sidecar_seed=[data['seed']])
    agent._load_reward_sidecar()
    assert len(agent.reward_sidecar['next_state']) == int(rows['valid'].sum())
    assert (agent.reward_sidecar_rows[:, 0] == 1).any()
    assert not (agent.reward_sidecar_rows[:, 0] == 2).any()
    agent.options['reward_sidecar_seed'][0] += 1
    with pytest.raises(ValueError, match='seed mismatch'):
        agent._load_reward_sidecar()
    agent.output_dir = tmp_path / 'fixed-goal-collection'
    agent.output_dir.mkdir()
    environment.cfg['env']['useFixedGoalStates'] = True
    with pytest.raises(ValueError, match='original training curriculum and goals'):
        agent._collect_reward_sidecar()


def test_evaluation_initialization_does_not_require_training_sidecar_or_replay(tmp_path):
    agent = make_mixed_agent(tmp_path)
    agent.options['tracking_fraction'] = .5
    agent.last_checkpoint = tmp_path / 'model-only.pt'
    agent.vec_env.env.reset_buf = torch.zeros(3, dtype=torch.long)
    agent.vec_env.reset = lambda: {'obs': torch.zeros(3, 16), 'states': torch.zeros(3, 20)}
    agent._initialize_rollout(restore_training=False)
    assert agent.rollout_step == 0
    assert agent.mixed_reward_bank is None


def test_procedural_scene_identity_survives_process_tempdir_changes(tmp_path):
    from types import SimpleNamespace
    agent = make_mixed_agent(tmp_path)
    agent.vec_env.env = SimpleNamespace(
        cfg={'env': {'objectName': 'handle_head_primitives'}},
        object_asset_files=[Path('/tmp/process-a/handle_head_primitives/000_shape_density.urdf')],
        object_scales=torch.ones(3, 3))
    original = agent._reward_scene_mapping()
    agent.vec_env.env.object_asset_files = [Path('/tmp/process-b/handle_head_primitives/000_shape_density.urdf')]
    assert agent._reward_scene_mapping() == original
    agent.vec_env.env.object_scales[1, 0] = 2
    assert agent._reward_scene_mapping() != original


@pytest.mark.parametrize('rank', range(4))
def test_actual_composed_collection_config_keeps_plain_digest(rank):
    import json
    from hydra import compose, initialize_config_dir
    from omegaconf import ListConfig, OmegaConf
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from rl_games.algos_torch.fb_cpr_agent import FBcprAgent

    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / 'cfg'), version_base='1.1'):
        config = compose(config_name='config', overrides=[
            'task=SimToolRealLSTMAsymmetric', 'train=SimToolRealFBV15',
            'headless=True', 'num_envs=6144', 'multi_gpu=True', 'seed=0',
            'force_render=False', 'graphics_device_id=-1', 'wandb_activate=False',
            'task.env.enableCameraSensors=False', 'task.env.capture_video=False',
            'task.env.objectScaleNoiseMultiplierRange=[0.9,1.1]',
            'task.env.forceConsecutiveNearGoalSteps=True',
            'task.env.clearActionQueueOnReset=True', 'task.env.referenceResetProbability=1.0',
            'train.params.config.fb.reference_reward_collection=True',
        ])
    config.seed = rank
    # This is the actual make()->omegaconf_to_dict conversion, not a synthetic
    # plain-dict approximation. Runtime-derived dimensions follow Sharpa.
    hybrid = omegaconf_to_dict(config.task)
    expected = OmegaConf.to_container(config.task, resolve=True)
    assert isinstance(hybrid, dict) and isinstance(hybrid['env']['stateList'], ListConfig)
    with pytest.raises(TypeError, match='ListConfig'):
        json.dumps(hybrid)
    for task in (hybrid, expected):
        task['env'].update(numStates=162, numObservations=140, numActions=29)
        task.update(rank=rank, rl_device='cuda:' + str(rank), device_type='cuda', device_id=rank, headless=True)
    rng = torch.get_rng_state()
    plain = FBcprAgent._reward_plain_config(hybrid)
    assert plain == expected
    assert FBcprAgent._reward_json_digest(plain) == FBcprAgent._reward_json_digest(expected)
    assert torch.equal(torch.get_rng_state(), rng)
    assert isinstance(hybrid['env']['stateList'], ListConfig)
    assert isinstance(plain['env']['stateList'], list)


def test_recursive_config_conversion_resolves_original_list_parent():
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from rl_games.algos_torch.fb_cpr_agent import FBcprAgent
    config = OmegaConf.create({'outside': 7, 'task': {'nested': [{'value': '${outside}'}]}})
    hybrid = omegaconf_to_dict(config.task)
    assert FBcprAgent._reward_plain_config(hybrid) == {'nested': [{'value': 7}]}
    plain = {'scalar': 3, 'items': [True, None, {'negative': -2.5}]}
    assert FBcprAgent._reward_json_digest(FBcprAgent._reward_plain_config(plain)) == FBcprAgent._reward_json_digest(plain)


@pytest.mark.parametrize('invalid_section', ['configuration', 'scene_mapping', 'curriculum'])
def test_collection_metadata_failure_precedes_reset_and_sampling(tmp_path, invalid_section):
    from types import SimpleNamespace
    agent = make_mixed_agent(tmp_path)
    agent.num_actors = 6144
    agent.options.update(expert_steps=1500, expert_environments=256)
    environment = SimpleNamespace(cfg={'env': {}}, success_tolerance=.1, initial_tolerance=.1,
                                  frame_since_restart=0, last_curriculum_update=0)
    mapping = {'assets': []}
    if invalid_section == 'configuration':
        environment.cfg['unsupported'] = object()
    elif invalid_section == 'scene_mapping':
        mapping['unsupported'] = object()
    else:
        environment.frame_since_restart = torch.tensor(0)
    agent.vec_env = SimpleNamespace(env=environment, reset=lambda: pytest.fail('Reset before metadata validation'))
    agent._load_reference_policy = lambda: (None, {}, tmp_path / 'absent-teacher')
    agent._reward_scene_mapping = lambda: mapping
    rng = torch.get_rng_state()
    sampling_rng = agent.reward_generator.get_state()
    with pytest.raises(TypeError, match='JSON serializable'):
        agent._collect_reward_sidecar()
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(agent.reward_generator.get_state(), sampling_rng)
    assert not (agent.output_dir / 'reward_sidecar.rank0.pt').exists()
