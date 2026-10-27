"""Raw-state geometry and production learner/inference contracts for v10."""
from types import SimpleNamespace
import gym
import torch

from rl_games.algos_torch.fb_cpr_agent import FBcprAgent
from rl_games.algos_torch.fb_cpr_networks import (
    BatchNormalizer, ObjectGoalNormalizer, CPRModel, ExpertSequences, make_state_normalizer,
)


def options():
    return dict(latent_size=4, residual_width=16, residual_layers=2, history_steps=4,
                backward_learning_rate=1e-5, discriminator_learning_rate=1e-5,
                discriminator_penalty=10., replay_blocks=2, replay_capacity=48, z_buffer_size=32,
                expert_sequence_length=2, exploration_std=.05, target_noise_clip=.3,
                orthogonal_coefficient=100., relabel_probability=.8, target_tau=.01,
                critic_target_tau=.005, auxiliary_coefficient=.02, style_coefficient=.05,
                task_reward_temperature=.05, cpu_threads=2, object_goal_features=True,
                backward_state_size=40, backward_state_indices=list(range(162, 202)),
                discriminator_state_indices=list(range(152)) + [158], local_prompt=True)


def test_geometry_is_independent_of_hand_and_retains_separate_goal():
    state = torch.randn(5, 162)
    features = ObjectGoalNormalizer.features(state)
    changed = state.clone()
    displacement = torch.randn(5, 3)
    changed[:, 87:90] += displacement
    changed[:, 125:137] -= displacement[:, None].expand(-1, 4, -1).flatten(1)
    changed[:, :87] += 10
    changed[:, 90:100] += 20
    changed[:, 110:125] += 30
    changed[:, 152:] += 40
    torch.testing.assert_close(ObjectGoalNormalizer.features(changed), features)
    changed = state.clone()
    changed[:, 137:149] += 1
    shifted = ObjectGoalNormalizer.features(changed)
    torch.testing.assert_close(shifted[:, :25], features[:, :25])
    torch.testing.assert_close(shifted[:, 25:37], features[:, 25:37] - 1)
    torch.testing.assert_close(shifted[:, 37:], features[:, 37:])


def test_normalization_preserves_original_fields_and_checkpoint_schema():
    torch.manual_seed(4)
    original, expanded = BatchNormalizer(162), make_state_normalizer(162, options())
    for _ in range(3):
        state = torch.randn(7, 162)
        torch.testing.assert_close(original(state), expanded(state)[:, :162], rtol=2e-7, atol=2e-7)
    expanded.eval()
    restored = make_state_normalizer(162, options()).eval()
    restored.load_state_dict(expanded.state_dict())
    torch.testing.assert_close(restored(state), expanded(state), rtol=0, atol=0)
    assert restored.running_mean.shape == (202,)
    assert isinstance(make_state_normalizer(162, {}), BatchNormalizer)


def test_backward_discriminator_and_critic_have_separate_inputs():
    model = CPRModel(140, 162, 29, options(), {})
    state, history, command = torch.randn(2, 202), torch.randn(2, 589), torch.randn(2, 4)
    changed = state.clone()
    changed[:, :162] += 1
    torch.testing.assert_close(model.backward(state), model.backward(changed), rtol=0, atol=0)
    changed = state.clone()
    changed[:, 162:] += 100
    torch.testing.assert_close(model.discriminate(state, command), model.discriminate(changed, command), rtol=0, atol=0)
    torch.testing.assert_close(model.critic_state(state, history), torch.cat((state[:, :162], history), -1))
    assert model.critic_state(state, history).shape == (2, 751)
    assert model.forward_maps[0].task_embedding[0][1].in_features == 755


def make_agent(path):
    vector = SimpleNamespace(env=SimpleNamespace(obs_list=['keypoints_rel_goal', 'joint_pos'],
        obs_type_size_dict={'keypoints_rel_goal': 12, 'joint_pos': 128}), get_env_info=lambda: {
        'observation_space': gym.spaces.Box(-10, 10, (140,)),
        'state_space': gym.spaces.Box(-10, 10, (162,)),
        'action_space': gym.spaces.Box(-1, 1, (29,))})
    agent = FBcprAgent('test', {'config': {'fb': options(), 'device': 'cpu', 'num_actors': 3,
        'vec_env': vector, 'seq_length': 4, 'horizon_length': 4, 'minibatch_size': 8,
        'learning_rate': 3e-4, 'gamma': .98, 'grad_norm': 1e30, 'train_dir': str(path),
        'name': 'test'}, 'network': {}})
    agent.experts = ExpertSequences(torch.randn(30, 3, 162), torch.ones(30, 3, dtype=torch.bool))
    return agent


def test_production_update_raw_replay_and_restore(tmp_path):
    agent = make_agent(tmp_path / 'train')
    agent.replay.add({
        'observation': torch.randn(4, 3, 140), 'next_observation': torch.randn(4, 3, 140),
        'state': torch.randn(4, 3, 162), 'next_state': torch.randn(4, 3, 162),
        'history': torch.randn(4, 3, 560), 'next_history': torch.randn(4, 3, 560),
        'previous_action': torch.randn(4, 3, 29), 'action': torch.randn(4, 3, 29).tanh(),
        'command': torch.randn(4, 3, 4), 'reward': torch.randn(4, 3, 1),
        'auxiliary_reward': -torch.rand(4, 3, 1), 'discount': torch.ones(4, 3, 1) * .98,
        'valid': torch.ones(4, 3, 1, dtype=torch.bool),
    })
    before = {key: value.clone() for key, value in agent.model.state_dict().items()}
    metrics = agent.update()
    assert all(torch.isfinite(value) for value in metrics.values())
    for prefix in ('backward', 'forward_maps', 'style_maps', 'auxiliary_maps', 'actor', 'discriminator'):
        assert any(not torch.equal(value, before[key]) for key, value in agent.model.state_dict().items() if key.startswith(prefix))
    assert agent.replay.sample(2)['state'].shape[-1] == 162
    agent.run_started = __import__('time').monotonic()
    checkpoint = tmp_path / 'roundtrip.pt'
    agent.save(checkpoint)
    restored = make_agent(tmp_path / 'restore')
    restored.restore(checkpoint)
    state = torch.randn(3, 162)
    torch.testing.assert_close(restored.model.backward(restored.state_normalizer(state)),
                               agent.model.backward(agent.state_normalizer(state)), rtol=0, atol=0)
