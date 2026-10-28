"""Task inference uses real learner/replay code, without native Gym dependency."""

import copy

import pytest
import torch

from test_fb_cpr import make_agent
from rl_games.algos_torch.fb_task_inference import (
    reward_components, stage_rewards, infer_command_bank, select_phase_commands,
)
from rl_games.algos_torch.fb_networks import project_task


def fill_replay(agent):
    components = torch.rand(4, 3, len(reward_components))
    components[..., [1, 5, 6, 8, 9]] *= -.01
    agent.replay.add({
        'observation': torch.randn(4, 3, 16), 'next_observation': torch.randn(4, 3, 16),
        'state': torch.randn(4, 3, 20), 'next_state': torch.randn(4, 3, 20),
        'history': torch.randn(4, 3, 64), 'next_history': torch.randn(4, 3, 64),
        'previous_action': torch.randn(4, 3, 3), 'action': torch.randn(4, 3, 3).tanh(),
        'command': torch.randn(4, 3, 4), 'reward': components.sum(-1, keepdim=True),
        'reward_components': components, 'auxiliary_reward': -torch.rand(4, 3, 1),
        'discount': torch.ones(4, 3, 1) * .98, 'valid': torch.ones(4, 3, 1, dtype=torch.bool),
    })


def export_pool(agent, tmp_path):
    agent.options.update(record_inference_pool=True, inference_samples=48, inference_seed=7300)
    agent._save_inference_pool(tmp_path / 'last.pt')
    return [tmp_path / 'last.rank0.inference.pt']


def test_original_reward_partition_and_phase_selection():
    components = torch.arange(1, 11).float()[None]
    phases = stage_rewards(components)
    penalties = 6 + 7 + 9 + 10
    torch.testing.assert_close(phases, torch.tensor([[10 + penalties, 13 + penalties, 55.]]))
    bank = torch.eye(3)
    lifted = torch.tensor([False, True, False])
    torch.testing.assert_close(select_phase_commands(bank, lifted, 'phased'), bank[[0, 1, 0]])
    torch.testing.assert_close(select_phase_commands(bank, lifted, 'full'), bank[2].expand(3, -1))


def test_pool_preserves_rng_and_matches_direct_reward_inference(tmp_path):
    agent = make_agent(tmp_path)
    fill_replay(agent)
    before = torch.get_rng_state().clone()
    paths = export_pool(agent, tmp_path)
    assert torch.equal(before, torch.get_rng_state())
    pool = torch.load(paths[0])
    assert pool['sample_count'] == 48
    assert pool['reward_components'].dtype == torch.float32
    features = agent.model.backward(agent.state_normalizer(pool['next_state']))
    rewards = stage_rewards(pool['reward_components'])
    for method in ('mean', 'reward_weighted'):
        weights = torch.ones_like(rewards) / len(rewards)
        if method == 'reward_weighted':
            weights = torch.softmax(10 * rewards, dim=0)
        expected = project_task((weights * rewards).T @ features)
        bank = infer_command_bank(agent.model.backward, agent.state_normalizer, paths,
                                  0, 0, 'cpu', method, .1, 7, 1)
        torch.testing.assert_close(bank['commands'], expected, atol=2e-6, rtol=2e-6)
        assert bank['sample_count'] == 48
    with pytest.raises(ValueError, match='counters'):
        infer_command_bank(agent.model.backward, agent.state_normalizer, paths, 1, 0, 'cpu', world_size=1)
    with pytest.raises(ValueError, match='every training rank'):
        infer_command_bank(agent.model.backward, agent.state_normalizer, paths, 0, 0, 'cpu', world_size=4)


def test_missing_positive_task_support_is_not_replaced_by_random_z(tmp_path):
    agent = make_agent(tmp_path)
    fill_replay(agent)
    paths = export_pool(agent, tmp_path)
    pool = torch.load(paths[0])
    pool['reward_components'][:, [4, 7]] = 0
    torch.save(pool, paths[0])
    with pytest.raises(ValueError, match='no positive task evidence'):
        infer_command_bank(agent.model.backward, agent.state_normalizer, paths, 0, 0, 'cpu', world_size=1)


def test_disabling_cached_tracking_z_does_not_change_learning(tmp_path):
    torch.manual_seed(19)
    original = make_agent(tmp_path / 'v2')
    fill_replay(original)
    torch.manual_seed(19)
    revised = make_agent(tmp_path / 'v3')
    revised.replay.load_state_dict(copy.deepcopy(original.replay.state_dict()))
    revised.options['cache_tracking_command'] = False
    old_command = revised.task_command.clone()
    for agent in (original, revised):
        torch.manual_seed(100)
        agent.update()
    torch.testing.assert_close(revised.task_command, old_command, rtol=0, atol=0)
    for key, value in original.model.state_dict().items():
        torch.testing.assert_close(revised.model.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(original.z_buffer[:original.z_size], revised.z_buffer[:revised.z_size])


def test_source_labels_follow_cached_commands_without_extra_sampling(tmp_path):
    agent = make_agent(tmp_path)
    commands = torch.arange(32.).reshape(8, 4)
    sources = torch.arange(8) % 3
    agent._cache_commands(commands, sources)
    chosen = agent._sample_commands(100)
    rows = (chosen[:, 0] / 4).long()
    torch.testing.assert_close(agent.sampled_command_source, sources[rows])


def test_frozen_cli_bank_and_production_player_loading(tmp_path, monkeypatch):
    import sys
    import time
    from rl_games.algos_torch import fb_task_inference
    from rl_games.algos_torch.fb_cpr_agent import FBcprPlayer
    agent = make_agent(tmp_path)
    agent.options.update(cache_tracking_command=False, backward_state_size=20,
                         evaluation_command_mode='phased', inference_world_size=1)
    fill_replay(agent)
    export_pool(agent, tmp_path)
    agent.run_started = time.monotonic()
    checkpoint = tmp_path / 'last.pt'
    agent.save(checkpoint)
    assert torch.load(checkpoint)['task_command'] is None
    output = tmp_path / 'bank.pt'
    monkeypatch.setattr(sys, 'argv', ['inference', '--checkpoint', str(checkpoint),
                                    '--pool-glob', str(tmp_path / '*.inference.pt'),
                                    '--output', str(output)])
    fb_task_inference.main()
    agent.options['command_bank_path'] = str(output)
    FBcprPlayer._prepare_task_commands(agent)
    assert agent.command_bank.shape == (3, 4)
    bank = torch.load(output)
    bank['checkpoint_sha256'] = 'wrong'
    torch.save(bank, output)
    with pytest.raises(ValueError, match='different checkpoint'):
        FBcprPlayer._prepare_task_commands(agent)


def test_command_reward_logging_uses_counts_not_average_of_means(tmp_path):
    from types import SimpleNamespace
    agent = make_agent(tmp_path)
    agent.writer = SimpleNamespace(add_scalar=lambda *args: None)
    result = agent._log({'command_count/random': 4.,
                         'command_reward_sum/random/total_reward': 6.,
                         'command_count/expert_mean': 2.,
                         'command_reward_sum/expert_mean/total_reward': 8.})
    assert result['command_reward_step/random/total_reward'] == 1.5
    assert result['command_reward_step/expert_mean/total_reward'] == 4.


def test_object_tracking_cli_uses_own_checkpoint_and_four_pools(tmp_path, monkeypatch):
    import sys
    from rl_games.algos_torch import fb_task_inference
    from rl_games.algos_torch.fb_cpr_networks import BackwardMap, BatchNormalizer
    torch.manual_seed(37)
    backward = BackwardMap(153, 4, list(range(152)) + [158])
    normalizer = BatchNormalizer(162)
    checkpoint = tmp_path / 'last.pt'
    torch.save({'options': {'backward_state_size': 153, 'latent_size': 4,
                           'backward_state_indices': list(range(152)) + [158]},
                'model': {'backward.' + key: value for key, value in backward.state_dict().items()},
                'state_normalizer': normalizer.state_dict(), 'schema': {'state': 162},
                'frame': 19120128, 'optimizer_step': 11440, 'training_world_size': 4}, checkpoint)
    states = []
    paths = []
    for rank in range(4):
        state = torch.randn(8, 162) * .02
        state[:, 137:149] *= rank + 1
        states.append(state)
        path = tmp_path / ('last.rank' + str(rank) + '.inference.pt')
        paths.append(path)
        torch.save({'next_state': state, 'reward_components': torch.ones(8, 10),
                    'schema_version': 1, 'reward_names': reward_components,
                    'rank': rank, 'world_size': 4, 'frame': 19120128, 'optimizer_step': 11440,
                    'sample_count': 8, 'population': 100}, path)
    backward.eval()
    normalizer.eval()
    all_states = torch.cat(states)
    features = backward(normalizer(all_states))
    distance = all_states[:, 137:149].reshape(-1, 4, 3).norm(dim=-1).amax(-1)
    expected = project_task((torch.exp(-distance / .05)[:, None] * features).mean(0, keepdim=True))
    for kind in ('original', 'object_tracking'):
        output = tmp_path / (kind + '.pt')
        monkeypatch.setattr(sys, 'argv', ['inference', '--checkpoint', str(checkpoint),
            '--pool-glob', str(tmp_path / '*.inference.pt'), '--output', str(output),
            '--reward-definition', kind, '--batch-size', '7'])
        fb_task_inference.main()
        bank = torch.load(output)
        assert bank['reward_kind'] == kind
        assert bank['sample_count'] == 32
        assert [row['rank'] for row in bank['sources']] == list(range(4))
        assert bank['checkpoint_sha256'] == fb_task_inference.checkpoint_digest(checkpoint)
        if kind == 'object_tracking':
            torch.testing.assert_close(bank['commands'], expected.expand(3, -1), atol=2e-6, rtol=2e-6)
        else:
            original = infer_command_bank(backward, normalizer, paths, 19120128, 11440, 'cpu', batch_size=7)
            torch.testing.assert_close(bank['commands'], original['commands'], atol=0, rtol=0)
