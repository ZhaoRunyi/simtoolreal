"""v4 geometry, chronological replay, and actual learner update contracts."""

import math
import torch

from test_fb_cpr import make_agent
from rl_games.algos_torch.fb_cpr_networks import BackwardMap, TemporalReplay, sphere_log_prompt


def test_sphere_log_identity_orthogonal_antipode_and_rotation():
    base = torch.tensor([[2., 0, 0, 0]])
    torch.testing.assert_close(sphere_log_prompt(base, base), torch.zeros_like(base))
    orthogonal = torch.tensor([[0., 2, 0, 0]])
    result = sphere_log_prompt(base, orthogonal)
    torch.testing.assert_close(result, torch.tensor([[0., math.pi, 0, 0]]))
    opposite = sphere_log_prompt(base, -base)
    torch.testing.assert_close(opposite.norm(dim=-1), torch.tensor([2 * math.pi]))
    assert (base * opposite).sum() == 0
    rotation = torch.linalg.qr(torch.randn(4, 4))[0]
    torch.testing.assert_close(sphere_log_prompt(base @ rotation, orthogonal @ rotation), result @ rotation,
                               atol=2e-6, rtol=2e-6)


def test_lifted_input_selects_158_and_ignores_other_history():
    backward = BackwardMap(153, 4, list(range(152)) + [158])
    state = torch.randn(2, 162)
    changed = state.clone()
    changed[:, [152, 153, 154, 155, 156, 157, 159, 160, 161]] += 50
    torch.testing.assert_close(backward(state), backward(changed), rtol=0, atol=0)
    changed[:, 158] += 3
    assert not torch.equal(backward(state), backward(changed))


def test_temporal_windows_survive_wrap_and_exclude_reset():
    replay = TemporalReplay(36, 3, 1, 'cpu')
    for step in range(30):
        state = torch.tensor([step, 100 + step, 200 + step]).reshape(1, 3, 1).float()
        valid = torch.ones(1, 3, 1, dtype=torch.bool)
        valid[:, 1] = step != 24
        discount = torch.full((1, 3, 1), .98)
        discount[:, 1] = 0 if step == 23 else .98
        replay.add({'state': state, 'next_state': state + 1, 'valid': valid, 'discount': discount})
    window = replay.sample_window(100, 8)
    assert window is not None
    torch.testing.assert_close(window['local_state'], window['state'] + 1)
    torch.testing.assert_close(window['goal_state'], window['state'] + 8)
    assert (window['state'] % 100 >= 18).all()
    assert (window['goal_state'] % 100 <= 30).all()
    assert not ((window['state'] >= 100) & (window['state'] < 200)).any()
    restored = TemporalReplay(1, 1, 1, 'cpu')
    restored.load_state_dict(replay.state_dict())
    sample = restored.sample_window(50, 8)
    torch.testing.assert_close(sample['goal_state'], sample['state'] + 8)


def test_local_prompt_and_logo_update_production_actor(tmp_path):
    agent = make_agent(tmp_path)
    agent.model.local_prompt = True
    agent.options.update(logo_coefficient=.005, logo_horizon=8,
                         logo_local_weight=4 / 7, logo_goal_weight=3 / 7)
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
    before = {name: parameter.clone() for name, parameter in agent.model.actor.named_parameters()}
    metrics = agent.update()
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics['logo/available'] == 1
    assert 'loss/logo_scaled' in metrics
    assert any(not torch.equal(parameter, before[name]) for name, parameter in agent.model.actor.named_parameters())
    assert all(parameter.grad is None for parameter in agent.target.parameters())


def test_diagnostics_real_checkpoint_replay_to_report(tmp_path, monkeypatch):
    import json
    import sys
    from rl_games.algos_torch import fb_diagnostics
    from rl_games.algos_torch.fb_cpr_networks import CPRModel, BatchNormalizer, TransitionReplay
    options = dict(latent_size=4, residual_width=16, residual_layers=2, history_steps=4,
                   backward_state_size=153, backward_state_indices=list(range(152)) + [158],
                   local_prompt=True, style_coefficient=.05, auxiliary_coefficient=.02)
    model = CPRModel(140, 162, 29, options, {})
    folder = tmp_path / 'run/nn'
    folder.mkdir(parents=True)
    checkpoint = dict(options=options, schema={'observation': 140, 'state': 162, 'action': 29},
                      frame=24, optimizer_step=16, model=model.state_dict(), target=model.state_dict(),
                      training_world_size=2)
    for name, size in [('observation', 140), ('state', 162), ('action', 29)]:
        checkpoint[name + '_normalizer'] = BatchNormalizer(size).state_dict()
    torch.save(checkpoint, folder / 'last.pt')
    (folder.parent / 'status.json').write_text(json.dumps(dict(state='trained', environment_steps=24, optimizer_step=16)))
    for rank in range(2):
        replay = TransitionReplay(12, 3, 4, 'cpu')
        replay.add({
            'observation': torch.randn(4, 3, 140), 'next_observation': torch.randn(4, 3, 140),
            'state': torch.randn(4, 3, 162), 'next_state': torch.randn(4, 3, 162),
            'history': torch.randn(4, 3, 560), 'next_history': torch.randn(4, 3, 560),
            'previous_action': torch.randn(4, 3, 29), 'action': torch.randn(4, 3, 29).tanh(),
            'command': torch.randn(4, 3, 4), 'reward': torch.randn(4, 3, 1),
            'discount': torch.ones(4, 3, 1) * .98, 'valid': torch.ones(4, 3, 1, dtype=torch.bool),
        })
        torch.save({'replay': replay.state_dict()}, folder / ('last.rank' + str(rank) + '.replay.pt'))
    monkeypatch.setattr(sys, 'argv', ['diagnostics', '--run', str(folder.parent), '--output', str(tmp_path / 'report'),
                                    '--device', 'cpu', '--samples', '24', '--batch-size', '8'])
    fb_diagnostics.main()
    report = json.loads((tmp_path / 'report/report.json').read_text())
    assert report['frame'] == 24
    assert len(report['sources']) == 2
    assert report['commands']['current_goal']['prompt_norm']['max'] < 1e-5
    assert report['sample_count'] == 24
    assert (tmp_path / 'report/sample.pt').exists()
    assert (tmp_path / 'report/outputs.pt').exists()
