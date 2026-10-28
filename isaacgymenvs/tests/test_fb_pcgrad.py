"""PCGrad geometry and the production CPR/LOGO actor update."""

import torch

from test_fb_cpr import make_agent
from rl_games.algos_torch.fb_actor_gradients import ActorGradientAccumulator, symmetric_projection
from rl_games.algos_torch.fb_cpr_networks import TemporalReplay


def test_projection_conflict_identity_and_auxiliary_unchanged():
    vectors = torch.tensor([[2., 0., 0.], [-1., 1., 0.], [.3, .4, .7]])
    combined, metrics = symmetric_projection(vectors)
    torch.testing.assert_close(combined, torch.tensor([1.3, 2.4, .7]))
    assert metrics['pcgrad/conflict'] == 1
    assert metrics['pcgrad/fb_projection_residual'] == 0
    assert metrics['pcgrad/style_projection_residual'] == 0


def test_zero_nonconflict_antiparallel_and_near_antiparallel():
    auxiliary = torch.tensor([.3, .7])
    for forward, style in [(torch.zeros(2), torch.ones(2)), (torch.ones(2), torch.ones(2))]:
        combined, metrics = symmetric_projection(torch.stack((forward, style, auxiliary)))
        torch.testing.assert_close(combined, forward + style + auxiliary, rtol=0, atol=0)
        assert metrics['pcgrad/conflict'] == 0
    combined, metrics = symmetric_projection(torch.stack((torch.ones(2), -torch.ones(2), auxiliary)))
    torch.testing.assert_close(combined, auxiliary, rtol=0, atol=0)
    assert metrics['pcgrad/pair_norm_ratio_valid'] == 0
    assert 'pcgrad/pair_norm_ratio' not in metrics
    near = torch.tensor([[1., 1.], [-1., -1.+1e-6], [.3, .7]])
    result, values = symmetric_projection(near)
    assert torch.isfinite(result).all()
    assert all(torch.isfinite(value) for value in values.values())


def test_all_branch_scales_and_weights_before_projection():
    parameter = torch.nn.Parameter(torch.tensor([.5, .2]))
    accumulator = ActorGradientAccumulator([parameter], 1, True)
    expected = torch.zeros(3, 2)
    for label, weight, multiplier in [('base',1.,1.), ('local',.005*4/7,2.), ('goal',.005*3/7,3.)]:
        forward = multiplier * parameter.sum().expand(4)
        style = (-parameter[0] + 2*parameter[1]).expand(4)
        auxiliary = parameter.square().sum().expand(4)
        scale = forward.detach().abs().mean()
        expected[0] += -weight * multiplier * torch.ones(2)
        expected[1] += -weight * scale * .05 * torch.tensor([-1.,2.])
        expected[2] += -weight * scale * .02 * 2*parameter.detach()
        accumulator.add((forward,style,auxiliary),(.05,.02),weight,label)
    torch.testing.assert_close(accumulator.vectors,expected)
    accumulator.assign()
    torch.testing.assert_close(parameter.grad,symmetric_projection(expected)[0])


def fill_replay(agent):
    agent.options.update(logo_coefficient=.005, logo_horizon=8, logo_local_weight=4/7,
                         logo_goal_weight=3/7, actor_gradient_probe_every=1024)
    agent.model.local_prompt = True
    agent.replay = TemporalReplay(60,3,1,'cpu')
    for step in range(16):
        state=torch.randn(1,3,20)
        agent.replay.add({'observation':torch.randn(1,3,16),'next_observation':torch.randn(1,3,16),
                         'state':state,'next_state':state+.1,'history':torch.randn(1,3,64),
                         'next_history':torch.randn(1,3,64),'previous_action':torch.randn(1,3,3),
                         'action':torch.randn(1,3,3).tanh(),'command':torch.randn(1,3,4),
                         'reward':torch.randn(1,3,1),'auxiliary_reward':-torch.rand(1,3,1),
                         'discount':torch.ones(1,3,1)*.98,'valid':torch.ones(1,3,1,dtype=torch.bool),
                         'command_source':torch.tensor([[[0],[1],[2]]]),
                         'actor_diagnostic':torch.tensor([[[.01,.0,.0,0,0],[.1,.1,.1,1,0],[.2,.2,.2,1,0]]])})


def test_projected_production_update_full_logo_and_adam(tmp_path):
    torch.manual_seed(33)
    agent=make_agent(tmp_path)
    fill_replay(agent)
    agent.options['actor_gradient_projection']=True
    before=[parameter.detach().clone() for parameter in agent.model.actor.parameters()]
    metrics=agent.update()
    assert metrics['logo/available']==1
    assert metrics['pcgrad/updates']==1
    assert 'pcgrad_branch/local/fb_norm' in metrics
    assert 'pcgrad_branch/goal/style_norm' in metrics
    assert 'pcgrad_full/fb_update_dot' in metrics
    assert metrics['pcgrad_strata/base/physical_low/count']>=0
    assert all(torch.isfinite(value) for value in metrics.values())
    assert all(state['step']==1 for state in agent.actor_optimizer.state.values())
    assert any(not torch.equal(old,new) for old,new in zip(before,agent.model.actor.parameters()))


def test_disabled_flag_matches_original_path_and_rng(tmp_path):
    torch.manual_seed(31)
    first=make_agent(tmp_path/'first')
    fill_replay(first)
    torch.manual_seed(31)
    second=make_agent(tmp_path/'second')
    fill_replay(second)
    second.options['actor_gradient_projection']=False
    state=torch.get_rng_state()
    first.update()
    after=torch.get_rng_state()
    torch.set_rng_state(state)
    second.update()
    assert torch.equal(after,torch.get_rng_state())
    for name,value in first.model.state_dict().items():
        torch.testing.assert_close(value,second.model.state_dict()[name],rtol=0,atol=0)


def assert_tree_equal(first, second):
    import numpy as np
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            assert_tree_equal(first[name], second[name])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_tree_equal(left, right)
    else:
        assert first == second


def test_disabled_update_matches_actual_git_baseline_all_state(tmp_path):
    import copy
    import random
    import subprocess
    import types
    import numpy as np
    from pathlib import Path
    repository = Path(__file__).parents[2]
    modules = []
    for filename in ('fb_agent', 'fb_cpr_agent'):
        source = subprocess.check_output([
            'git', 'show', '3bd73c66ae78c8ce5334bcf37bcc46cdb0e9eec2:rl_games/rl_games/algos_torch/' + filename + '.py',
        ], cwd=repository).decode()
        module = types.ModuleType('historical_' + filename)
        exec(compile(source, 'historical_' + filename + '.py', 'exec'), module.__dict__)
        modules.append(module)
    modules[1].FBAgent = modules[0].FBAgent
    # Re-execute with the historical parent class, not the current import.
    source = source.replace('from rl_games.algos_torch.fb_agent import FBAgent', '')
    exec(compile(source, 'historical_fb_cpr_agent.py', 'exec'), modules[1].__dict__)
    torch.manual_seed(981)
    current = make_agent(tmp_path / 'current')
    fill_replay(current)
    current.options['actor_gradient_projection'] = False
    historical = modules[1].FBcprAgent.__new__(modules[1].FBcprAgent)
    historical.__dict__ = copy.deepcopy(current.__dict__)
    for _ in range(3):
        rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        expected_metrics = historical.update()
        expected_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        torch.set_rng_state(rng[0])
        np.random.set_state(rng[1])
        random.setstate(rng[2])
        actual_metrics = current.update()
        assert_tree_equal(expected_rng, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
        assert_tree_equal(expected_metrics, actual_metrics)
        for name in ('model', 'target', 'actor_optimizer', 'representation_optimizer',
                     'auxiliary_optimizer', 'style_optimizer', 'discriminator_optimizer',
                     'observation_normalizer', 'state_normalizer', 'action_normalizer', 'reward_normalizer'):
            assert_tree_equal(getattr(historical, name).state_dict(), getattr(current, name).state_dict())
        for name in ('z_size', 'z_position', 'z_source_buffer', 'task_command', 'optimizer_step'):
            assert_tree_equal(getattr(historical, name), getattr(current, name))
        assert_tree_equal(historical.z_buffer[:historical.z_size], current.z_buffer[:current.z_size])


def test_projection_probe_does_not_change_rng_normalizers_or_adam(tmp_path):
    import copy
    torch.manual_seed(142)
    plain = make_agent(tmp_path / 'plain')
    fill_replay(plain)
    plain.options['actor_gradient_projection'] = True
    monitored = copy.deepcopy(plain)
    plain.options['actor_gradient_probe_every'] = 0
    rng = torch.get_rng_state()
    plain.update()
    final_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    monitored.update()
    assert_tree_equal(final_rng, torch.get_rng_state())
    for name in ('model', 'target', 'actor_optimizer', 'representation_optimizer',
                 'auxiliary_optimizer', 'style_optimizer', 'discriminator_optimizer',
                 'observation_normalizer', 'state_normalizer', 'action_normalizer', 'reward_normalizer'):
        assert_tree_equal(getattr(plain, name).state_dict(), getattr(monitored, name).state_dict())


def test_projection_checkpoint_restores_all_optimizers_and_statistics(tmp_path):
    import time
    torch.manual_seed(218)
    original = make_agent(tmp_path / 'original')
    fill_replay(original)
    original.options['actor_gradient_projection'] = True
    original.update()
    original.run_started = time.monotonic()
    checkpoint = tmp_path / 'checkpoint.pt'
    original.save(checkpoint)
    restored = make_agent(tmp_path / 'restored')
    restored.restore(checkpoint)
    for name in ('model', 'target', 'actor_optimizer', 'representation_optimizer',
                 'auxiliary_optimizer', 'style_optimizer', 'discriminator_optimizer',
                 'observation_normalizer', 'state_normalizer', 'action_normalizer', 'reward_normalizer'):
        assert_tree_equal(getattr(original, name).state_dict(), getattr(restored, name).state_dict())
    for name in ('projection_updates', 'projection_conflicts', 'projection_sums'):
        assert_tree_equal(getattr(original, name), getattr(restored, name))
    restored.replay = TemporalReplay(60, 3, 1, 'cpu')
    restored.replay.load_state_dict(original.replay.state_dict())
    rng = torch.get_rng_state()
    expected = original.replay.sample_window(8, 8)
    torch.set_rng_state(rng)
    assert_tree_equal(expected, restored.replay.sample_window(8, 8))


def test_legacy_replay_optional_metadata_is_unknown_and_rng_neutral(tmp_path):
    agent = make_agent(tmp_path)
    fill_replay(agent)
    legacy = agent.replay.state_dict()
    for name in ('command_source', 'actor_diagnostic'):
        legacy['storage'].pop(name)
    replay = TemporalReplay(60, 3, 1, 'cpu')
    replay.load_state_dict(legacy)
    block = {name: value[0].clone() for name, value in legacy['storage'].items()}
    block['command_source'] = torch.ones(1, 3, 1, dtype=torch.long)
    block['actor_diagnostic'] = torch.full((1, 3, 5), .15001)
    block['actor_diagnostic_valid'] = torch.ones(1, 3, 1, dtype=torch.bool)
    rng = torch.get_rng_state()
    replay.add(block)
    assert_tree_equal(rng, torch.get_rng_state())
    assert (replay.storage['command_source'][:16] == -1).all()
    assert not replay.storage['actor_diagnostic_valid'][:16].any()
    assert replay.storage['actor_diagnostic'].dtype == torch.float32
    old = replay._gather(torch.tensor([0]), torch.tensor([0]))
    new = replay._gather(torch.tensor([16]), torch.tensor([0]))
    assert agent._gradient_strata(old)['physical_unknown'].all()
    assert agent._gradient_strata(old)['behavior_unknown'].all()
    assert not agent._gradient_strata(new)['physical_unknown'].any()
    assert list(agent._gradient_strata(old)) == list(agent._gradient_strata(new))


def test_production_logo_missing_nonfinite_save_and_source_semantics(tmp_path):
    import pytest
    torch.manual_seed(202)
    agent = make_agent(tmp_path)
    fill_replay(agent)
    agent.replay.size = 4
    agent.options['actor_gradient_projection'] = True
    metrics = agent.update()
    assert metrics['logo/available'] == 0
    assert 'pcgrad_branch/local/fb_norm' not in metrics
    batch = {'state': torch.zeros(3, 20), 'discount': torch.tensor([[1.], [0.], [1.]])}
    masks = agent._gradient_strata(batch, torch.tensor([0, 1, 2]), torch.tensor([[True], [False], [True]]))
    assert masks['behavior_unknown'].all()
    assert masks['final_unknown'].tolist() == [False, True, False]
    assert masks['final_random'].tolist() == [True, False, False]
    assert masks['candidate_replay_goal'].tolist() == [False, True, False]
    original = agent._actor_objective
    def invalid_objective(*args):
        loss, forward, style, auxiliary = original(*args)
        return loss, forward * float('nan'), style, auxiliary
    agent._actor_objective = invalid_objective
    with pytest.raises(FloatingPointError):
        agent.update()
    payload = torch.load(agent.nn_dir / 'nonfinite.rank0.pt')
    assert payload['resumable'] is False
    assert payload['optimizer_step'] == 1
    assert 'replay' in payload and 'rng' in payload and 'actor_optimizer' in payload


def _two_rank_worker(rank, rendezvous, directory):
    import datetime
    from pathlib import Path
    import torch.distributed as distributed
    from rl_games.algos_torch.fb_actor_gradients import gradient_vector
    torch.set_num_threads(1)
    distributed.init_process_group('gloo', init_method='file://' + rendezvous,
                                  rank=rank, world_size=2, timeout=datetime.timedelta(seconds=90))
    try:
        parameter = torch.nn.Parameter(torch.tensor([.5, .2]))
        accumulator = ActorGradientAccumulator([parameter], 2, True)
        expected = torch.zeros(3, 2)
        for label, weight, multiplier in [('base', 1., 1.), ('local', .005 * 4 / 7, 2.),
                                          ('goal', .005 * 3 / 7, 3.)]:
            forward_direction = torch.tensor([2., 0.]) if rank == 0 else torch.tensor([0., 2.])
            style_direction = torch.tensor([-1., 1.]) if rank == 0 else torch.tensor([3., -1.])
            forward = ((10. + rank) * multiplier - parameter.dot(forward_direction)).expand(4)
            style = (-parameter.dot(style_direction)).expand(4)
            auxiliary = parameter.square().sum().expand(4)
            scale = forward.detach().abs().mean()
            expected[0] += weight * forward_direction
            expected[1] += weight * scale * .05 * style_direction
            expected[2] -= weight * scale * .02 * 2 * parameter.detach()
            accumulator.add((forward, style, auxiliary), (.05, .02), weight, label,
                            {'rank_zero_only': torch.full((4,), rank == 0, dtype=torch.bool)})
        local_projection = symmetric_projection(expected)[0]
        distributed.all_reduce(local_projection)
        local_projection /= 2
        distributed.all_reduce(expected)
        expected /= 2
        accumulator.assign()
        torch.testing.assert_close(parameter.grad, symmetric_projection(expected)[0], rtol=2e-6, atol=1e-7)
        assert not torch.allclose(parameter.grad, local_projection)
        assert accumulator.metrics['pcgrad_strata/base/rank_zero_only/count'] == 4

        torch.manual_seed(145)
        agent = make_agent(Path(directory) / ('rank' + str(rank)))
        # Constructor stays CPU; production collectives are enabled afterwards.
        agent.global_rank = rank
        agent.world_size = 2
        torch.manual_seed(146 + rank)
        fill_replay(agent)
        if rank == 0:
            for name in ('command_source', 'actor_diagnostic'):
                agent.replay.storage.pop(name)
        agent.options['actor_gradient_projection'] = True
        branch_vectors = []
        original_objective = agent._actor_objective
        def observed_objective(*args):
            result = original_objective(*args)
            _, forward, style, auxiliary = result
            scale = forward.detach().abs().mean()
            losses = (-forward.mean(), -scale * .05 * style.mean(), -scale * .02 * auxiliary.mean())
            branch_vectors.append(torch.stack([gradient_vector(loss, tuple(agent.model.actor.parameters()))
                                              for loss in losses]))
            return result
        agent._actor_objective = observed_objective
        original_step = agent._step_accumulated_gradients
        captured = []
        def observed_step(optimizer, parameters, synchronize=True):
            if optimizer is agent.actor_optimizer:
                assert synchronize is False
                captured.append(torch.cat([parameter.grad.flatten() for parameter in agent.model.actor.parameters()]))
            return original_step(optimizer, parameters, synchronize=synchronize)
        agent._step_accumulated_gradients = observed_step
        metrics = agent.update()
        assert len(branch_vectors) == 3
        expected = branch_vectors[0] + .005 * (4 / 7 * branch_vectors[1] + 3 / 7 * branch_vectors[2])
        distributed.all_reduce(expected)
        expected /= 2
        forward, style, auxiliary = expected.double()
        inner = forward.dot(style)
        if inner < 0 and forward.square().sum() > 0 and style.square().sum() > 0:
            projected_forward = forward - inner / style.square().sum() * style
            projected_style = style - inner / forward.square().sum() * forward
        else:
            projected_forward, projected_style = forward, style
        expected_combined = projected_forward.float() + projected_style.float() + auxiliary.float()
        torch.testing.assert_close(captured[0], expected_combined, rtol=2e-5, atol=2e-6)
        assert all(state['step'] == 1 for state in agent.actor_optimizer.state.values())
        assert metrics['pcgrad/updates'] == 1
        assert metrics['pcgrad_strata/base/behavior_unknown/count'] == 8
        for value in agent.model.state_dict().values():
            original = value.clone()
            distributed.broadcast(value, 0)
            torch.testing.assert_close(original, value, rtol=0, atol=0)
        branch_vectors.clear()
        if rank == 0:
            agent.replay.size = 4
        metrics = agent.update()
        assert metrics['logo/available'] == 0
        assert len(branch_vectors) == 1
        assert all(state['step'] == 2 for state in agent.actor_optimizer.state.values())

        agent.options['actor_gradient_probe_every'] = 0
        def invalid_objective(*args):
            result = original_objective(*args)
            loss, forward, style, auxiliary = result
            if rank == 1:
                style = style * float('nan')
            return loss, forward, style, auxiliary
        agent._actor_objective = invalid_objective
        failed = False
        try:
            agent.update()
        except FloatingPointError:
            failed = True
        assert failed
        snapshot = torch.load(agent.nn_dir / ('nonfinite.rank' + str(rank) + '.pt'))
        assert snapshot['resumable'] is False and snapshot['rank'] == rank
        assert snapshot['optimizer_step'] == 2
        assert all(state['step'] == 2 for state in agent.actor_optimizer.state.values())
    finally:
        distributed.destroy_process_group()


def test_two_rank_global_projection_production_and_coordinated_failure(tmp_path):
    import torch.multiprocessing as multiprocessing
    multiprocessing.spawn(_two_rank_worker,
                          args=(str(tmp_path / 'gloo_store'), str(tmp_path)), nprocs=2, join=True)


def test_nonconflicting_production_matches_weighted_original_without_rescaling(tmp_path):
    import copy
    import types
    torch.manual_seed(119)
    original = make_agent(tmp_path / 'original')
    fill_replay(original)
    projected = copy.deepcopy(original)
    projected.options['actor_gradient_projection'] = True
    def aligned_objective(self, *args):
        _, forward, _, _ = type(self)._actor_objective(self, *args)
        scale = forward.detach().abs().mean()
        assert scale > 0
        style = forward / scale
        auxiliary = forward / scale
        loss = -forward.mean() - scale * (.05 * style.mean() + .02 * auxiliary.mean())
        return loss, forward, style, auxiliary
    original._actor_objective = types.MethodType(aligned_objective, original)
    projected._actor_objective = types.MethodType(aligned_objective, projected)
    rng = torch.get_rng_state()
    original.update()
    after_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    metrics = projected.update()
    assert metrics['pcgrad/conflict'] == 0
    torch.testing.assert_close(metrics['pcgrad/pair_norm_ratio'], torch.tensor(1., dtype=torch.float64),
                               rtol=1e-7, atol=1e-9)
    assert metrics['pcgrad/objective_backward_calls'] == 9
    assert_tree_equal(after_rng, torch.get_rng_state())
    for first, second in zip(original.model.actor.parameters(), projected.model.actor.parameters()):
        torch.testing.assert_close(first.grad, second.grad, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-6)
    for name in ('observation_normalizer', 'state_normalizer', 'action_normalizer', 'reward_normalizer',
                 'representation_optimizer', 'auxiliary_optimizer', 'style_optimizer', 'discriminator_optimizer'):
        assert_tree_equal(getattr(original, name).state_dict(), getattr(projected, name).state_dict())
    original_state = original.actor_optimizer.state_dict()['state']
    projected_state = projected.actor_optimizer.state_dict()['state']
    for index in original_state:
        assert original_state[index]['step'] == projected_state[index]['step'] == 1
        for name in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(original_state[index][name], projected_state[index][name],
                                       rtol=2e-5, atol=2e-7)


def test_sparse_probe_does_not_overwrite_current_counters_or_reuse_invalid_ratios(tmp_path, monkeypatch):
    from rl_games.algos_torch.fb_agent import FBAgent
    agent = make_agent(tmp_path)
    fill_replay(agent)
    agent.options['actor_gradient_projection'] = True
    metrics = agent.update()
    metrics.update(agent.update())
    metrics['pcgrad/pair_norm_ratio_valid'] = torch.tensor(0.)
    metrics['pcgrad/pair_norm_ratio'] = torch.tensor(7.)
    monkeypatch.setattr(FBAgent, '_log', lambda self, values: values)
    logged = agent._log(metrics)
    assert logged['pcgrad/updates'] == 2
    assert logged['pcgrad/optimizer_step'] == 2
    assert logged['pcgrad_probe/updates'] == 1
    assert logged['pcgrad_full/probe_optimizer_step'] == 1
    assert 'pcgrad/pair_norm_ratio' not in logged


def test_nonfinite_original_aggregate_loss_fails_before_actor_step(tmp_path):
    import pytest
    agent = make_agent(tmp_path)
    fill_replay(agent)
    agent.options['actor_gradient_projection'] = True
    original_objective = agent._actor_objective
    def invalid_aggregate(*args):
        loss, *values = original_objective(*args)
        return (loss * float('inf'), *values)
    agent._actor_objective = invalid_aggregate
    with pytest.raises(FloatingPointError):
        agent.update()
    assert not agent.actor_optimizer.state
    assert (agent.nn_dir / 'nonfinite.rank0.pt').exists()
