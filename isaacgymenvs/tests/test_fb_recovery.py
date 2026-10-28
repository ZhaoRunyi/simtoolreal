"""Two-rank production failure arbitration and complete recovery publication."""
import copy
import datetime
import json
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from test_fb_cpr import make_agent
from test_fb_q_loss import equal_nested, fill_replay


def prepare_saved_state(agent):
    agent.run_started = time.monotonic()
    agent.vec_env.get_env_state = lambda: {}
    for name in ('observations', 'actor_state', 'rollout_command', 'command_age',
                 'reset_before', 'episode_reward', 'previous_block', 'previous_initial_state'):
        setattr(agent, name, None)


def recovery_worker(rank, root, rendezvous):
    root = Path(root)
    # Construct on CPU before init: production __init__ correctly chooses CUDA for DDP.
    torch.manual_seed(651 + rank)
    agent = make_agent(root / 'shared')
    fill_replay(agent)
    agent.global_rank = rank
    agent.world_size = 2
    agent.options.update(q_loss_coefficient=.1, full_recovery_interval=4096)
    prepare_saved_state(agent)
    distributed.init_process_group('gloo', init_method='file://' + rendezvous,
                                   rank=rank, world_size=2, timeout=datetime.timedelta(seconds=35))
    try:
        before = copy.deepcopy(agent.representation_optimizer.state_dict())
        parameters = copy.deepcopy(agent.model.forward_maps.state_dict())
        original_backward = agent.model.backward.forward
        if rank == 1:
            agent.model.backward.forward = lambda value: original_backward(value) * 0
        with pytest.raises(FloatingPointError):
            agent.update()
        equal_nested(before, agent.representation_optimizer.state_dict())
        equal_nested(parameters, agent.model.forward_maps.state_dict())
        assert agent.optimizer_step == 0
        failure_path = agent.output_dir / ('q_failure.step0.rank' + str(rank) + '.pt')
        evidence = torch.load(failure_path, weights_only=False)
        assert evidence['rank'] == rank
        assert evidence['covariance'].shape == (4, 4)
        assert (evidence['error'] is not None) == (rank == 1)
        agent.model.backward.forward = original_backward
        distributed.barrier()
        # Also force finite B but a nonfinite target only on one rank.
        forward = torch.randn(2, 8, 4, requires_grad=True)
        backward = torch.randn(8, 4)
        target = torch.randn(2, 8, 4)
        if rank == 1:
            target.fill_(float('inf'))
        agent.optimizer_step = 1
        with pytest.raises(FloatingPointError):
            agent._scalar_fb_loss(forward, backward, target, torch.randn(8, 4),
                                  torch.ones(8, 1), torch.ones(8, dtype=torch.bool), False)
        # Save is tested with a legitimate complete boundary (not the failed agent).
        fresh = make_agent(root / ('clean' + str(rank)))
        fill_replay(fresh)
        fresh.global_rank = rank
        fresh.world_size = 2
        fresh.options['full_recovery_interval'] = 4096
        fresh.frame = 378
        fresh.epoch_num = 63
        prepare_saved_state(fresh)
        path = root / 'recovery-0.pt'
        rng = torch.get_rng_state()
        fresh.save(path, replay=True)
        assert torch.equal(rng, torch.get_rng_state())
        manifest = fresh._validate_recovery(path)
        assert manifest['optimizer_step'] == 0
        assert len(manifest['files']) == 3
        saved = torch.load(root / ('recovery-0.rank' + str(rank) + '.replay.pt'), weights_only=False)
        assert saved['recovery_identity'] == manifest['identity']
        assert saved['rank'] == rank
        equal_nested(saved['rank_state']['observation_normalizer'], fresh.observation_normalizer.state_dict())
        original_manifest = path.with_suffix('.complete.json').read_bytes()
        with pytest.raises(RuntimeError, match='already exists'):
            fresh.save(path, replay=True)
        assert path.with_suffix('.complete.json').read_bytes() == original_manifest
        fresh.restore(path)
        fresh.optimizer_step = rank
        with pytest.raises(RuntimeError, match='counters disagree'):
            fresh.save(root / 'recovery-mismatch.pt', replay=True)
        assert not (root / 'recovery-mismatch.complete.json').exists()
        fresh.optimizer_step = 0
        original_save = fresh._save_files
        if rank == 1:
            def fail_write(*args, **kwargs):
                raise OSError('injected replay write failure')
            fresh._save_files = fail_write
        with pytest.raises(RuntimeError, match='injected replay write failure'):
            fresh.save(root / 'recovery-partial.pt', replay=True)
        assert not (root / 'recovery-partial.complete.json').exists()
        fresh._save_files = original_save
        if rank == 1:
            shard = root / 'recovery-0.rank1.replay.pt'
            with shard.open('r+b') as stream:
                stream.seek(-1, 2)
                original = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([original[0] ^ 1]))
        distributed.barrier()
        with pytest.raises(RuntimeError, match='checksum mismatch'):
            fresh.restore(path)
        (root / ('rank' + str(rank) + '.json')).write_text(json.dumps({'status': 'passed', 'rank': rank}))
    finally:
        distributed.destroy_process_group()


def test_two_rank_failure_and_recovery_publication(tmp_path):
    context = multiprocessing.spawn(recovery_worker,
        args=(str(tmp_path), str(tmp_path / 'gloo-init')), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('Two-rank protocol deadlocked')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(json.loads((tmp_path / ('rank' + str(rank) + '.json')).read_text())['status'] == 'passed'
               for rank in range(2))


def test_production_loop_recovery_boundaries_and_exact_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rl_games.algos_torch.fb_agent import FBAgent
    snapshots = []
    agent = SimpleNamespace(
        options=dict(full_recovery_interval=4096, warmup_blocks=64, updates_per_iteration=16,
                     target_frames=19120128, log_interval=16, speed_gate_seconds=900,
                     max_seconds=21600, stop_on_slow=False),
        config=dict(max_epochs=1000, save_frequency=1000),
        epoch_num=0, optimizer_step=0, frame=0, last_checkpoint=None,
        replay=SimpleNamespace(size=0), env_info={}, elapsed_before_resume=0,
        global_rank=0, world_size=1, device=torch.device('cpu'), writer=None,
        output_dir=tmp_path, nn_dir=tmp_path,
        actor_optimizer=SimpleNamespace(param_groups=[{'lr': 3e-4}]),
        _initialize_logging=lambda: None, _initialize_rollout=lambda: None,
        _status=lambda *args, **kwargs: None, _log=lambda metrics: metrics,
    )
    def collect():
        agent.frame += 24576
        agent.replay.size += 1
        return {}
    def update():
        agent.optimizer_step += 1
        return {}
    def save(path, replay=False):
        snapshots.append((path.name, agent.frame, agent.epoch_num, agent.optimizer_step, replay))
    agent.collect = collect
    agent.update = update
    agent.save = save
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 0)
    FBAgent.train(agent)
    assert snapshots == [
        ('recovery-0.pt', 63 * 24576, 63, 0, True),
        ('recovery-4096.pt', 319 * 24576, 319, 4096, True),
        ('recovery-8192.pt', 575 * 24576, 575, 8192, True),
        ('last.pt', 19120128, 778, 11440, True),
    ]
