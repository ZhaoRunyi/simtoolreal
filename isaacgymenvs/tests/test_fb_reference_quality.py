import copy
import pytest
import torch

from rl_games.algos_torch.fb_reference_quality import certify_episode, select_episodes, pack_reference_states, keypoint_error
from rl_games.algos_torch.fb_cpr_networks import ExpertSequences


def episode(goals=50, terminal=True):
    count=goals * 10
    poses=torch.zeros(count,13)
    poses[:,2]=.85
    poses[:,6]=1
    initial=poses.clone()
    initial[:,2]=.63
    rows=dict(object=poses,goal=poses.clone(),object_initial=initial,
              state=torch.arange(count).float()[:,None].expand(count,3).clone(),
              done=torch.zeros(count,dtype=torch.bool),reset_before=torch.zeros(count,dtype=torch.bool),
              goal_reset_before=torch.zeros(count,dtype=torch.bool),goal_success=torch.zeros(count,dtype=torch.bool),
              failure_causes=torch.zeros(count,6,dtype=torch.bool))
    rows['reset_before'][0]=True
    rows['done'][-1]=terminal
    rows['goal_reset_before'][::10]=True
    rows['goal_success'][9::10]=True
    return rows


def test_all_fifty_goals_geometry_and_concurrent_failure():
    rows=episode()
    result=certify_episode(rows,torch.zeros(4,3))
    assert result['complete'] and result['goal_count']==50 and result['valid_adjacent_pairs']==497
    assert result['lift_certified_at']==11
    rows['failure_causes'][-1,0]=True
    result=certify_episode(rows,torch.zeros(4,3))
    assert not result['complete'] and result['prefix_eligible'] and result['stop']==490


def test_prefix_requires_lift_within_last_success():
    rows=episode(3)
    result=certify_episode(rows,torch.zeros(4,3))
    assert not result['complete'] and result['prefix_eligible'] and result['stop']==30
    rows['object'][:,2]=.64
    rows['goal'][:,2]=.64
    assert not certify_episode(rows,torch.zeros(4,3))['prefix_eligible']


def test_right_censor_cannot_be_complete_and_goal_labels_are_not_certificate():
    rows=episode(50,False)
    assert not certify_episode(rows,torch.zeros(4,3))['complete']
    rows=episode()
    rows['object'][4,0]=.1
    result=certify_episode(rows,torch.zeros(4,3))
    assert not result['label_geometry_match'] and not result['complete']


def test_left_cut_and_cross_terminal_are_rejected():
    rows=episode()
    rows['reset_before'][0]=False
    with pytest.raises(ValueError): certify_episode(rows,torch.zeros(4,3))
    rows=episode()
    rows['done'][20]=True
    with pytest.raises(ValueError): certify_episode(rows,torch.zeros(4,3))


def test_geometry_rotation_and_offsets():
    obj=torch.tensor([[0.,0.,0.,0.,0.,0.,1.]])
    goal=obj.clone()
    goal[:,2]=.02
    offsets=torch.tensor([[1.,0,0],[0,1,0],[-1,0,0],[0,-1,0]])
    torch.testing.assert_close(keypoint_error(obj,goal,offsets),torch.tensor([.02],dtype=torch.float64))
    goal[:,2]=0
    goal[:,5]=1
    goal[:,6]=0
    torch.testing.assert_close(keypoint_error(obj,goal,offsets),torch.tensor([2.],dtype=torch.float64))


def test_whole_episode_selection_never_duplicates_or_trims():
    candidates=[]
    for i,(full,count) in enumerate([(True,6),(True,6),(False,4),(False,4),(False,3)]):
        candidates.append(dict(shard=0,end_step=i,environment=i,certificate=dict(complete=full,prefix_eligible=not full,valid_adjacent_pairs=count)))
    chosen,report=select_episodes(candidates,capacity=20,complete_minimum=10,minimum=19)
    assert report['ready'] and report['complete_pairs']==12 and report['valid_adjacent_pairs']==20
    assert len({row['environment'] for row in chosen})==len(chosen)==4


def test_packed_windows_preserve_boundaries_and_adjacent_pair_count():
    first=episode()
    second=episode(30)
    second['state']+=1000
    certificates=[certify_episode(row,torch.zeros(4,3)) for row in (first,second)]
    data=pack_reference_states(list(zip((first,second),certificates)))
    valid=data['valid']
    assert int((valid[1:]&valid[:-1]).sum())==sum(row['valid_adjacent_pairs'] for row in certificates)
    sequences=ExpertSequences(data['states'],valid)
    for length in (8,9,250):
        sample=sequences.sample(100,length)
        torch.testing.assert_close(sample[:,1:]-sample[:,:-1],torch.ones_like(sample[:,1:]))


def auditable_episode(goals=50):
    from rl_games.algos_torch.fb_reference_audit import physical_error
    rows = episode(goals)
    count = len(rows['done'])
    rows['step'] = torch.arange(count)
    rows['next_state'] = rows['state'] + 1
    rows['action'] = torch.zeros(count, 23)
    rows['reward_components'] = torch.zeros(count, 10)
    rows['reward'] = torch.zeros(count)
    rows['noisy_initial'] = rows['object_initial'].clone()
    rows['scale_noise'] = torch.ones(count, 3)
    rows['table'] = torch.zeros(count, 13)
    rows['table'][:, 2] = .38
    rows['table'][:, 6] = 1
    rows['native_lifted'] = torch.ones(count, dtype=torch.bool)
    rows['progress'] = (torch.arange(count) + 1) % 10
    rows['successes'] = (torch.arange(count) + 1) // 10
    rows['table_force'] = torch.zeros(count)
    rows['fingertip_distance'] = torch.zeros(count, 4)
    rows['incoming_reset'] = torch.zeros(count, dtype=torch.bool)
    unit = torch.tensor([[1., 1., 1.], [-1., -1., 1.], [1., -1., -1.], [-1., 1., -1.]])
    offsets = unit * .1 * 1.5 / 2
    rows['fixed_keypoint_error'] = torch.from_numpy(physical_error(rows['object'], rows['goal'], offsets)).float()
    rows['actual_keypoint_error'] = rows['fixed_keypoint_error'].clone()
    identity = {
        'configuration': {'env': {'fixedSize': [.1, .1, .1], 'liftingBonusThreshold': .15, 'resetWhenDropped': True}},
        'fixed_offsets': offsets[None], 'physical_offsets': offsets[None],
        'keypoint_unit_offsets': unit, 'object_scales': torch.ones(1, 3), 'object_base_size': .1,
        'with_table_force_sensor': False,
        'table_asset': {'urdf': '<robot><link><collision><geometry><box size=".475 .4 .3"/></geometry></collision></link></robot>'},
    }
    identity['selected_environments'] = torch.tensor([0])
    identity['asset_manifest'] = [identity['table_asset']]
    return rows, identity


def test_independent_audit_recomputes_pose_and_timing(monkeypatch):
    from rl_games.algos_torch.fb_reference_audit import audit_episode
    import rl_games.algos_torch.fb_reference_quality as collector
    rows, identity = auditable_episode()
    monkeypatch.setattr(collector, 'certify_episode', lambda *args: pytest.fail('Audit reused collector'))
    monkeypatch.setattr(collector, 'keypoint_error', lambda *args: pytest.fail('Audit reused geometry'))
    result = audit_episode(rows, identity, 0)
    assert result['complete'] and result['valid_adjacent_pairs'] == 497
    assert result['goal_activation_steps'] == list(range(0, 500, 10))
    rows['object'][12, 0] += .1
    with pytest.raises(ValueError, match='geometry disagrees'):
        audit_episode(rows, identity, 0)


@pytest.mark.parametrize('field', ['goal_reset_before', 'progress', 'successes', 'done', 'native_lifted', 'incoming_reset'])
def test_independent_audit_rejects_corrupt_native_events(field):
    from rl_games.algos_torch.fb_reference_audit import audit_episode
    rows, identity = auditable_episode()
    rows[field][20] = not bool(rows[field][20])
    with pytest.raises(ValueError):
        audit_episode(rows, identity, 0)


def test_repeated_success_without_goal_switch_cannot_forge_fifty_goals():
    rows = episode()
    rows['goal_reset_before'][10] = False
    with pytest.raises(ValueError, match='Goal reset timing'):
        certify_episode(rows, torch.zeros(4, 3))


def test_duplicate_source_episode_is_not_extra_budget():
    row = dict(shard=0, end_step=100, environment=0, episode=1,
               certificate=dict(complete=True, prefix_eligible=False, valid_adjacent_pairs=497))
    with pytest.raises(ValueError, match='Duplicate'):
        select_episodes([row, copy.deepcopy(row)])


def test_stored_actual_size_geometry_is_independent_of_fixed_success_mask():
    from rl_games.algos_torch.fb_reference_audit import physical_error
    rows, identity = auditable_episode()
    goal = rows['goal'].clone()
    goal[:, 5] = .1
    goal[:, 6] = (1 - .1 ** 2) ** .5
    fixed_error = physical_error(rows['object'], goal, identity['fixed_offsets'][0])
    actual_error = physical_error(rows['object'], goal, identity['fixed_offsets'][0] * 10)
    assert (actual_error > fixed_error * 9.99).all()


def test_quality_loader_stops_before_unapproved_training(tmp_path):
    from types import SimpleNamespace
    from rl_games.algos_torch.fb_cpr_agent import FBcprAgent
    environment = SimpleNamespace(reference_reset_bank=None)
    agent = SimpleNamespace(output_dir=tmp_path, global_rank=0,
                            options={'quality_reference_path': str(tmp_path / 'quality.pt')},
                            vec_env=SimpleNamespace(env=environment))
    with pytest.raises(ValueError, match='training-precondition approval'):
        FBcprAgent._prepare_experts(agent)
    assert environment.reference_reset_bank is None


def test_default_cache_loading_preserves_rng_and_environment_curriculum(tmp_path):
    import random
    import numpy as np
    from types import SimpleNamespace
    from rl_games.algos_torch.fb_cpr_agent import FBcprAgent
    bank = {'object': torch.arange(13).float()[None]}
    torch.save({'states': torch.zeros(12, 1, 3), 'valid': torch.ones(12, 1, dtype=torch.bool),
                'resets': bank, 'report': {'original': True}}, tmp_path / 'expert.rank0.pt')
    environment = SimpleNamespace(reference_reset_bank=None, frame_since_restart=0,
                                  success_tolerance=.075, last_curriculum_update=0,
                                  _tyler_curriculum_scale=0.0)
    agent = SimpleNamespace(output_dir=tmp_path, global_rank=0, options={}, device='cpu',
                            vec_env=SimpleNamespace(env=environment))
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    FBcprAgent._prepare_experts(agent)
    assert python_state == random.getstate()
    assert all(np.array_equal(first, second) for first, second in zip(numpy_state, np.random.get_state()))
    assert torch.equal(torch_state, torch.get_rng_state())
    assert environment.frame_since_restart == environment.last_curriculum_update == 0
    assert environment.success_tolerance == .075 and environment._tyler_curriculum_scale == 0.0
    torch.testing.assert_close(environment.reference_reset_bank['object'], bank['object'])
    assert not agent.experts.starts


def test_lift_achieved_only_after_last_success_does_not_qualify_prefix():
    rows = episode(4)
    rows['object'][:20, 2] = .64
    rows['goal'][:20, 2] = .64
    rows['goal'][20:, 2] = .95
    rows['goal_success'][20:] = False
    rows['goal_reset_before'][30] = False
    result = certify_episode(rows, torch.zeros(4, 3))
    assert result['lift_certified_at'] == 31
    assert result['goal_success_steps'] == [9, 19]
    assert not result['prefix_eligible'] and result['stop'] == 0


def test_conservative_physical_clearance_uses_collision_extents_and_orientation():
    import numpy as np
    from rl_games.algos_torch.fb_reference_audit import conservative_object_bottom
    poses = np.array([[0., 0., 1., 0., 0., 0., 1.],
                      [0., 0., 1., 0., 2 ** -.5, 0., 2 ** -.5]])
    asset = {'urdf': '<robot><link><collision><geometry><box size=".4 .2 .1"/></geometry></collision></link></robot>'}
    np.testing.assert_allclose(conservative_object_bottom(poses, asset), [.95, .8])


def test_hybrid_configuration_resolves_nested_omegaconf_before_recording():
    import json
    from omegaconf import OmegaConf
    from rl_games.algos_torch.fb_reference_quality import configuration_container
    source = OmegaConf.create({'size': 3, 'nested': ['${size}', {'range': [0.9, 1.1]}]})
    hybrid = {'env': {'values': source.nested, 'ordinary_tuple': (1, 2)}}
    result = configuration_container(hybrid)
    assert result == {'env': {'values': [3, {'range': [.9, 1.1]}], 'ordinary_tuple': [1, 2]}}
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_one_step_right_censored_episode_is_archived_without_aborting_collection():
    rows = {name: value[:1].clone() for name, value in episode().items()}
    result = certify_episode(rows, torch.zeros(4, 3))
    assert result['right_censored'] and result['raw_transitions'] == 1
    assert not result['complete'] and not result['prefix_eligible']
    assert result['valid_adjacent_pairs'] == 0
