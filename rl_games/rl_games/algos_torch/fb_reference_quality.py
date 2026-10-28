"""PPO reference certificates and whole-episode sampling for quality-controlled FB.

Simulation collection uses the unchanged PPO/environment path. Certification is
CPU float64 geometry on stored float32 poses, independent of reward success labels.
"""
from pathlib import Path
import json
import hashlib
import uuid
import xml.etree.ElementTree as ET

import torch


def configuration_container(value):
    """Resolve hybrid dict/ListConfig trees before taking any rollout steps."""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {name: configuration_container(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [configuration_container(item) for item in value]
    return value


def asset_manifest(path):
    """Retain generated geometry before its temporary asset directory disappears."""
    path = Path(path)
    payload = path.read_bytes()
    meshes = []
    for mesh in ET.fromstring(payload).iter('mesh'):
        filename = mesh.attrib['filename']
        source = path.parent / filename
        content = source.read_bytes()
        meshes.append({'filename': filename, 'sha256': hashlib.sha256(content).hexdigest(),
                       'content': content})
    return {'name': path.name, 'sha256': hashlib.sha256(payload).hexdigest(),
            'urdf': payload.decode(), 'meshes': meshes}


def keypoint_error(object_pose, goal_pose, offsets):
    """Four-point max Euclidean error; quaternions are xyzw."""
    def transform(pose):
        position, quaternion = pose.double()[..., :3], pose.double()[..., 3:7]
        vectors = offsets.double().expand(*position.shape[:-1], -1, -1)
        xyz = quaternion[..., None, :3].expand_as(vectors)
        rotated = vectors + 2 * torch.cross(xyz, torch.cross(xyz, vectors, dim=-1)
                                           + quaternion[..., None, 3:4] * vectors, dim=-1)
        return rotated + position[..., None, :]
    return (transform(object_pose) - transform(goal_pose)).norm(dim=-1).amax(-1)


def certify_episode(rows, fixed_offsets, required_goals=50, success_steps=10):
    """Reject left/right truncation, nonconsecutive goals and concurrent failure."""
    count = len(rows['done'])
    if count < 1 or not bool(rows['reset_before'][0]) or rows['reset_before'][1:].any():
        raise ValueError('Certificate must contain one full physical episode start')
    if count > required_goals * 599:
        return {'complete': False, 'prefix_eligible': False, 'stop': 0,
                'raw_transitions': count, 'valid_state_rows': 0, 'valid_adjacent_pairs': 0,
                'oversized_episode': True, 'right_censored': not bool(rows['done'][-1])}
    if not torch.equal(rows['goal_reset_before'][1:].bool(), rows['goal_success'][:-1].bool()):
        raise ValueError('Goal reset timing differs from previous native success')
    unchanged = ~rows['goal_reset_before'][1:].bool()
    if not torch.equal(rows['goal'][1:][unchanged], rows['goal'][:-1][unchanged]):
        raise ValueError('Effective goal changed without native goal activation')
    for name in ('object', 'goal', 'object_initial'):
        if rows[name].dtype != torch.float32 or not torch.isfinite(rows[name]).all():
            raise ValueError('Certificate requires finite original float32 poses: ' + name)
    error = keypoint_error(rows['object'], rows['goal'], fixed_offsets)
    near = error <= .015
    height = rows['object'][:, 2].double() - rows['object_initial'][:, 2].double()
    goal_success = torch.zeros(count, dtype=torch.bool)
    near_steps = 0
    lift_steps = 0
    lift_at = None
    goal_count = 0
    goal_events = []
    failure = rows['failure_causes'].bool().any(-1)
    for index in range(count):
        # Native reset_target_pose clears near_goal_steps on the next physical
        # step after success, before the new randomly sampled goal is evaluated.
        if bool(rows['goal_reset_before'][index]):
            near_steps = 0
        near_steps = near_steps + 1 if near[index] else 0
        lift_steps = lift_steps + 1 if height[index] >= .15 else 0
        if lift_steps >= 12 and lift_at is None:
            lift_at = index
        if near_steps >= success_steps:
            goal_success[index] = True
            goal_count += 1
            goal_events.append(index)
        if index + 1 < count and bool(rows['done'][index]):
            raise ValueError('Certificate crosses a physical terminal')
    labelled_success = rows['goal_success'].bool()
    labels_match = torch.equal(labelled_success, goal_success)
    complete = (bool(rows['done'][-1]) and bool(goal_success[-1]) and goal_count == required_goals
                and labels_match and not failure.any() and lift_at is not None)
    valid_events = [index for index in goal_events if not failure[:index + 1].any()]
    last_success = valid_events[-1] if valid_events else None
    prefix_eligible = (not complete and labels_match and last_success is not None
                       and lift_at is not None and lift_at <= last_success
                       and not failure[:last_success + 1].any())
    if complete:
        stop = count
    elif prefix_eligible:
        stop = last_success + 1
    else:
        stop = 0
    valid = ~rows['reset_before'][:stop].bool() & ~rows['done'][:stop].bool()
    pairs = int((valid[1:] & valid[:-1]).sum())
    return {
        'complete': bool(complete), 'prefix_eligible': bool(prefix_eligible),
        'label_geometry_match': labels_match, 'goal_count': goal_count,
        'goal_success_steps': goal_events, 'lift_certified_at': lift_at,
        'goals': [{'activation_step': 0 if number == 0 else goal_events[number - 1] + 1,
                   'success_step': index, 'pose': rows['goal'][index].tolist(),
                   'error_m': float(error[index]), 'height_m': float(height[index])}
                  for number, index in enumerate(goal_events)],
        'stop': stop, 'raw_transitions': count, 'valid_state_rows': int(valid.sum()),
        'valid_adjacent_pairs': pairs, 'right_censored': not bool(rows['done'][-1]),
        'maximum_height_m': float(height.max()), 'maximum_goal_error_m': float(error.max()),
    }


def select_episodes(candidates, capacity=384000, complete_minimum=192000, minimum=380160):
    """Stable, whole-candidate greedy order; no duplication or quality weighting."""
    ordered = sorted(candidates, key=lambda row: (row['shard'], row['end_step'], row['environment']))
    identities = [(row['shard'], row['environment'], row.get('episode', row['end_step'])) for row in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate source episode cannot contribute independent transitions')
    chosen, total, complete_count = [], 0, 0
    for row in ordered:
        count = row['certificate']['valid_adjacent_pairs']
        if row['certificate']['complete'] and complete_count < complete_minimum and total + count <= capacity:
            chosen.append(row)
            total += count
            complete_count += count
    if complete_count >= complete_minimum:
        for row in ordered:
            count = row['certificate']['valid_adjacent_pairs']
            if row['certificate']['prefix_eligible'] and total + count <= capacity:
                chosen.append(row)
                total += count
    ready = minimum <= total <= capacity and complete_count >= complete_minimum and complete_count >= .5 * total
    return chosen, {'ready': ready, 'valid_adjacent_pairs': total,
                    'complete_pairs': complete_count, 'complete_fraction': complete_count / max(total, 1),
                    'candidate_count': len(candidates), 'selected_episodes': len(chosen)}


def pack_reference_states(episodes):
    """Retain original uniform valid-window sampler without crossing episodes."""
    states, valid, categories = [], [], []
    for rows, certificate in episodes:
        stop = certificate['stop']
        state = rows['state'][:stop]
        mask = ~rows['reset_before'][:stop].bool() & ~rows['done'][:stop].bool()
        states.extend((state, torch.zeros_like(state[:1])))
        valid.extend((mask, torch.zeros(1, dtype=torch.bool)))
        categories.extend((torch.full((stop,), int(certificate['complete']), dtype=torch.long),
                           torch.full((1,), -1, dtype=torch.long)))
    return {'states': torch.cat(states).half()[:, None], 'valid': torch.cat(valid)[:, None],
            'categories': torch.cat(categories)[:, None]}


@torch.no_grad()
def collect_quality_references(agent):
    """Collect once, retain rejected evidence, and publish only certified pools."""
    import time
    import random
    import numpy as np
    from rl_games.algos_torch.fb_task_inference import reward_components, checkpoint_digest
    environment = agent.vec_env.env
    configuration = environment.cfg['env']
    expected = {'forceConsecutiveNearGoalSteps': True, 'successSteps': 10,
                'maxConsecutiveSuccesses': 50, 'fixedSizeKeypointReward': True,
                'keypointScale': 1.5, 'referenceResetProbability': 0.0,
                'loadInitialStates': False, 'goodResetBoundary': 0,
                'useFixedGoalStates': False, 'goalObjectPose': None,
                'recordReferenceQuality': True, 'use_hack_object_pos_offset': False,
                'episodeLength': 600, 'useFixedInitObjectPose': False,
                'forceNoReset': False, 'fixedGoalStates': None, 'fixedGoalStatesJsonPath': None}
    for name, value in expected.items():
        if configuration.get(name) != value:
            raise ValueError('Quality collection configuration mismatch: ' + name)
    if not 0 < agent.options['quality_collection_steps'] <= 30000:
        raise ValueError('Collection vector-step guard exceeds approved first tranche')
    if not 0 < agent.options['quality_collection_seconds'] <= 7200:
        raise ValueError('Collection wall-time guard exceeds approved first tranche')
    if environment.max_episode_length != 600:
        raise ValueError('Runtime episode length differs from approved native path')
    if environment.success_tolerance != .01 or environment.initial_tolerance != .01:
        raise ValueError('Quality collection requires final 1cm native tolerance from initialization')
    if environment.target_tolerance != .01 or getattr(environment, 'reference_reset_bank', None) is not None:
        raise ValueError('Quality collection cannot use reset references or change tolerance')
    if agent.num_actors != 6144 or agent.options['expert_environments'] != 256:
        raise ValueError('Quality collection scene and stream budget differ from design')
    if environment._extra_reset_rules.__func__.__qualname__ != 'SimToolReal._extra_reset_rules':
        raise ValueError('Additional reset rules require an independent audit implementation')
    if not environment.reset_buf.bool().all():
        raise ValueError('Collection must begin with every environment awaiting an ordinary reset')
    config = configuration_container(environment.cfg)
    json.dumps(config, allow_nan=False)
    directory = agent.output_dir / ('reference_quality.rank' + str(agent.global_rank))
    directory.mkdir(exist_ok=False)
    teacher, checkpoint, teacher_root = agent._load_reference_policy()
    memory = [value.to(agent.device) for value in teacher.get_default_rnn_state()]
    observations = {key: value.clone() for key, value in agent.vec_env.reset().items()}
    selected = torch.randperm(agent.num_actors, device=agent.device)[:agent.options['expert_environments']]
    selected_cpu = selected.cpu()
    identity = {
        'schema_version': 2, 'collection_id': str(uuid.uuid4()), 'shard': 0,
        'rank': agent.global_rank, 'world_size': agent.world_size,
        'control_dt': environment.control_dt, 'simulation_dt': environment.dt,
        'control_frequency_inv': environment.control_freq_inv,
        'keypoint_unit_offsets': torch.tensor(environment.keypoints_offsets),
        'object_base_size': environment.object_base_size,
        'all_object_scales': environment.object_scales.cpu(),
        'asset_manifest': environment.reference_quality_assets,
        'table_asset': environment.reference_quality_table_asset,
        'with_table_force_sensor': environment.with_table_force_sensor,
        'torch_initial_seed': torch.initial_seed(), 'selected_environments': selected_cpu,
        'rng_before_collection': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                  'torch_cpu': torch.get_rng_state(),
                                  'torch_cuda': torch.cuda.get_rng_state(agent.device)},
        'curriculum_before_collection': {
            'frame_since_restart': environment.frame_since_restart,
            'last_curriculum_update': environment.last_curriculum_update,
            'success_tolerance': environment.success_tolerance,
            'tyler_curriculum_scale': environment._tyler_curriculum_scale,
        },
        'teacher_sha256': checkpoint_digest(teacher_root / 'model.pth'),
        'teacher_config_sha256': checkpoint_digest(teacher_root / 'config.yaml'),
        'configuration': config, 'fixed_offsets': environment.object_keypoint_offsets_fixed_size[selected].cpu(),
        'physical_offsets': environment.object_keypoint_offsets[selected].cpu(),
        'object_scales': environment.object_scales[selected].cpu(),
        'asset_paths': [str(value) for value in environment.object_asset_files] if hasattr(environment, 'object_asset_files') else None,
        'failure_cause_names': ['object_z_low', 'timeout', 'table_force_too_high', 'hand_far', 'dropped', 'extra_reset'],
        'timing': 'state/action pre-step; geometry/goal/rewards/failure masks post-step; reset applied inside step',
    }
    torch.save(identity, directory / 'identity.pt')
    buffers = [[] for _ in selected_cpu]
    episode_ids = [0 for _ in selected_cpu]
    start_steps = [0 for _ in selected_cpu]
    candidates, chunks, raw_manifest = [], [], []
    reset_before = torch.ones(agent.num_actors, dtype=torch.bool, device=agent.device)
    started = time.monotonic()
    chunk_index = 0
    completed_count = 0
    episode_summary = {'complete': [], 'prefix': [], 'rejected': [], 'right_censored': []}
    report = {'ready': False}

    def save_chunk():
        nonlocal chunk_index
        if not chunks:
            return
        path = directory / ('raw-' + str(chunk_index).zfill(5) + '.pt')
        temporary = path.with_suffix('.tmp')
        torch.save({name: torch.stack([row[name] for row in chunks]) for name in chunks[0]}, temporary)
        temporary.replace(path)
        raw_manifest.append({'name': path.name, 'sha256': checkpoint_digest(path),
                             'start_step': int(chunks[0]['step'][0]),
                             'end_step': int(chunks[-1]['step'][0])})
        chunks.clear()
        chunk_index += 1

    def finish_episode(stream, right_censored=False):
        nonlocal completed_count
        if not buffers[stream]:
            return
        rows = {name: torch.stack([row[name][stream] for row in buffers[stream]]) for name in buffers[stream][0]}
        certificate = certify_episode(rows, identity['fixed_offsets'][stream])
        record = {'collection_id': identity['collection_id'], 'rank': agent.global_rank, 'shard': 0, 'environment': int(selected_cpu[stream]), 'stream': stream,
                  'episode': episode_ids[stream], 'start_step': start_steps[stream],
                  'end_step': int(rows['step'][-1]), 'certificate': certificate}
        with (directory / 'certificates.jsonl').open('a') as output:
            output.write(json.dumps(record, allow_nan=False) + '\n')
        if certificate['complete'] or certificate['prefix_eligible']:
            path = directory / ('episode-' + str(stream) + '-' + str(episode_ids[stream]) + '.pt')
            torch.save({'rows': rows, 'certificate': certificate, 'identity': record}, path)
            record['path'] = str(path)
            record['sha256'] = checkpoint_digest(path)
            candidates.append(record)
        category = 'complete' if certificate['complete'] else 'prefix' if certificate['prefix_eligible'] else 'rejected'
        episode_summary[category].append(len(rows['done']))
        if right_censored:
            episode_summary['right_censored'].append(len(rows['done']))
        else:
            completed_count += 1
        buffers[stream] = []

    for step in range(agent.options['quality_collection_steps']):
        inputs = torch.cat((observations['obs'], observations['obs'].new_full((agent.num_actors, 1), 50.)), -1)
        result = teacher({'is_train': False, 'prev_actions': None, 'obs': inputs, 'rnn_states': memory})
        action, memory = result['mus'].clamp(-1, 1), result['rnn_states']
        state = observations['states'][selected].clone()
        goal_reset_before = environment.reset_goal_buf[selected].bool().clone()
        next_observations, reward, done, info = agent.vec_env.step(action)
        done = done.bool()
        components = torch.stack([info['episode_cumulative'][name] for name in reward_components], -1)
        if not torch.allclose(components.sum(-1), reward.reshape(-1), rtol=1e-5, atol=1e-5):
            raise ValueError('PPO reference reward components do not match original reward')
        row = {
            'state': state, 'next_state': next_observations['states'][selected].clone(),
            'action': action[selected].clone(), 'applied_action': environment.actions[selected].clone(),
            'reward': reward.reshape(-1)[selected].clone(), 'reward_components': components[selected].clone(),
            'reset_before': reset_before[selected].clone(), 'done': done[selected].clone(),
            'goal_reset_before': goal_reset_before, 'goal_success': environment.reset_goal_buf[selected].bool().clone(),
            'failure_causes': environment.reference_quality_failure_causes[selected].clone(),
            'object': environment.root_state_tensor[environment.object_indices[selected]].clone(),
            'goal': environment.goal_states[selected].clone(), 'object_initial': environment.object_init_state[selected].clone(),
            'table': environment.root_state_tensor[environment.table_indices[selected]].clone(),
            'joint_position': environment.arm_hand_dof_pos[selected].clone(),
            'joint_target': environment.prev_targets[selected].clone(),
            'fingertip_distance': environment.curr_fingertip_distances[selected].clone(),
            'scale_noise': environment.object_scale_noise_multiplier[selected].clone(),
            'fixed_keypoint_error': environment.keypoints_max_dist_fixed_size[selected].clone(),
            'actual_keypoint_error': environment.keypoints_max_dist[selected].clone(),
            'successes': environment.successes[selected].clone(),
            'progress': environment.progress_buf[selected].clone(),
            'native_lifted': environment.lifted_object[selected].clone(),
            'incoming_reset': environment.reference_quality_incoming_reset[selected].clone(),
            'noisy_initial': environment.reference_quality_noisy_initial[selected].clone(),
            'table_force': (environment.max_table_sensor_force_norm_smoothed[selected].clone()
                            if environment.with_table_force_sensor else torch.zeros_like(reward.reshape(-1)[selected])),
            'step': torch.full((len(selected),), step, dtype=torch.long, device=agent.device),
        }
        row = {name: value.cpu() for name, value in row.items()}
        if step == 0:
            row_bytes = sum(value.numel() * value.element_size() for value in row.values())
            identity['raw_vector_row_bytes'] = row_bytes
            identity['maximum_raw_buffer_bytes'] = row_bytes * agent.options['quality_collection_steps']
            torch.save(identity, directory / 'identity.pt')
        chunks.append(row)
        for stream in range(len(selected)):
            if row['reset_before'][stream]:
                if buffers[stream]:
                    raise ValueError('Physical reset occurred before previous episode was closed')
                start_steps[stream] = step
                episode_ids[stream] += 1
            buffers[stream].append(row)
            if row['done'][stream]:
                finish_episode(stream)
        for value in memory:
            value[:, done] = 0
        reset_before = done
        observations = {key: value.clone() for key, value in next_observations.items()}
        if len(chunks) >= agent.options['quality_chunk_steps']:
            save_chunk()
        if step % 100 == 0:
            _, report = select_episodes(candidates)
            status = {**report, 'step': step, 'simulation_transitions': (step + 1) * agent.num_actors,
                      'stored_transitions': (step + 1) * len(selected), 'completed_episodes': completed_count,
                      'seconds': time.monotonic() - started}
            (directory / 'status.json').write_text(json.dumps(status, indent=2))
            agent._status('quality_collection', collection_step=step, quality_ready=report['ready'])
        if report['ready'] or time.monotonic() - started >= agent.options['quality_collection_seconds']:
            break
    save_chunk()
    right_censored_count = sum(bool(buffer) for buffer in buffers)
    for stream in range(len(selected)):
        finish_episode(stream, right_censored=True)
    selected_episodes, report = select_episodes(candidates)
    report.update(episode_lengths=episode_summary, seconds=time.monotonic() - started, steps=step + 1,
                  simulation_transitions=(step + 1) * agent.num_actors,
                  stored_transitions=(step + 1) * len(selected), completed_episodes=completed_count,
                  right_censored_episodes=right_censored_count,
                  collection_identity_sha256=checkpoint_digest(directory / 'identity.pt'))
    (directory / 'raw_manifest.json').write_text(json.dumps(raw_manifest, indent=2))
    (directory / 'selection.json').write_text(json.dumps({'report': report, 'selected': selected_episodes,
                                                        'candidates': candidates}, indent=2))
    if report['ready']:
        episodes = [torch.load(row['path'], map_location='cpu') for row in selected_episodes]
        data = pack_reference_states([(episode['rows'], episode['certificate']) for episode in episodes])
        data['report'] = report
        data['sources'] = selected_episodes
        path = directory / 'quality_references.pt'
        torch.save(data, path)
        report['cache_sha256'] = checkpoint_digest(path)
    (directory / 'status.json').write_text(json.dumps(report, indent=2))
    agent._status('quality_collection_complete' if report['ready'] else 'quality_collection_incomplete', **report)
    # A deficient quota remains archived evidence. The controller cannot start a
    # learner until every rank passes independent certificate validation.
