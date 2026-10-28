"""Independent CPU audit of immutable PPO reference evidence.

This deliberately does not import the collector's geometry, certificate, selector,
or packer. A successful data audit is not a training-initialization approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch


DESIGN_SHA256 = 'ae1b1a689276e042aa9acd8dd43fd3fb2f72aa123ccbaf485a3133d6485cb013'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def physical_error(objects, goals, offsets):
    """Independent matrix formulation of the environment's xyzw rotations."""
    def points(poses):
        poses = np.asarray(poses, dtype=np.float64)
        x, y, z, w = poses[:, 3:7].T
        matrices = np.stack((1 - 2 * (y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                             2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w),
                             2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)), -1).reshape(-1, 3, 3)
        return np.einsum('tij,tkj->tki', matrices,
                         np.broadcast_to(offsets, (len(poses), *np.shape(offsets)[-2:]))) + poses[:, None, :3]
    return np.linalg.norm(points(objects) - points(goals), axis=-1).max(axis=-1)


def conservative_object_bottom(objects, asset):
    """Lower bound for retained one-link primitive collision geometry.

    The native loader replaces cylinders with capsules. Using cylinder half-length
    plus spherical radius bounds either capsule length convention conservatively.
    Observation scale noise does not change actor collision geometry in this task.
    """
    model = ET.fromstring(asset['urdf'])
    require(len(model.findall('link')) == 1 and not model.findall('joint'),
            'Physical clearance audit requires the approved one-link primitives')
    x, y, z, w = np.asarray(objects, dtype=np.float64)[:, 3:7].T
    vertical = np.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)
    bottoms = []
    for collision in model.findall('.//collision'):
        origin = collision.find('origin')
        xyz = np.zeros(3) if origin is None else np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
        rpy = np.zeros(3) if origin is None else np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')
        cr, cp, cy = np.cos(rpy)
        sr, sp, sy = np.sin(rpy)
        rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                             [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                             [-sp, cp*sr, cp*cr]])
        direction = vertical @ rotation
        geometry = collision.find('geometry')
        box = geometry.find('box')
        cylinder = geometry.find('cylinder')
        sphere = geometry.find('sphere')
        if box is not None:
            support = np.abs(direction) @ (np.fromstring(box.attrib['size'], sep=' ') / 2)
        elif cylinder is not None:
            support = np.abs(direction[:, 2]) * float(cylinder.attrib['length']) / 2 + float(cylinder.attrib['radius'])
        elif sphere is not None:
            support = np.full(len(objects), float(sphere.attrib['radius']))
        else:
            raise ValueError('Unsupported primitive physical-clearance geometry')
        bottoms.append(objects[:, 2] + vertical @ xyz - support)
    require(bool(bottoms), 'Missing collision geometry')
    return np.min(bottoms, axis=0)


def audit_episode(rows, identity, stream):
    """Derive goals, reset causes, lift, boundaries and counts from raw evidence."""
    values = {name: value.numpy() for name, value in rows.items()}
    count = len(values['done'])
    require(count >= 2 and count <= 29950, 'Invalid/oversized physical episode length')
    require(values['reset_before'][0] and not values['reset_before'][1:].any(), 'Missing/reset-crossing episode start')
    require(not values['done'][:-1].any(), 'Episode crosses terminal')
    require(np.array_equal(np.diff(values['step']), np.ones(count - 1)), 'Noncontiguous source steps')
    for name in ('object', 'goal', 'object_initial', 'noisy_initial', 'state', 'next_state', 'action', 'reward_components'):
        require(values[name].dtype == np.float32 and np.isfinite(values[name]).all(), 'Invalid float32 source: ' + name)
    for name in ('object', 'goal', 'noisy_initial'):
        require(np.allclose(np.linalg.norm(values[name][:, 3:7], axis=-1), 1, rtol=1e-4, atol=1e-5),
                'Physical source has invalid quaternion: ' + name)
    require(np.array_equal(values['next_state'][:-1], values['state'][1:]), 'State/action timing does not chain')
    for name in ('object_initial', 'noisy_initial', 'scale_noise', 'table'):
        require(np.array_equal(values[name], np.broadcast_to(values[name][0], values[name].shape)), 'Reset-only field changed: ' + name)
    config = identity['configuration']['env']
    fixed_offsets = identity['fixed_offsets'][stream].numpy()
    unit_offsets = identity['keypoint_unit_offsets'].numpy()
    expected_offsets = unit_offsets * np.asarray(config['fixedSize']) * 1.5 / 2
    require(np.allclose(fixed_offsets, expected_offsets, rtol=1e-6, atol=1e-9), 'Fixed-size geometry identity mismatch')
    actual_offsets = identity['physical_offsets'][stream].numpy()
    expected_actual = unit_offsets * identity['object_scales'][stream].numpy() * identity['object_base_size'] * 1.5 / 2
    require(np.allclose(actual_offsets, expected_actual, rtol=1e-6, atol=1e-9), 'Actual-size geometry identity mismatch')
    error = physical_error(values['object'], values['goal'], fixed_offsets)
    actual_error = physical_error(values['object'], values['goal'], actual_offsets[None] * values['scale_noise'][:, None])
    require(np.allclose(error, values['fixed_keypoint_error'], rtol=1e-4, atol=2e-6), 'Native fixed geometry disagrees with stored poses')
    require(np.allclose(actual_error, values['actual_keypoint_error'], rtol=1e-4, atol=2e-6), 'Native actual geometry disagrees with stored poses')
    height = values['object'][:, 2].astype(np.float64) - values['object_initial'][:, 2]
    # Mirror only the native latch/reset arithmetic in float32, independently of
    # the new float64 15cm/12-step quality definition.
    native_lift = np.logical_or.accumulate((np.float32(.05) + values['object'][:, 2] - values['object_initial'][:, 2]) > config['liftingBonusThreshold'])
    require(np.array_equal(native_lift, values['native_lifted'].astype(bool)), 'Native lifted latch mismatch')
    near_count = 0
    lift_count = 0
    lift_at = None
    success_steps = []
    activations = [0]
    progress = 0
    failure = np.zeros((count, 6), dtype=bool)
    successes = np.zeros(count, dtype=bool)
    for step in range(count):
        expected_reset = step > 0 and successes[step - 1]
        if step:
            require(bool(values['goal_reset_before'][step]) == expected_reset, 'Goal activation timing mismatch')
            if expected_reset:
                near_count = 0
                activations.append(step)
            else:
                require(np.array_equal(values['goal'][step], values['goal'][step - 1]), 'Goal changed without activation')
        near_count = near_count + 1 if error[step] <= .015 else 0
        successes[step] = near_count >= 10
        if successes[step]:
            success_steps.append(step)
        progress = 0 if successes[step] else progress + 1
        require(progress == values['progress'][step], 'Timeout progress order mismatch')
        require(len(success_steps) == values['successes'][step], 'Native cumulative success mismatch')
        failure[step, 0] = values['object'][step, 2] < .1
        failure[step, 1] = progress >= 599
        failure[step, 2] = identity['with_table_force_sensor'] and values['table_force'][step] > 100
        failure[step, 3] = values['fingertip_distance'][step].max() > 1.5
        failure[step, 4] = config['resetWhenDropped'] and native_lift[step] and values['object'][step, 2] < values['object_initial'][step, 2]
        # The approved base task's _extra_reset_rules is identity. A future task
        # override requires its own independent evidence instead of trusting a flag.
        require(not bool(values['incoming_reset'][step]), 'Unaccounted original reset_buf')
        expected_done = failure[step].any() or len(success_steps) >= 50
        require(bool(values['done'][step]) == expected_done, 'Terminal reason mismatch')
        lift_count = lift_count + 1 if height[step] >= .15 else 0
        if lift_count >= 12 and lift_at is None:
            lift_at = step
    require(np.array_equal(failure, values['failure_causes'].astype(bool)), 'Failure-cause source mismatch')
    require(np.array_equal(successes, values['goal_success'].astype(bool)), 'Goal labels disagree with geometry')
    complete = bool(values['done'][-1] and successes[-1] and len(success_steps) == 50 and not failure.any() and lift_at is not None)
    admissible = [step for step in success_steps if not failure[:step + 1].any()]
    stop = count if complete else admissible[-1] + 1 if admissible and lift_at is not None and lift_at <= admissible[-1] else 0
    valid = ~values['reset_before'][:stop].astype(bool) & ~values['done'][:stop].astype(bool)
    table = ET.fromstring(identity['table_asset']['urdf'])
    box = table.find('.//collision/geometry/box')
    require(box is not None, 'Table physical clearance requires retained box geometry')
    table_height = float(box.attrib['size'].split()[2])
    table_origin = table.find('.//collision/origin')
    origin_z = float(table_origin.attrib.get('xyz', '0 0 0').split()[2]) if table_origin is not None else 0.0
    clearance = values['object'][:, 2] - values['table'][:, 2] - table_height / 2 - origin_z
    asset_index = int(identity['selected_environments'][stream]) % len(identity['asset_manifest'])
    bottom = conservative_object_bottom(values['object'], identity['asset_manifest'][asset_index])
    physical_clearance = bottom - values['table'][:, 2] - table_height / 2 - origin_z
    return {'complete': complete, 'prefix_eligible': bool(stop and not complete), 'stop': stop,
            'raw_transitions': count, 'valid_state_rows': int(valid.sum()),
            'valid_adjacent_pairs': int(np.logical_and(valid[:-1], valid[1:]).sum()),
            'goal_success_steps': success_steps, 'goal_activation_steps': activations,
            'lift_certified_at': lift_at, 'actual_keypoint_within_5cm_fraction': float((actual_error <= .05).mean()),
            'object_center_table_clearance_min_m': float(clearance.min()),
            'physical_table_clearance_conservative_min_m': float(physical_clearance.min()),
            'noisy_initial_object_z_m': float(values['noisy_initial'][0, 2])}


def audit_directory(directory):
    directory = Path(directory)
    selection = json.loads((directory / 'selection.json').read_text())
    report = selection['report']
    require(report['ready'], 'Collection quota incomplete')
    require(digest(directory / 'identity.pt') == report['collection_identity_sha256'], 'Collection identity digest mismatch')
    identity = torch.load(directory / 'identity.pt', map_location='cpu')
    require(identity['schema_version'] == 2, 'Missing raw evidence schema')
    config = identity['configuration']['env']
    required = {'forceConsecutiveNearGoalSteps': True, 'successSteps': 10,
                'maxConsecutiveSuccesses': 50, 'fixedSizeKeypointReward': True,
                'keypointScale': 1.5, 'referenceResetProbability': 0.0,
                'loadInitialStates': False, 'goodResetBoundary': 0,
                'useFixedGoalStates': False, 'goalObjectPose': None,
                'use_hack_object_pos_offset': False, 'episodeLength': 600,
                'useFixedInitObjectPose': False, 'forceNoReset': False,
                'fixedGoalStates': None, 'fixedGoalStatesJsonPath': None,
                'successTolerance': .01, 'targetSuccessTolerance': .01}
    require(all(config.get(name) == expected for name, expected in required.items()),
            'Raw source configuration differs from approved quality protocol')
    require(report['steps'] <= 30000, 'Collection exceeded approved step budget')
    selected_envs = identity['selected_environments'].tolist()
    require(len(selected_envs) == len(set(selected_envs)) == 256, 'Nonunique/wrong stream population')
    require(identity['world_size'] == 4, 'Formal collection requires four ranks')
    for asset in identity['asset_manifest'] + [identity['table_asset']]:
        require(hashlib.sha256(asset['urdf'].encode()).hexdigest() == asset['sha256'], 'Asset source digest mismatch')
        for mesh in asset['meshes']:
            require(hashlib.sha256(mesh['content']).hexdigest() == mesh['sha256'], 'Asset mesh digest mismatch')
    records = [json.loads(line) for line in (directory / 'certificates.jsonl').read_text().splitlines()]
    key = lambda row: (row['shard'], row['environment'], row['episode'])
    require(len({key(row) for row in records}) == len(records), 'Duplicate archived episode identity')
    candidates = selection['candidates']
    require(len({key(row) for row in candidates}) == len(candidates), 'Duplicate candidate episode')
    record_map = {key(row): row for row in records}
    for row in candidates:
        require(key(row) in record_map and row['certificate'] == record_map[key(row)]['certificate'], 'Candidate missing from complete source catalog')
        require(row['collection_id'] == identity['collection_id'] and row['rank'] == identity['rank'], 'Source collection/rank mismatch')
        require(selected_envs[row['stream']] == row['environment'], 'Source env mapping mismatch')
    ordered = sorted(candidates, key=lambda row: (row['shard'], row['end_step'], row['environment']))
    chosen = []
    total = full = 0
    for row in ordered:
        amount = row['certificate']['valid_adjacent_pairs']
        if row['certificate']['complete'] and full < 192000 and total + amount <= 384000:
            chosen.append(row)
            total += amount
            full += amount
    if full >= 192000:
        for row in ordered:
            amount = row['certificate']['valid_adjacent_pairs']
            if row['certificate']['prefix_eligible'] and total + amount <= 384000:
                chosen.append(row)
                total += amount
    require(chosen == selection['selected'], 'Selection violates preregistered whole-episode order')
    require(380160 <= total <= 384000 and full >= 192000 and full >= total / 2, 'Independent transition quota failed')
    require(report['valid_adjacent_pairs'] == total and report['complete_pairs'] == full, 'Reported quota mismatch')
    episodes = {}
    audits = []
    source_digests = set()
    intervals = {}
    for row in chosen:
        # Resolve within this immutable directory, independent of its original mount.
        path = directory / Path(row['path']).name
        require(digest(path) == row['sha256'], 'Selected episode source digest mismatch')
        source = torch.load(path, map_location='cpu')
        require(source['certificate'] == row['certificate'], 'Selected certificate artifact mismatch')
        require(all(source['identity'][name] == row[name] for name in source['identity']),
                'Selected episode identity metadata mismatch')
        result = audit_episode(source['rows'], identity, row['stream'])
        for field in ('complete', 'prefix_eligible', 'stop', 'raw_transitions', 'valid_state_rows', 'valid_adjacent_pairs', 'goal_success_steps', 'lift_certified_at'):
            require(result[field] == row['certificate'][field], 'Independent certificate mismatch: ' + field)
        steps = source['rows']['step']
        require(int(steps[0]) == row['start_step'] and int(steps[-1]) == row['end_step'], 'Episode interval provenance mismatch')
        previous = intervals.setdefault(row['stream'], [])
        require(all(row['end_step'] < start or row['start_step'] > stop for start, stop in previous), 'Overlapping independent source transitions')
        previous.append((row['start_step'], row['end_step']))
        signature = hashlib.sha256(source['rows']['object'].numpy().tobytes() + source['rows']['goal'].numpy().tobytes()).hexdigest()
        require(signature not in source_digests, 'Copied episode trajectory cannot count as independent data')
        source_digests.add(signature)
        episodes[key(row)] = source['rows']
        audits.append({'identity': key(row), **result})
    manifest = json.loads((directory / 'raw_manifest.json').read_text())
    next_step = 0
    previous_done = torch.ones(256, dtype=torch.bool)
    matched_rows = {key(row): 0 for row in chosen}
    for shard in manifest:
        path = directory / shard['name']
        require(digest(path) == shard['sha256'], 'Raw shard digest mismatch')
        require(shard['start_step'] == next_step, 'Raw archive gap or duplicate')
        raw = torch.load(path, map_location='cpu')
        require(torch.equal(raw['step'], torch.arange(shard['start_step'], shard['end_step'] + 1)[:, None].expand(-1, 256)), 'Raw time/env axis mismatch')
        expected_reset = torch.cat((previous_done[None], raw['done'][:-1].bool()))
        require(torch.equal(raw['reset_before'].bool(), expected_reset), 'Raw physical episode starts mismatch terminals')
        previous_done = raw['done'][-1].bool()
        for row in chosen:
            start = max(row['start_step'], shard['start_step'])
            stop = min(row['end_step'], shard['end_step']) + 1
            if start >= stop:
                continue
            source = episodes[key(row)]
            for name, value in source.items():
                require(torch.equal(value[start - row['start_step']:stop - row['start_step']], raw[name][start - shard['start_step']:stop - shard['start_step'], row['stream']]), 'Selected source does not match raw archive: ' + name)
            matched_rows[key(row)] += stop - start
        next_step = shard['end_step'] + 1
    require(next_step == report['steps'], 'Raw archive misses final collection steps')
    require(all(matched_rows[key(row)] == row['certificate']['raw_transitions'] for row in chosen), 'Selected source has unarchived rows')
    cache = torch.load(directory / 'quality_references.pt', map_location='cpu')
    require(cache['sources'] == chosen and cache['report'] == report, 'Cache source/report identity mismatch')
    cursor = 0
    for row in chosen:
        source = episodes[key(row)]
        stop = row['certificate']['stop']
        require(torch.equal(cache['states'][cursor:cursor + stop, 0], source['state'][:stop].half()), 'Cache state is not selected source')
        valid = ~source['reset_before'][:stop].bool() & ~source['done'][:stop].bool()
        require(torch.equal(cache['valid'][cursor:cursor + stop, 0], valid), 'Cache mask differs from v4 boundary rule')
        require(bool((cache['categories'][cursor:cursor + stop, 0] == int(row['certificate']['complete'])).all()), 'Cache category mismatch')
        cursor += stop
        require(not cache['valid'][cursor, 0] and cache['categories'][cursor, 0] == -1, 'Missing episode separator')
        cursor += 1
    require(cursor == len(cache['states']) == len(cache['valid']) == len(cache['categories']), 'Extra cache rows')
    valid = cache['valid'][:, 0].numpy().astype(bool)
    require(int((valid[:-1] & valid[1:]).sum()) == total, 'Cache adjacent-pair budget mismatch')
    windows = {}
    generator = np.random.default_rng(918273 + identity['rank'])
    for length in (8, 9, 250):
        starts = np.flatnonzero(np.convolve(valid.astype(np.int32), np.ones(length, dtype=np.int32), 'valid') == length)
        require(len(starts) > 0, 'Missing required sampler window')
        categories = cache['categories'][starts, 0].numpy()
        sampled = categories[generator.integers(0, len(starts), size=100000)]
        windows[str(length)] = {'starts': len(starts), 'complete_starts': int(categories.sum()),
                                'complete_probability': float(categories.mean()), 'sample_count': len(sampled),
                                'independent_uniform_sample_complete_fraction': float(sampled.mean())}
    return {'status': 'PASS_DATA_CPU_ONLY', 'training_approved': False, 'design_sha256': DESIGN_SHA256,
            'rank': identity['rank'], 'collection_id': identity['collection_id'],
            'torch_initial_seed': identity['torch_initial_seed'], 'valid_adjacent_pairs': total,
            'complete_pairs': full, 'valid_state_rows': int(valid.sum()), 'state_slots': len(valid),
            'separator_rows': len(chosen), 'window_categories': windows,
            'input_sha256': {name: digest(directory / name) for name in ('identity.pt', 'selection.json', 'raw_manifest.json', 'certificates.jsonl', 'quality_references.pt')},
            'raw_shards': manifest, 'episode_audits': audits, 'trajectory_signatures': sorted(source_digests)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reports = [audit_directory(path) for path in args.directories]
    require(len(reports) == 4 and {row['rank'] for row in reports} == set(range(4)), 'Aggregate requires all four ranks')
    require(len({row['collection_id'] for row in reports}) == 4, 'Collection identities copied across ranks')
    require(len({row['torch_initial_seed'] for row in reports}) == 4, 'Rank RNG seed identity repeated')
    signatures = [signature for row in reports for signature in row['trajectory_signatures']]
    require(len(set(signatures)) == len(signatures), 'Trajectory copied across ranks')
    result = {'status': 'PASS_DATA_CPU_ONLY', 'training_approved': False,
              'design_sha256': DESIGN_SHA256, 'verifier_sha256': digest(__file__), 'ranks': reports}
    args.output.write_text(json.dumps(result, indent=2))


def require_training_approval(options, rank, reference, reset_bank):
    """Fail closed until independent data AND initialization review is bound.

    The approval is an external review artifact, never produced by collection or
    by this CPU geometry audit. Its initialization evidence must document the
    adopted protocol/loader/scene contract; historical full state is unavailable.
    """
    path = options.get('quality_training_approval_path')
    expected = options.get('quality_training_approval_sha256')
    require(bool(path and expected), 'Quality references require independent training-precondition approval')
    require(digest(path) == expected, 'Training-precondition approval digest changed')
    approval = json.loads(Path(path).read_text())
    require(approval['status'] == 'PASS_TRAINING_PRECONDITIONS', 'Training preconditions are not approved')
    require(approval['design_sha256'] == DESIGN_SHA256, 'Training approval design mismatch')
    from rl_games.algos_torch.fb_reference_initialization import CONTRACT_SHA256
    require(approval['schema_version'] == 2, 'Legacy historical-state boolean approval is invalid')
    require(approval['initialization_contract_sha256'] == CONTRACT_SHA256, 'Initialization contract mismatch')
    require(approval['source_kind'] == 'formal_ppo_collection', 'Synthetic data cannot authorize training')
    evidence = approval['initialization_evidence']
    require(digest(evidence['path']) == evidence['sha256'], 'Initialization evidence changed')
    initialization = json.loads(Path(evidence['path']).read_text())
    require(initialization['contract_sha256'] == CONTRACT_SHA256, 'Initialization evidence contract mismatch')
    require(initialization['historical_full_state_available'] is False,
            'Do not fabricate historical PhysX/RNG full-state availability')
    require('historical_state_match' not in initialization and 'historical_state_match' not in evidence,
            'Historical full-state equality is not this protocol contract')
    for gate in ('launch_protocol', 'cpu_quality_loader', 'reset_scene_provenance'):
        receipt = initialization[gate]
        require(digest(receipt['path']) == receipt['sha256'], 'Initialization gate evidence changed: ' + gate)
        payload = json.loads(Path(receipt['path']).read_text())
        require(payload['status'] == 'PASS' and payload['scope'] == gate,
                'Initialization gate is incomplete: ' + gate)
        require(payload['contract_sha256'] == CONTRACT_SHA256, 'Gate contract mismatch: ' + gate)
    require(len(initialization['rank_events']) == len(initialization['expected_scenes']) == len(initialization['expected_launches']) == 4,
            'Initialization protocol must cover four ranks')
    require(bool(initialization['production_files']), 'Startup source/dependency identities are required')
    require([event['rank'] for event in initialization['rank_events']] == list(range(4)),
            'Initialization rank trace is missing or reordered')
    require(initialization['cpu_quality_cache_sha256'][rank] == digest(reference),
            'CPU loader test did not use this formal quality cache')
    require(initialization['original_reset_sha256'][rank] == digest(reset_bank),
            'Initialization provenance did not use this original bank')
    audit_identity = approval['data_audit']
    require(digest(audit_identity['path']) == audit_identity['sha256'], 'Independent data audit changed')
    audit = json.loads(Path(audit_identity['path']).read_text())
    require(audit['status'] == 'PASS_DATA_CPU_ONLY' and audit['design_sha256'] == DESIGN_SHA256,
            'Missing independent four-rank data certificate')
    rank_reports = audit['ranks']
    require(len(rank_reports) == 4 and {row['rank'] for row in rank_reports} == set(range(4)),
            'Data certificate must cover every rank')
    report = next(row for row in rank_reports if row['rank'] == rank)
    require(report['input_sha256']['quality_references.pt'] == digest(reference), 'Audited cache identity mismatch')
    require(approval['original_reset_sha256'][rank] == digest(reset_bank), 'Approved reset bank identity mismatch')
    return {'initialization': initialization, 'approval_sha256': expected, 'approval_path': str(path)}


if __name__ == '__main__':
    main()
