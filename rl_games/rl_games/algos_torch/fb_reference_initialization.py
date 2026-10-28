"""Visible initialization and scene checks; never claim a historical solver snapshot."""
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from rl_games.algos_torch.fb_reference_audit import digest, require
from rl_games.algos_torch.fb_reference_quality import asset_manifest, configuration_container
from rl_games.algos_torch.fb_native_preparation import ACTORS, load_prepared_roots


CONTRACT_SHA256 = 'eb4e57e74754fd2dd666d0be02376f36f4160e4deb2be25fc7801fcd5d3b0019'


def state_digest(value):
    """Hash visible values without sampling, serializing modules, or CUDA setup."""
    output = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            output.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            output.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            output.update(str((item.dtype, item.shape)).encode())
            output.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            output.update(type(item).__name__.encode())
            for part in item:
                add(part)
        elif isinstance(item, (str, int, float, bool, type(None), np.generic)):
            output.update((type(item).__name__ + ':' + repr(item)).encode())
        else:
            raise TypeError('Unsupported visible-state leaf: ' + type(item).__name__)
        output.update(b'\0')
    add(configuration_container(value))
    return output.hexdigest()


def visible_snapshot(agent):
    """Observe all tensor/scalar state plus modules/optimizers and RNG streams.

    Opaque simulator handles are listed, not hashed as historical hidden state.
    The three permitted loader outputs are deliberately excluded.
    """
    visible = {}
    opaque = []
    for owner, fields in (('agent', vars(agent)), ('environment', vars(agent.vec_env.env))):
        for name, value in fields.items():
            if name in ('experts', 'expert_report', 'reference_reset_bank'):
                continue
            key = owner + '/' + name
            if isinstance(value, torch.nn.Module):
                value = {'state': value.state_dict(),
                         'training_modes': {key: module.training for key, module in value.named_modules()},
                         'requires_grad': {key: parameter.requires_grad for key, parameter in value.named_parameters()}}
            elif isinstance(value, torch.optim.Optimizer):
                value = value.state_dict()
            try:
                visible[key] = state_digest(value)
            except TypeError:
                opaque.append(key)
    visible['rng/python'] = state_digest(random.getstate())
    visible['rng/numpy'] = state_digest(np.random.get_state())
    visible['rng/torch_cpu'] = state_digest(torch.get_rng_state())
    if torch.device(agent.device).type == 'cuda':
        require(torch.cuda.is_initialized(), 'Initialization probe must not initialize CUDA RNG')
        visible['rng/torch_cuda'] = state_digest(torch.cuda.get_rng_state(agent.device))
    return {'visible': visible, 'opaque_fields': sorted(opaque),
            'historical_full_state_available': False, 'runtime_device': str(agent.device)}


def _vector(value):
    return [float(getattr(value, name)) for name in ('x', 'y', 'z')]


def _dof_properties(value):
    # Field-wise encoding excludes potentially uninitialized structured padding.
    return {name: value[name].tolist() for name in value.dtype.names}


def _shape_ranges(values):
    return [[int(value.start), int(value.count)] for value in values]


def capture_scene(environment):
    """Read current assets and all actors; no step, reset or property mutation."""
    from isaacgym import gymapi
    assets = environment.reference_quality_assets
    robot = asset_manifest(environment._robot_urdf_path)
    gym = environment.gym
    asset_handles = environment.reference_initialization_assets
    topologies = {}
    for label, handles in asset_handles.items():
        topologies[label] = [{'dof_names': list(gym.get_asset_dof_names(handle)),
                             'body_names': list(gym.get_asset_rigid_body_names(handle)),
                             'body_shape_ranges': _shape_ranges(gym.get_asset_rigid_body_shape_indices(handle))}
                            for handle in handles]
    root_states = environment.root_state_tensor.detach().cpu()
    rows = []
    mappings = []
    for index, env in enumerate(environment.envs):
        actors = {}
        mapping = []
        for actor_number in range(gym.get_actor_count(env)):
            handle = gym.get_actor_handle(env, actor_number)
            label = gym.get_actor_name(env, handle)
            bodies = list(gym.get_actor_rigid_body_names(env, handle))
            physical = [{'mass': float(value.mass), 'com_position': _vector(value.com),
                         'inertia': [_vector(getattr(value.inertia, axis)) for axis in ('x', 'y', 'z')]}
                        for value in gym.get_actor_rigid_body_properties(env, handle)]
            shapes = [{name: getattr(value, name) for name in
                       ('friction', 'rolling_friction', 'torsion_friction', 'restitution',
                        'compliance', 'thickness', 'contact_offset', 'rest_offset', 'filter')
                       if hasattr(value, name)} for value in gym.get_actor_rigid_shape_properties(env, handle)]
            actor_index = gym.get_actor_index(env, handle, gymapi.DOMAIN_SIM)
            actors[label] = {
                'dof_names': list(gym.get_actor_dof_names(env, handle)),
                'dof_properties': _dof_properties(gym.get_actor_dof_properties(env, handle)),
                'body_names': bodies,
                'body_shape_ranges': _shape_ranges(gym.get_actor_rigid_body_shape_indices(env, handle)),
                'body_env_indices': [gym.find_actor_rigid_body_index(env, handle, name, gymapi.DOMAIN_ENV) for name in bodies],
                'actor_sim_index': actor_index,
                'initial_root_state': root_states[actor_index].tolist(),
                'actor_scale': float(gym.get_actor_scale(env, handle)),
                'rigid_body_properties': physical, 'shape_properties': shapes,
            }
            mapping.append({'name': label, 'actor_sim_index': actor_index})
        rows.append({'environment': index, 'asset_index': index % len(assets), 'actors': actors})
        mappings.append({'environment': index, 'asset_index': index % len(assets), 'actors': mapping})
    static = {'environment_count': environment.num_envs,
              'assets': [{'name': value['name'], 'sha256': value['sha256'],
                          'mesh_sha256': [mesh['sha256'] for mesh in value['meshes']]} for value in assets],
              'table_sha256': environment.reference_quality_table_asset['sha256'],
              'robot_sha256': robot['sha256'],
              'robot_meshes': [{'filename': value['filename'], 'sha256': value['sha256']} for value in robot['meshes']],
              'table_meshes': [{'filename': value['filename'], 'sha256': value['sha256']}
                               for value in environment.reference_quality_table_asset['meshes']],
              'object_scales': environment.object_scales.cpu().tolist(),
              'fixed_offsets_sha256': state_digest(environment.object_keypoint_offsets_fixed_size),
              'physical_offsets_sha256': state_digest(environment.object_keypoint_offsets),
              'task_configuration': configuration_container(environment.cfg), 'environments': mappings}
    return {'schema_version': 2, 'exact_static_mapping': static,
            'runtime_asset_topologies': topologies, 'environments': rows}


def validate_scene(environment, scene, expected):
    """Check frozen mapping and source-derived constraints, not historical native bytes."""
    from isaacgymenvs.tasks.simtoolreal.utils import populate_dof_properties
    require(expected['schema_version'] == 2, 'Expected split native scene schema')
    require(expected['contract_sha256'] == CONTRACT_SHA256, 'Native scene contract differs')
    require(scene['exact_static_mapping'] == expected['exact_static_mapping'], 'Static bank-to-scene mapping differs')
    constraints = expected['native_constraints']
    require(not environment.randomize and environment.cfg['task']['randomize'] is False,
            'Actual v4 protocol disables domain randomization')
    require(environment.frame_since_restart == 0, 'Scene must precede rollout/reset stepping')
    require(environment.cfg['env']['modifyAssetFrictions'] is True, 'Expected fixed production friction path')
    require(len(constraints['robot_dof_names']) == 29, 'Old reset bank requires all 29 DOF columns')
    require(0 <= constraints['root_atol'] <= 1e-6, 'Nominal pose tolerance must be predeclared and bounded')
    actor_names = ['robot', 'object', 'table_object', 'goal_object']
    require(set(constraints['nominal_root_states']) == set(actor_names), 'All actor construction poses are required')
    prepared, centers = load_prepared_roots(expected['exact_static_mapping'], constraints['prepared_roots'])
    require(len(scene['environments']) == len(prepared), 'Native environment row count differs')
    repeated = {}
    for index, row in enumerate(scene['environments']):
        require(row['environment'] == index and row['asset_index'] == index % len(centers),
                'Native environment/asset mapping differs')
        require(list(row['actors']) == actor_names, 'Actor creation order/count differs')
        body_cursor = 0
        for label, actor in row['actors'].items():
            asset_index = row['asset_index'] if label in ('object', 'goal_object') else 0
            topology = scene['runtime_asset_topologies'][label][asset_index]
            for field in ('dof_names', 'body_names', 'body_shape_ranges'):
                require(actor[field] == topology[field], 'Current asset-to-actor topology differs: ' + label + '/' + field)
            count = len(actor['body_names'])
            require(actor['body_env_indices'] == list(range(body_cursor, body_cursor + count)), 'Body env indices differ')
            body_cursor += count
            require(actor['actor_scale'] == 1.0, 'Native actor scale changed')
            actor_offset = ACTORS.index(label)
            require(actor['actor_sim_index'] == index * len(ACTORS) + actor_offset,
                    'Native actor simulation mapping differs')
            # Do not round a perturbed observation down to float32 before checking.
            observed_root = np.asarray(actor['initial_root_state'], dtype=np.float64)
            expected_root = prepared[index, actor_offset]
            require(observed_root.shape == (13,), 'Native root component count differs')
            mismatch = np.flatnonzero(observed_root != expected_root)
            require(len(mismatch) == 0,
                    'Prepared native root differs: environment=' + str(index) + ' actor=' + label +
                    ' components=' + str(mismatch.tolist()))
            if label in ('object', 'goal_object'):
                require(len(actor['rigid_body_properties']) == 1, 'Prepared one-link native body count differs')
                require(np.array_equal(actor['rigid_body_properties'][0]['com_position'], centers[asset_index]),
                        'Native COM differs from source URDF: environment=' + str(index) + ' actor=' + label)
            ranges = actor['body_shape_ranges']
            shape_indices = [index for start, count in ranges for index in range(start, start + count)]
            require(shape_indices == list(range(len(actor['shape_properties']))), 'Body-to-shape partition differs')
            if label in ('robot', 'object', 'table_object'):
                for body, (start, count) in zip(actor['body_names'], ranges):
                    friction = 1.5 if label == 'robot' and body in environment.fingertips else .5
                    require(all(shape['friction'] == friction for shape in actor['shape_properties'][start:start + count]),
                            'Fixed production friction differs: ' + label + '/' + body)
            if label == 'robot':
                require(actor['dof_names'] == constraints['robot_dof_names'], 'Old bank DOF column ordering differs')
                require(actor['body_names'] == list(constraints['robot_body_filters']), 'Frozen import body ordering differs')
                for body, (start, count) in zip(actor['body_names'], ranges):
                    require(all(value['filter'] == constraints['robot_body_filters'][body]
                                for value in actor['shape_properties'][start:start + count]),
                            'Production adjacent-link filters differ')
                # Populate only a detached copy, preserving imported fields not set by production.
                properties = environment.gym.get_actor_dof_properties(environment.envs[row['environment']],
                                                                      environment.robots[row['environment']]).copy()
                populate_dof_properties(properties, environment.num_arm_dofs, environment.num_hand_dofs)
                require(_dof_properties(properties) == actor['dof_properties'], 'Source-fixed robot DOF properties differ')
            for body in actor['rigid_body_properties']:
                require(np.isfinite([body['mass']] + body['com_position'] + sum(body['inertia'], [])).all(),
                        'Nonfinite imported rigid-body properties')
                require(body['mass'] >= 0, 'Negative native mass')
            for shape in actor['shape_properties']:
                require(np.isfinite(list(shape.values())).all(), 'Nonfinite native shape property')
            identity = (label, asset_index)
            physical = {field: actor[field] for field in ('dof_properties', 'rigid_body_properties', 'shape_properties')}
            signature = state_digest(physical)
            require(repeated.setdefault(identity, signature) == signature, 'Repeated identical asset properties differ')


def canonical_launch(payload):
    """Remove only the approval SHA self-reference; retain every other launch field."""
    result = copy.deepcopy(payload)
    result['configuration']['train']['params']['config']['fb']['quality_training_approval_sha256'] = '<BOUND_APPROVAL_SHA256>'
    prefix = 'train.params.config.fb.quality_training_approval_sha256='
    matches = [index for index, value in enumerate(result['argv']) if value.lstrip('+').startswith(prefix)]
    require(len(matches) == 1, 'Exactly one approval SHA CLI override is required')
    index = matches[0]
    result['argv'][index] = result['argv'][index].split('=', 1)[0] + '=<BOUND_APPROVAL_SHA256>'
    return result


def verify_launch_approval(payload, approval):
    """The only canonicalized value must still match the independently opened approval."""
    options = payload['configuration']['train']['params']['config']['fb']
    require(options['quality_training_approval_sha256'] == approval['approval_sha256'], 'Actual launch approval SHA differs')
    require(options['quality_training_approval_path'] == approval['approval_path'], 'Actual launch approval path differs')
    prefix = 'train.params.config.fb.quality_training_approval_sha256='
    values = [value.split('=', 1)[1] for value in payload['argv'] if value.lstrip('+').startswith(prefix)]
    require(values == [approval['approval_sha256']], 'Actual approval SHA CLI differs or is duplicated')


def _persist_native_failure(agent, stage, scene, before, error, expected=None):
    """Preserve the observed failed scene; never emit a successful receipt."""
    path = agent.output_dir / ('initialization_failed.rank' + str(agent.global_rank) +
                              '.' + str(time.time_ns()) + '.json')
    payload = {'status': 'FAILED_NATIVE_INITIALIZATION', 'stage': stage,
               'rank': agent.global_rank, 'frame': agent.frame, 'optimizer_step': agent.optimizer_step,
               'frame_since_restart': agent.vec_env.env.frame_since_restart,
               'contract_sha256': CONTRACT_SHA256, 'error': repr(error),
               'expected': expected, 'before': before, 'after': visible_snapshot(agent),
               'runtime_observed_scene': scene, 'historical_full_state_available': False}
    with path.open('x') as stream:
        json.dump(payload, stream)
    return path


def begin_loader_check(agent, approval):
    """Fail before loading/sampling if the approved launch/scene differs."""
    require(agent.frame == agent.optimizer_step == agent.epoch_num == 0,
            'Quality initialization requires the approved fresh/cache-hit protocol')
    require(not agent.training_initialized, 'Quality loader cannot start from resumed learner state')
    require(agent.num_actors == 6144 and agent.world_size == 4, 'Initialization population differs')
    report = approval['initialization']
    for source in report['production_files']:
        require(digest(source['path']) == source['sha256'], 'Approved startup source/dependency changed')
    launch_path = agent.output_dir / ('launch.rank' + str(agent.global_rank) + '.json')
    expected_launch = report['expected_launches'][agent.global_rank]
    require(digest(expected_launch['path']) == expected_launch['sha256'], 'Expected launch evidence changed')
    actual_launch = json.loads(launch_path.read_text())
    verify_launch_approval(actual_launch, approval)
    require(canonical_launch(actual_launch) == json.loads(Path(expected_launch['path']).read_text()),
            'Actual full per-rank launch differs from reviewed protocol')
    expected_seed = report['rank_events'][agent.global_rank]['runner_effective_seed']
    require(torch.initial_seed() == expected_seed, 'Actual RNG initialization seed differs from launch trace')
    before = visible_snapshot(agent)
    scene = capture_scene(agent.vec_env.env)
    expected = report['expected_scenes'][agent.global_rank]
    try:
        require(digest(expected['path']) == expected['sha256'], 'Expected scene provenance changed')
        expected_scene = json.loads(Path(expected['path']).read_text())
        validate_scene(agent.vec_env.env, scene, expected_scene)
    except Exception as error:
        _persist_native_failure(agent, 'before_reference_loader', scene, before, error,
                                {'scene': expected, 'approval_sha256': approval['approval_sha256']})
        raise
    # Freeze native observations before any data-conditioned command or update.
    output = agent.output_dir / ('initialization.rank' + str(agent.global_rank) + '.json')
    require(not output.exists(), 'Preserve earlier initialization record; do not overwrite a fresh-run identity')
    return {'before': before, 'scene_sha256': state_digest(scene), 'path': output,
            'approval_sha256': approval['approval_sha256']}


def finish_loader_check(agent, check, reference, reset_bank):
    scene_after = capture_scene(agent.vec_env.env)
    try:
        require(state_digest(scene_after) == check['scene_sha256'], 'Quality loading changed native scene properties')
        after = visible_snapshot(agent)
        require(after == check['before'], 'Quality loading changed non-intervention visible state/RNG')
        original = torch.load(reset_bank, map_location='cpu')
        require(state_digest(agent.vec_env.env.reference_reset_bank) == state_digest(original['resets']),
                'Loaded reset bank values differ from immutable original')
    except Exception as error:
        _persist_native_failure(agent, 'after_reference_loader', scene_after, check['before'], error,
                                {'scene_sha256': check['scene_sha256'], 'approval_sha256': check['approval_sha256']})
        raise
    report = {'status': 'PASS_CURRENT_LOADER_NO_SIDE_EFFECTS', 'contract_sha256': CONTRACT_SHA256,
              'historical_full_state_available': False, 'rank': agent.global_rank,
              'before': check['before'], 'after': after, 'scene_sha256': check['scene_sha256'],
              'runtime_observed_scene': scene_after,
              'approval_sha256': check['approval_sha256'], 'cache_sha256': digest(reference),
              'reset_sha256': digest(reset_bank)}
    check['path'].write_text(json.dumps(report, indent=2))


def record_launch_protocol(cfg, resolved_train, runner):
    """Record the actual per-rank configuration after Runner seed setup."""
    import os
    import sys
    resolved = configuration_container(cfg)
    require(resolved['checkpoint'] == '' and not resolved['test'], 'Expected fresh training launch')
    require(resolved['wandb_activate'] is True, 'Preserve the actual v4 W&B startup branch')
    require('seed=0' in sys.argv[1:], 'Reuse actual v4 CLI seed, not the saved rank-3 seed')
    require(resolved['task']['env'].get('validateReferenceInitialization') is True,
            'Full scene initialization recording is required')
    directory = Path(resolved_train['params']['config']['train_dir']) / resolved_train['params']['config']['name']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('launch.rank' + str(runner.global_rank) + '.json')
    require(not path.exists(), 'Preserve the original per-rank launch evidence')
    payload = {'schema_version': 1, 'rank': runner.global_rank, 'world_size': runner.world_size,
               'configuration': resolved, 'argv': sys.argv[1:], 'runner_effective_seed': runner.seed,
               'resolved_train_seed_before_wandb': resolved_train['params']['seed'],
               'torch_initial_seed_before_scene': torch.initial_seed(),
               'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
               'contract_sha256': CONTRACT_SHA256}
    path.write_text(json.dumps(payload, indent=2))
