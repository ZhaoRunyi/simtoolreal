"""Exact, deliberately narrow native-preparation expectation; no simulator calls."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from rl_games.algos_torch.fb_reference_audit import digest, require


ACTORS = ('robot', 'object', 'table_object', 'goal_object')
STAGE = 'frozen_physx_prepare_sim_one_dt_over_substeps_v1'
BASE = Path('/home/MLTL-INTERN/shaoyang/zvi/dex_fb_ws')
SOURCE_IDENTITIES = {
    str(BASE / 'SimToolReal-v13-good/isaacgymenvs/tasks/simtoolreal/env.py'):
        '947174f9ef467b1f9bbe3640087267e05aec610e429a5fda49a79cd5c49ded53',
    str(BASE / 'SimToolReal-v13-good/isaacgymenvs/tasks/base/vec_task.py'):
        '0f8b34b2560d44b97bf32d6f6e61faaa4c036781d1cccac2a69117a524da56ed',
    str(BASE / 'third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64/libcarb.gym.plugin.so'):
        'c1bef07dc97d13cb3b4691309ff9818d69439a444fa083492202eb37c5b352dd',
}
SUPPORTED_TASK_SHA256 = ('859617c919608775f4f704c39d9a4a1c93d4bc4a15cc41701442efc8537b8bd3', 'c74003bfaefd7f47f253b6cfacb771c9aa157761c48b06b4035f495b0a21dd70', '7e96ea960a1e2cc213d9a2e9545d6544b30f3008e8925c3834927d25b6be6758', 'c908044f4c44670c4d4d53e4031a2517f7f009fbaf651ec73a13c40fb224a2e5')

def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_supported_static(static):
    configuration = static['task_configuration']
    rank = configuration['rank']
    require(type(rank) is int and 0 <= rank < 4, 'Unsupported native preparation rank')
    require(json_digest(configuration) == SUPPORTED_TASK_SHA256[rank],
            'Unsupported native preparation configuration (including dt/substeps/gravity/grid)')
    require(static['environment_count'] == 6144 and len(static['assets']) == 1200,
            'Unsupported native preparation population')
    require(len(static['environments']) == 6144, 'Incomplete frozen actor mapping')
    for index, row in enumerate(static['environments']):
        require(row == {'environment': index, 'asset_index': index % 1200,
                        'actors': [{'name': label, 'actor_sim_index': 4 * index + offset}
                                   for offset, label in enumerate(ACTORS)]},
                'Unsupported source environment/asset/actor mapping')
    return configuration


def verify_sources():
    for path, expected in SOURCE_IDENTITIES.items():
        require(digest(path) == expected, 'Unsupported native preparation source or binary: ' + path)


def grid_origins(configuration):
    """Mirror native repeated scalar float32 addition, not multiplied doubles."""
    count = configuration['env']['numEnvs']
    spacing = configuration['env']['envSpacing']
    single = np.float32
    stride = single(single(spacing) - single(-spacing))
    columns = int(np.sqrt(count))
    result = np.zeros((count, 3), dtype='<f4')
    x = y = single(0)
    for index in range(count):
        result[index] = (x, y, 0)
        if (index + 1) % columns == 0:
            x = single(0)
            y = single(y + stride)
        else:
            x = single(x + stride)
    return result


def source_roots(configuration, centers):
    """Use only source creation expressions and parsed URDF inertial origins."""
    single = np.float32
    origins = grid_origins(configuration)
    robot = single([0, .8, 0])
    table = single([float(robot[0]), float(robot[1]) - .8,
                    float(robot[2]) + configuration['env']['tableResetZ']])
    obj = single([float(robot[0]), float(robot[1]) - .8,
                  float(robot[2]) + configuration['env']['tableResetZ'] + .25])
    goal = single(obj + single([-.35, -.06, .12]))
    goal[2] = single(float(goal[2]) - .04)
    sim = configuration['sim']
    interval = single(single(sim['dt']) / single(sim['substeps']))
    velocity = single(single(sim['gravity']) * interval)
    displacement = single(velocity * interval)
    roots = np.zeros((6144, len(ACTORS), 13), dtype='<f4')
    roots[:, :, 6] = 1
    for position, (label, pose) in enumerate(zip(ACTORS, (robot, obj, table, goal))):
        center = centers[np.arange(6144) % 1200] if label in ('object', 'goal_object') else np.zeros((6144, 3), dtype='<f4')
        world_center = single(single(origins + pose) + center)
        if label == 'object':
            world_center = single(world_center + displacement)
            roots[:, position, 7:10] = velocity
        roots[:, position, :3] = single(single(world_center - center) - origins)
    return roots, {'robot': robot.tolist(), 'object': obj.tolist(),
                   'table_object': table.tolist(), 'goal_object': goal.tolist()}


def generate_prepared_roots(static, asset_directory, provenance):
    """Pure offline generation: never accepts or reads runtime scene observations."""
    configuration = verify_supported_static(static)
    verify_sources()
    centers = []
    for asset in static['assets']:
        path = Path(asset_directory) / asset['name']
        require(digest(path) == asset['sha256'], 'Source URDF identity differs')
        links = ET.parse(path).getroot().findall('link')
        require(len(links) == 1, 'Only frozen one-link source assets are supported')
        origin = links[0].find('inertial/origin')
        require(origin is None or origin.attrib.get('rpy', '0 0 0') == '0 0 0', 'Unsupported inertial rotation')
        center = [0., 0., 0.] if origin is None else [float(value) for value in origin.attrib['xyz'].split()]
        require(len(center) == 3 and np.isfinite(center).all(), 'Invalid source center of mass')
        centers.append(center)
    centers = np.asarray(centers, dtype='<f4')
    roots, poses = source_roots(configuration, centers)
    require(digest(provenance['path']) == provenance['sha256'], 'Original expected-scene provenance changed')
    original = json.loads(Path(provenance['path']).read_text())
    require(original['exact_static_mapping'] == static, 'Original source scene mapping differs')
    return {'schema_version': 1, 'stage': STAGE, 'input_kind': 'source_configuration_and_urdf',
            'recipe_sha256': digest(__file__), 'source_identities': SOURCE_IDENTITIES,
            'configuration_sha256': json_digest(configuration), 'ordered_assets_sha256': json_digest(static['assets']),
            'original_expected_scene': provenance, 'actor_order': list(ACTORS),
            'dtype': '<f4', 'shape': list(roots.shape),
            'array_sha256': hashlib.sha256(roots.tobytes()).hexdigest(), 'values': roots.tolist(),
            'source_urdf_com': centers.tolist(), 'source_creation_positions': poses}


def load_prepared_roots(static, record):
    """Verify the approved offline array without changing the native scene."""
    configuration = verify_supported_static(static)
    verify_sources()
    require(record['schema_version'] == 1 and record['stage'] == STAGE,
            'Native preparation stage differs')
    require(record['input_kind'] == 'source_configuration_and_urdf', 'Native expectation provenance differs')
    require(record['recipe_sha256'] == digest(__file__), 'Native preparation recipe changed')
    require(record['source_identities'] == SOURCE_IDENTITIES, 'Native source binding differs')
    require(record['configuration_sha256'] == json_digest(configuration), 'Native task configuration binding differs')
    require(record['ordered_assets_sha256'] == json_digest(static['assets']), 'Native ordered asset binding differs')
    provenance = record['original_expected_scene']
    require(digest(provenance['path']) == provenance['sha256'], 'Original expected scene changed')
    require(json.loads(Path(provenance['path']).read_text())['exact_static_mapping'] == static,
            'Original expected scene mapping changed')
    require(record['actor_order'] == list(ACTORS) and record['shape'] == [6144, 4, 13] and record['dtype'] == '<f4',
            'Prepared native array layout differs')
    values = np.asarray(record['values'], dtype=np.float64)
    roots = np.asarray(record['values'], dtype='<f4')
    require(roots.shape == (6144, 4, 13) and np.isfinite(roots).all() and np.array_equal(values, roots),
            'Prepared native array is not exact finite float32')
    require(hashlib.sha256(roots.tobytes()).hexdigest() == record['array_sha256'], 'Prepared native array SHA differs')
    centers = np.asarray(record['source_urdf_com'], dtype='<f4')
    require(centers.shape == (1200, 3) and np.isfinite(centers).all(), 'Prepared source COM layout differs')
    predicted, poses = source_roots(configuration, centers)
    require(np.array_equal(predicted, roots) and poses == record['source_creation_positions'],
            'Prepared array does not follow frozen source recipe')
    return roots, centers
