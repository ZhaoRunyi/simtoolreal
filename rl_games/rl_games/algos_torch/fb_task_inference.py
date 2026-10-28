"""Replay reward inference and phase commands; no simulator or training mutation.

The I-BFM reward_inference / reward_wr_inference interfaces motivate these
operations. SimToolReal's own rewards and its pre/post-lift phase boundary are
used here; no humanoid-specific phase controller is copied.
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import torch

from rl_games.algos_torch.fb_networks import project_task


reward_components = (
    'fingertip_delta_rew', 'hand_delta_penalty', 'lifting_rew', 'lift_bonus_rew',
    'keypoint_rew', 'kuka_actions_penalty', 'hand_actions_penalty', 'bonus_rew',
    'object_lin_vel_penalty', 'object_ang_vel_penalty',
)
stage_names = ('grasp_lift', 'track', 'full')


def stage_rewards(components):
    """Keep original scales, clipping, historical deltas and event bonuses."""
    if components.ndim != 2 or components.shape[1] != len(reward_components):
        raise ValueError('Unexpected original reward component schema')
    prelift = components[:, :4].sum(-1)
    tracking = components[:, 4] + components[:, 7]
    penalties = components[:, [5, 6, 8, 9]].sum(-1)
    return torch.stack((prelift + penalties, tracking + penalties, components.sum(-1)), -1)


@torch.no_grad()
def mixed_reward_commands(features, components, samples_per_source, evidence=None):
    """Equal-population signed reward projection; return a gate, never a fallback."""
    if len(features) != 2 * samples_per_source or len(components) != len(features):
        raise ValueError('Mixed reward inference requires equal PPO/online counts')
    features, components = features.float(), components.float()
    rewards = stage_rewards(components)
    support = torch.stack((components[:, :4].clamp_min(0).sum(-1),
                           components[:, [4, 7]].clamp_min(0).sum(-1),
                           components[:, [0, 2, 3, 4, 7]].clamp_min(0).sum(-1)), -1)
    contribution_ppo = rewards[:samples_per_source].T @ features[:samples_per_source] / len(features)
    contribution_online = rewards[samples_per_source:].T @ features[samples_per_source:] / len(features)
    vectors = contribution_ppo + contribution_online
    if evidence is not None:
        evidence.update(ppo_contribution=contribution_ppo, online_contribution=contribution_online)
    norms = vectors.norm(dim=-1)
    valid = (torch.isfinite(features).all() & torch.isfinite(components).all()
             & torch.isfinite(vectors).all() & torch.isfinite(norms).all() & (norms >= 1e-8).all()
             & (support.sum(0) > 0).all())
    statistics = {}
    for index, name in enumerate(stage_names):
        mass = rewards[:, index].abs()
        statistics[name + '/norm'] = norms[index]
        statistics[name + '/reward_mean'] = rewards[:, index].mean()
        statistics[name + '/positive_task_support'] = support[:, index].sum()
        statistics[name + '/absolute_reward_ess'] = mass.sum().square() / mass.square().sum().clamp_min(1e-20)
        for source, section, contribution in (
            ('ppo', slice(None, samples_per_source), contribution_ppo),
            ('online', slice(samples_per_source, None), contribution_online),
        ):
            statistics[name + '/' + source + '_positive_task_count'] = (support[section, index] > 0).float().sum()
            statistics[name + '/' + source + '_positive_task_support'] = support[section, index].sum()
            statistics[name + '/' + source + '_vector_norm'] = contribution[index].norm()
            statistics[name + '/' + source + '_direction_contribution'] = (
                contribution[index] * vectors[index]).sum() / norms[index].square().clamp_min(1e-20)
    for index, name in ((3, 'lift_bonus'), (7, 'success_bonus')):
        statistics[name + '/positive_count'] = (components[:, index] > 0).float().sum()
        for source, section in (('ppo', slice(None, samples_per_source)),
                                ('online', slice(samples_per_source, None))):
            statistics[name + '/' + source + '_positive_count'] = (components[section, index] > 0).float().sum()
            statistics[name + '/' + source + '_positive_mass'] = components[section, index].clamp_min(0).sum()
    # Invalid directions are returned for evidence only. Caller synchronizes
    # the validity gate across ranks and never uses them to control the policy.
    return project_task(vectors), statistics, valid


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_phase_commands(commands, lifted, mode):
    if commands.shape[0] != len(stage_names):
        raise ValueError('Command bank must contain grasp_lift, track and full')
    if mode == 'phased':
        return commands[lifted.long()]
    if mode not in stage_names:
        raise ValueError('Unknown phase: ' + mode)
    return commands[stage_names.index(mode)].expand(lifted.numel(), -1)


@torch.no_grad()
def infer_command_bank(backward, normalizer, paths, frame, optimizer_step, device,
                       method='mean', temperature=0.1, batch_size=2048, world_size=4):
    """Encode frozen, uniform valid-replay samples with the matching checkpoint.

    The pool stores next states and their observed transition rewards. Original
    progress/bonus rewards depend on episode history, so this is a conditional
    reward projection, not an exact arbitrary state-only reward reconstruction.
    """
    if method not in ('mean', 'reward_weighted') or temperature <= 0 or batch_size <= 0:
        raise ValueError('Invalid inference method, temperature or batch size')
    if len(paths) != world_size:
        raise ValueError('Expected one matching inference pool for every training rank')
    states, components, ranks, sources = [], [], set(), []
    for path in paths:
        pool = torch.load(path, map_location='cpu')
        if pool['frame'] != frame or pool['optimizer_step'] != optimizer_step:
            raise ValueError('Inference pool and checkpoint counters do not match')
        if pool['world_size'] != world_size or pool['rank'] in ranks:
            raise ValueError('Duplicate rank or inconsistent inference world size')
        if pool['schema_version'] != 1 or tuple(pool['reward_names']) != reward_components:
            raise ValueError('Unsupported replay inference schema')
        count = pool['sample_count']
        if len(pool['next_state']) != count or len(pool['reward_components']) != count:
            raise ValueError('Inference sample count mismatch')
        ranks.add(pool['rank'])
        states.append(pool['next_state'])
        components.append(pool['reward_components'])
        sources.append({'path': str(Path(path).resolve()), 'sha256': checkpoint_digest(path),
                        'rank': pool['rank'], 'population': pool['population'], 'samples': count})
    if ranks != set(range(world_size)):
        raise ValueError('Missing training rank')
    states, components = torch.cat(states), torch.cat(components)
    if not len(states) or not torch.isfinite(states).all() or not torch.isfinite(components).all():
        raise ValueError('Empty or nonfinite inference data')
    rewards = stage_rewards(components)
    # A nonzero penalty vector alone must not masquerade as a manipulation task.
    support = torch.stack((components[:, :4].clamp_min(0).sum(-1),
                           components[:, [4, 7]].clamp_min(0).sum(-1),
                           components[:, [0, 2, 3, 4, 7]].clamp_min(0).sum(-1)), -1)
    if (support.sum(0) <= 0).any():
        missing = [stage_names[i] for i in range(3) if support[:, i].sum() <= 0]
        raise ValueError('Replay has no positive task evidence for phases: ' + str(missing))
    weights = torch.full_like(rewards, 1. / len(rewards))
    if method == 'reward_weighted':
        weights = torch.softmax(rewards / temperature, dim=0)
    # I-BFM WR multiplies BOTH reward and softmax weight; never softmax alone.
    coefficients = rewards * weights
    normalizer.eval()
    backward.eval()
    vector = None
    for start in range(0, len(states), batch_size):
        features = backward(normalizer(states[start:start + batch_size].to(device)))
        partial = coefficients[start:start + batch_size].to(device).T @ features
        vector = partial if vector is None else vector + partial
    norms = vector.norm(dim=-1)
    if not torch.isfinite(vector).all() or (norms < 1e-8).any():
        raise ValueError('Reward inference produced a zero or nonfinite task direction')
    statistics = {}
    for index, name in enumerate(stage_names):
        mass = coefficients[:, index].abs()
        statistics[name] = {
            'reward_mean': float(rewards[:, index].mean()),
            'reward_min': float(rewards[:, index].min()),
            'reward_max': float(rewards[:, index].max()),
            'positive_task_samples': int((support[:, index] > 0).sum()),
            'effective_samples_abs_reward': float(mass.sum().square() / mass.square().sum().clamp_min(1e-20)),
            'unprojected_norm': float(norms[index]),
        }
    return {'schema_version': 1, 'commands': project_task(vector).cpu(),
            'stage_names': list(stage_names), 'frame': frame, 'optimizer_step': optimizer_step,
            'method': method, 'temperature': temperature, 'sample_count': len(states),
            'sources': sources, 'statistics': statistics,
            'reward_definition': 'original scaled transition components; phase-specific task terms plus original action/velocity penalties'}


def main():
    from rl_games.algos_torch.fb_cpr_networks import BackwardMap, BatchNormalizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pool-glob', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--method', choices=('mean', 'reward_weighted'), default='mean')
    parser.add_argument('--temperature', type=float, default=.1)
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    options = checkpoint['options']
    normalizer = BatchNormalizer(checkpoint['schema']['state']).to(args.device)
    normalizer.load_state_dict(checkpoint['state_normalizer'])
    backward = BackwardMap(options['backward_state_size'], options['latent_size'],
                           options.get('backward_state_indices')).to(args.device)
    backward.load_state_dict({key[len('backward.'):]: value for key, value in checkpoint['model'].items()
                             if key.startswith('backward.')})
    frame, step = checkpoint['frame'], checkpoint['optimizer_step']
    world_size = checkpoint['training_world_size']
    del checkpoint
    bank = infer_command_bank(backward, normalizer, sorted(glob.glob(args.pool_glob)),
                              frame, step, args.device, args.method, args.temperature,
                              args.batch_size, world_size)
    bank['checkpoint_sha256'] = checkpoint_digest(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, args.output)
    metadata = {key: value for key, value in bank.items() if key != 'commands'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
