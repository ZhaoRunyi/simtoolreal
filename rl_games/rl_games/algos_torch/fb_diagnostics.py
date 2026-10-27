"""Read-only FB diagnostics on a coordinated checkpoint and its real replay.

Save the sampled transitions and per-sample outputs so physical state, reward,
representation, command response and critic predictions remain joinable.
"""

import argparse
import json
from pathlib import Path

import torch

from rl_games.algos_torch.fb_cpr_networks import (
    CPRModel, BatchNormalizer, TransitionReplay, TemporalReplay, make_state_normalizer,
)
from rl_games.algos_torch.fb_networks import project_task, successor_loss
from rl_games.algos_torch.fb_task_inference import checkpoint_digest


def summarize(values):
    values = values.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic values')
    return {'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max()),
            'median': float(values.median()), 'p95': float(torch.quantile(values.flatten(), .95))}


@torch.no_grad()
def representation_report(features, rewards):
    """Fit only on the first half; report reward reconstruction on held-out rows."""
    count = len(features) // 2
    train, test = features[:count].double(), features[count:].double()
    covariance = train.T @ train / count
    eigenvalues = torch.linalg.eigvalsh(covariance)
    inverse = torch.linalg.pinv(covariance, rtol=1e-6)
    command = train.T @ rewards[:count].double() / count
    predicted = test @ inverse @ command
    actual = rewards[count:].double()
    return {'effective_rank': float(covariance.trace().square() / covariance.square().sum()),
            'eigenvalues': eigenvalues.cpu().tolist(),
            'reward_heldout_rmse': float((predicted - actual).square().mean().sqrt()),
            'reward_heldout_std': float(actual.std()),
            'positive_samples': int((actual > 0).sum()),
            'positive_prediction_mean': float(predicted[actual > 0].mean()) if (actual > 0).any() else None,
            'nonpositive_prediction_mean': float(predicted[actual <= 0].mean()) if (actual <= 0).any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1741)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    status = json.loads((args.run / 'status.json').read_text())
    if status['state'] not in ('trained', 'budget_stopped', 'iteration_limit'):
        raise ValueError('Diagnostics require a coordinated final save, not a periodic model')
    path = args.run / 'nn/last.pt'
    digest = checkpoint_digest(path)
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint['frame'] != status['environment_steps'] or checkpoint['optimizer_step'] != status['optimizer_step']:
        raise ValueError('Checkpoint and trainer status do not match')
    options, schema = checkpoint['options'], checkpoint['schema']
    model = CPRModel(schema['observation'], schema['state'], schema['action'], options, {}).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    target = CPRModel(schema['observation'], schema['state'], schema['action'], options, {}).to(args.device).eval()
    target.load_state_dict(checkpoint['target'])
    normalizers = {}
    for name, dimension in [('state', schema['state']), ('observation', schema['observation']), ('action', schema['action'])]:
        normalizer = make_state_normalizer(dimension, options) if name == 'state' else BatchNormalizer(dimension)
        normalizer = normalizer.to(args.device).eval()
        normalizer.load_state_dict(checkpoint[name + '_normalizer'])
        normalizers[name] = normalizer
    frame, optimizer_step = checkpoint['frame'], checkpoint['optimizer_step']
    paths = sorted(path.parent.glob('last.rank*.replay.pt'))
    world_size = checkpoint.get('training_world_size', options.get('inference_world_size', 4))
    if len(paths) != world_size or args.samples < 2 * world_size:
        raise ValueError('Missing replay shards or insufficient diagnostic samples')
    del checkpoint
    sources, samples = [], []
    for rank, replay_path in enumerate(paths):
        signature = replay_path.stat()
        saved = torch.load(replay_path, map_location='cpu')
        replay_type = TemporalReplay if options.get('temporal_replay', False) else TransitionReplay
        replay = replay_type(1, 1, 1, 'cpu')
        replay.load_state_dict(saved['replay'])
        count = args.samples // world_size + int(rank < args.samples % world_size)
        batch = replay.sample(count, generator=torch.Generator().manual_seed(1741 + rank))
        batch['rank'] = torch.full((count, 1), rank)
        samples.append(batch)
        sources.append({'path': str(replay_path), 'bytes': signature.st_size, 'mtime_ns': signature.st_mtime_ns,
                        'population': sum(len(row) for row in replay.valid_indices[:replay.size])})
        del replay, saved
    batch = {name: torch.cat([sample[name] for sample in samples]) for name in samples[0]}
    if not batch['valid'].bool().all():
        raise ValueError('Invalid reset rows in diagnostic sample')
    torch.save(batch, args.output / 'sample.pt')
    outputs = {}
    gradient_rows = []
    evaluation_command = None
    for bank_path in (args.run / 'nn/reward_inference/full_reward.pt', args.run.parent / 'commands/mean.pt'):
        if bank_path.exists():
            bank = torch.load(bank_path, map_location='cpu')
            bank_digest = bank.get('checkpoint_sha256', bank.get('identity', {}).get('checkpoint_sha256'))
            if bank_digest != digest:
                raise ValueError('Evaluation command bank belongs to a different checkpoint')
            evaluation_command = bank['command'] if 'command' in bank else bank['commands'][bank['stage_names'].index('full')]
    model.requires_grad_(False)
    target.requires_grad_(False)
    for start in range(0, len(batch['state']), args.batch_size):
        data = {name: value[start:start + args.batch_size].to(args.device) for name, value in batch.items()}
        with torch.no_grad():
            state = normalizers['state'](data['state'])
            next_state = normalizers['state'](data['next_state'])
            history = normalizers['observation'](data['history'].reshape(-1, schema['observation'])).reshape(len(state), -1)
            history = torch.cat((history, normalizers['action'](data['previous_action'])), -1)
            next_history = normalizers['observation'](data['next_history'].reshape(-1, schema['observation'])).reshape(len(state), -1)
            next_history = torch.cat((next_history, normalizers['action'](data['action'])), -1)
            critic_state = model.critic_state(state, history)
            next_critic = model.critic_state(next_state, next_history)
            features = model.backward(next_state)
            current_features = model.backward(state)
            commands = {'behavior': data['command'], 'random': project_task(torch.randn_like(features)),
                        'replay_goal': features, 'current_goal': current_features}
            commands['near_behavior'] = project_task(data['command'] + .01 * project_task(torch.randn_like(features)))
            if evaluation_command is not None:
                commands['evaluation_full'] = evaluation_command.to(args.device).expand(len(state), -1)
            rows = {'backward': features}
            for name, command in commands.items():
                prompt = model.actor_prompt(state, command)
                action = model.actor(history, prompt)
                prediction = model(critic_state, action, command)
                rows[name + '/action'] = action
                rows[name + '/command'] = command
                rows[name + '/prompt_norm'] = prompt.norm(dim=-1)
                rows[name + '/q_fb'] = (prediction * command).sum(-1).T
                rows[name + '/q_style'] = model.style(critic_state, action, command).T
                rows[name + '/q_aux'] = model.auxiliary(critic_state, action, command).T
            command = data['command']
            next_action = model.actor(next_history, model.actor_prompt(next_state, command))
            prediction = model(critic_state, data['action'], command)
            target_prediction = target(next_critic, next_action, command)
            target_features = target.backward(next_state)
            loss, orthogonal = successor_loss(prediction, features, target_prediction, target_features,
                                              data['discount'], data['valid'])
            rows['fb_loss_batch'] = loss.expand(len(state))
            rows['orthogonal_batch'] = orthogonal.expand(len(state))
            rows['behavior/q_recorded_action'] = (prediction * command).sum(-1).T
        # Gradient with respect to the normalized actor output, not parameters.
        action = rows['behavior/action'].detach().requires_grad_(True)
        objectives = [(model(critic_state, action, command) * command).sum(-1).min(0).values,
                      model.style(critic_state, action, command).min(0).values,
                      model.auxiliary(critic_state, action, command).min(0).values]
        gradients = [torch.autograd.grad(value.sum(), action, retain_graph=index < 2)[0]
                     for index, value in enumerate(objectives)]
        gradient_rows.append(torch.stack(gradients, dim=1).cpu())
        for name, value in rows.items():
            outputs.setdefault(name, []).append(value.detach().cpu())
    outputs = {name: torch.cat(values) for name, values in outputs.items()}
    outputs['action_value_gradients'] = torch.cat(gradient_rows)
    torch.save(outputs, args.output / 'outputs.pt')
    report = {'schema_version': 1, 'checkpoint': str(path), 'checkpoint_sha256': digest,
              'frame': frame, 'optimizer_step': optimizer_step, 'options': options, 'sources': sources,
              'sample_count': args.samples, 'sampling': 'equal-size uniform valid samples per rank; rank populations reported',
              'representation': representation_report(outputs['backward'], batch['reward'].flatten()),
              'commands': {}, 'environment_groups': {}, 'evaluation': []}
    for name in commands:
        action = outputs[name + '/action']
        report['commands'][name] = {
            'arm_saturation': float((action[:, :7].abs() > .95).float().mean()),
            'hand_saturation': float((action[:, 7:].abs() > .95).float().mean()),
            'delta_from_behavior': summarize((action - outputs['behavior/action']).norm(dim=-1)),
            'prompt_norm': summarize(outputs[name + '/prompt_norm']),
            'q_fb': summarize(outputs[name + '/q_fb'].min(-1).values),
            'ensemble_disagreement': summarize(outputs[name + '/q_fb'].std(-1)),
        }
    distance = batch['next_observation'][:, 125:137].reshape(-1, 4, 3).norm(dim=-1).amax(-1)
    for name, mask in {'lifted': batch['next_state'][:, 158] > .5,
                       'not_lifted': batch['next_state'][:, 158] <= .5,
                       'near_goal_5cm': distance < .05, 'far_goal': distance >= .05}.items():
        report['environment_groups'][name] = {'count': int(mask.sum()),
            'reward': summarize(batch['reward'][mask]) if mask.any() else None,
            'goal_distance': summarize(distance[mask]) if mask.any() else None}
    gradient = outputs['action_value_gradients']
    scale = outputs['behavior/q_fb'].min(-1).values.abs().mean()
    report['action_gradient'] = {'fb_norm': summarize(gradient[:, 0].norm(dim=-1)),
                                'style_norm': summarize(gradient[:, 1].norm(dim=-1)),
                                'aux_norm': summarize(gradient[:, 2].norm(dim=-1)),
                                'fb_outward_fraction': float(((gradient[:, 0] * outputs['behavior/action']) > 0).float().mean()),
                                'sample_fb_value_scale': float(scale),
                                'weighted_style_norm': summarize((gradient[:, 1] * scale * options['style_coefficient']).norm(dim=-1)),
                                'weighted_aux_norm': summarize((gradient[:, 2] * scale * options['auxiliary_coefficient']).norm(dim=-1))}
    report['actor_vs_recorded_q_gain'] = summarize(outputs['behavior/q_fb'].min(-1).values - outputs['behavior/q_recorded_action'].min(-1).values)
    report['local_command_sensitivity'] = summarize(
        (outputs['near_behavior/action'] - outputs['behavior/action']).norm(dim=-1)
        / (outputs['near_behavior/command'] - outputs['behavior/command']).norm(dim=-1).clamp_min(1e-8))
    report['physical_joint_limit_fraction'] = {
        'arm': float((batch['state'][:, :7].abs() > .95).float().mean()),
        'hand': float((batch['state'][:, 7:29].abs() > .95).float().mean())}
    features = torch.nn.functional.normalize(outputs['backward'], dim=-1)
    similarity = features @ features.T
    similarity.fill_diagonal_(-2)
    nearest = similarity.argmax(-1)
    torch.save({'nearest_index': nearest, 'cosine': similarity.max(-1).values,
                'goal_distance_difference': (distance - distance[nearest]).abs(),
                'reward_difference': (batch['reward'] - batch['reward'][nearest]).abs(),
                'lifted_disagrees': (batch['next_state'][:, 158] > .5) != (batch['next_state'][nearest, 158] > .5)},
               args.output / 'representation_neighbors.pt')
    metrics_path = args.run / 'metrics.jsonl'
    report['recent_training'] = [json.loads(line) for line in metrics_path.read_text().splitlines()[-10:]] if metrics_path.exists() else []
    for evaluation_path in sorted((args.run.parent / 'evaluation').rglob('evaluation.json')):
        evaluation = json.loads(evaluation_path.read_text())
        report['evaluation'].append({'path': str(evaluation_path), 'episodes': evaluation['episodes'],
                                     'mean_goal_fraction': evaluation['mean_goal_fraction'],
                                     'mean_return': evaluation['mean_return']})
    if checkpoint_digest(path) != digest:
        raise ValueError('Checkpoint changed during diagnostics')
    for source in sources:
        signature = Path(source['path']).stat()
        if signature.st_size != source['bytes'] or signature.st_mtime_ns != source['mtime_ns']:
            raise ValueError('Replay changed during diagnostics')
    report['interpretation_limits'] = ['Critic value gains and action gradients require rollout validation; not success evidence.',
                                      'Held-out reward projection diagnoses representation support, not policy optimality.',
                                      'Lifted is a historical flag and includes reference initialization.',
                                      'Legacy replay epoch association relies on coordinated-save trainer status.']
    temporary = args.output / 'report.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False, default=str))
    temporary.replace(args.output / 'report.json')
    print(json.dumps({'state': 'diagnosed', 'frame': frame, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
