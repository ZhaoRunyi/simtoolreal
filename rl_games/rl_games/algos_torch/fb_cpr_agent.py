# BFM-Zero / Meta FB-CPR architecture and loss adaptation.
# Reference components: CC BY-NC 4.0; see docs/licenses/bfm-zero-LICENSE.
"""FB-CPR with auxiliary critic, adapted to the existing SimToolReal interface.

BFM-Zero 05ad611 is the algorithm reference. Expert observations are collected
from the published SimToolReal PPO teacher, not synthesized from object goals.
"""

import copy
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn import functional
import yaml

from rl_games.algos_torch.fb_agent import FBAgent
from rl_games.algos_torch.fb_networks import project_task, successor_loss
from rl_games.algos_torch.fb_cpr_networks import (
    BatchNormalizer, CPRModel, ExpertSequences, RewardNormalizer, TransitionReplay,
)
from rl_games.algos_torch.model_builder import ModelBuilder


class FBcprAgent(FBAgent):
    model_class = CPRModel

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.model.initialize()
        if self.world_size > 1:
            for parameter in self.model.parameters():
                distributed.broadcast(parameter.data, 0)
        self.target.load_state_dict(self.model.state_dict())
        self.observation_normalizer = BatchNormalizer(self.observation_size).to(self.device).eval()
        self.state_normalizer = BatchNormalizer(self.state_size).to(self.device).eval()
        self.action_normalizer = BatchNormalizer(self.action_size).to(self.device).eval()
        self.reward_normalizer = RewardNormalizer().to(self.device)
        self.style_optimizer = torch.optim.Adam(self.model.style_maps.parameters(), lr=self.config['learning_rate'])
        self.discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=self.options['discriminator_learning_rate']
        )
        self.replay = TransitionReplay(
            math.ceil(self.options['replay_capacity'] / self.world_size), self.num_actors,
            self.config['horizon_length'], self.device,
        )
        self.z_buffer = torch.empty(self.options['z_buffer_size'], self.options['latent_size'], device=self.device)
        self.z_position = 0
        self.z_size = 0
        self.experts = None
        self.expert_report = None

    def _initialize_logging(self):
        super()._initialize_logging()
        if self.global_rank == 0 and self.expert_report is not None:
            import wandb
            if wandb.run is not None:
                summary = {'expert_collection_rank0/' + name: value
                           for name, value in self.expert_report.items() if isinstance(value, (int, float))}
                wandb.log({**summary, 'global_step': self.optimizer_step})

    def _checkpoint_extra(self):
        return {
            'style_optimizer': self.style_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'action_normalizer': self.action_normalizer.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict(),
            'z_buffer': self.z_buffer, 'z_position': self.z_position, 'z_size': self.z_size,
            'expert_report': self.expert_report,
        }

    def _restore_extra(self, checkpoint):
        for name in ('style_optimizer', 'discriminator_optimizer', 'action_normalizer', 'reward_normalizer'):
            getattr(self, name).load_state_dict(checkpoint[name])
        for name in ('z_buffer', 'z_position', 'z_size', 'expert_report'):
            setattr(self, name, checkpoint[name])

    @torch.no_grad()
    def _prepare_experts(self):
        path = self.output_dir / ('expert.rank' + str(self.global_rank) + '.pt')
        if path.exists():
            data = torch.load(path, map_location=self.device)
            self.experts = ExpertSequences(data['states'], data['valid'])
            self.expert_report = data['report']
            self.vec_env.env.reference_reset_bank = data['resets']
            return
        teacher_root = Path(self.options['teacher_path'])
        config = yaml.safe_load((teacher_root / 'config.yaml').read_text())
        if config['task']['env']['obsList'] != self.vec_env.env.obs_list:
            raise ValueError('Teacher observation ordering differs from the environment')
        teacher_params = config['train']['params']
        teacher = ModelBuilder().load(teacher_params).build({
            'actions_num': self.action_size, 'input_shape': (self.observation_size + 1,),
            'num_seqs': self.num_actors, 'value_size': 1,
            'normalize_value': teacher_params['config']['normalize_value'],
            'normalize_input': teacher_params['config']['normalize_input'],
            'type': 'extra_param', 'coef_ids': torch.linspace(50, 0, 6, device=self.device),
            'coef_id_idx': self.observation_size,
        }).to(self.device).eval()
        checkpoint = torch.load(teacher_root / 'model.pth', map_location=self.device)
        if 0 in checkpoint:
            checkpoint = checkpoint[0]
        teacher.load_state_dict(checkpoint['model'])
        if 'running_mean_std' in checkpoint:
            teacher.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        memory = [value.to(self.device) for value in teacher.get_default_rnn_state()]
        observations = {key: value.clone() for key, value in self.vec_env.reset().items()}
        selected = torch.randperm(self.num_actors, device=self.device)[:self.options['expert_environments']]
        states, validity, reset_snapshots = [], [], []
        previous_done = torch.ones(self.num_actors, dtype=torch.bool, device=self.device)
        episode_return = torch.zeros(self.num_actors, device=self.device)
        return_sum, episodes, successes, lifted = 0., 0, 0., 0.
        started = time.monotonic()
        for step in range(self.options['expert_steps']):
            inputs = torch.cat((observations['obs'], observations['obs'].new_full((self.num_actors, 1), 50.)), -1)
            result = teacher({'is_train': False, 'prev_actions': None, 'obs': inputs, 'rnn_states': memory})
            action, memory = result['mus'].clamp(-1, 1), result['rnn_states']
            next_observations, reward, done, info = self.vec_env.step(action)
            done = done.bool()
            # Invalid rows include the physical reset step and terminal boundary;
            # all expert slices therefore lie within a single physical episode.
            states.append(observations['states'][selected].half().clone())
            validity.append((~previous_done & ~done)[selected].clone())
            episode_return += reward.reshape(-1)
            episodes += int(done.sum())
            return_sum += float(episode_return[done].sum())
            successes += float(self.vec_env.env.successes[done].sum())
            lifted += float(self.vec_env.env.lifted_object.float().mean())
            episode_return.masked_fill_(done, 0)
            for value in memory:
                value[:, done] = 0
            if step % self.options['reference_stride'] == 0:
                environment = self.vec_env.env
                reset_snapshots.append({
                    'joint_position': environment.arm_hand_dof_pos.clone(),
                    'joint_velocity': environment.arm_hand_dof_vel.clone(),
                    'target': environment.prev_targets.clone(),
                    'object': environment.root_state_tensor[environment.object_indices].clone(),
                    'table': environment.root_state_tensor[environment.table_indices].clone(),
                    'lifted': environment.lifted_object.clone(),
                    'valid': (~previous_done & ~done).clone(),
                })
            previous_done = done
            observations = {key: value.clone() for key, value in next_observations.items()}
            if step % 100 == 0:
                self._status('expert_collection', expert_steps=step, episodes=episodes)
        data = {
            'states': torch.stack(states), 'valid': torch.stack(validity),
            'resets': {name: torch.stack([snapshot[name] for snapshot in reset_snapshots]) for name in reset_snapshots[0]},
            'report': {'teacher': str(teacher_root / 'model.pth'), 'steps': self.options['expert_steps'],
                       'stored_environments': len(selected), 'episodes': episodes,
                       'mean_return': return_sum / max(1, episodes),
                       'mean_successes': successes / max(1, episodes),
                       'lifted_fraction': lifted / self.options['expert_steps'],
                       'seconds': time.monotonic() - started},
        }
        temporary = path.with_suffix('.tmp')
        torch.save(data, temporary)
        temporary.replace(path)
        self.experts = ExpertSequences(data['states'], data['valid'])
        self.expert_report = data['report']
        self.vec_env.env.reference_reset_bank = data['resets']
        # Validate both interfaces before starting the formal learner.
        self.experts.sample(1, self.options['expert_sequence_length'])
        self.experts.sample(1, self.options['tracking_length'])
        (path.with_suffix('.json')).write_text(json.dumps(self.expert_report, indent=2))
        del teacher, checkpoint, states, memory
        torch.cuda.empty_cache()

    def train(self):
        self._prepare_experts()
        return super().train()

    @torch.no_grad()
    def _sample_commands(self, count):
        if getattr(self, 'z_size', 0):
            return self.z_buffer[torch.randint(self.z_size, (count,), device=self.device)].clone()
        return project_task(torch.randn(count, self.options['latent_size'], device=self.device))

    @torch.no_grad()
    def _cache_commands(self, commands):
        commands = commands[-len(self.z_buffer):]
        indices = (torch.arange(len(commands), device=self.device) + self.z_position) % len(self.z_buffer)
        self.z_buffer[indices] = commands
        self.z_position = (self.z_position + len(commands)) % len(self.z_buffer)
        self.z_size = min(len(self.z_buffer), self.z_size + len(commands))

    def _initialize_rollout(self):
        self.vec_env.env.reset_buf.fill_(1)
        self.observations = {key: value.clone() for key, value in self.vec_env.reset().items()}
        self.history = torch.zeros(self.num_actors, self.options['history_steps'], self.observation_size, device=self.device)
        self.previous_action = torch.zeros(self.num_actors, self.action_size, device=self.device)
        self.rollout_command = self._sample_commands(self.num_actors)
        self.command_age = torch.zeros(self.num_actors, dtype=torch.long, device=self.device)
        self.reset_before = torch.ones(self.num_actors, dtype=torch.bool, device=self.device)
        self.history_reset = self.reset_before.clone()
        self.episode_reward = torch.zeros(self.num_actors, device=self.device)
        # Parent checkpoint format keeps these legacy fields; v2 reconstructs
        # its finite history from new episodes after resume.
        self.actor_state = ()
        self.previous_block = self.previous_initial_state = None
        self.tracking_indices = torch.arange(int(self.num_actors * self.options['tracking_fraction']), device=self.device)
        self.tracking_prompts = None
        self.rollout_step = 0

    def _history_input(self, history, previous_action):
        normalized = self.observation_normalizer(history.reshape(-1, self.observation_size)).reshape(history.shape[0], -1)
        return torch.cat((normalized, self.action_normalizer(previous_action)), dim=-1)

    @torch.no_grad()
    def _rollout_prompts(self):
        refresh = self.reset_before | (self.command_age >= self.options['command_steps'])
        candidates = self._sample_commands(self.num_actors)
        self.rollout_command = torch.where(refresh[:, None], candidates, self.rollout_command)
        self.command_age = torch.where(refresh, 0, self.command_age)
        length = self.options['tracking_length']
        if self.rollout_step % length == 0:
            self.tracking_indices = torch.randperm(self.num_actors, device=self.device)[:len(self.tracking_indices)]
            experts = self.experts.sample(len(self.tracking_indices), length)
            encoded = self.model.backward(self.state_normalizer(experts.flatten(0, 1))).reshape(len(experts), length, -1)
            window = self.options['expert_sequence_length']
            self.tracking_prompts = torch.stack([
                project_task(encoded[:, index:min(index + window, length)].mean(1)) for index in range(length)
            ], dim=1)
        self.rollout_command[self.tracking_indices] = self.tracking_prompts[:, self.rollout_step % length]

    @torch.no_grad()
    def collect(self):
        records, episode_returns = [], []
        step_rewards, episode_rewards = {}, {}
        environment = self.vec_env.env
        for _ in range(self.config['horizon_length']):
            self._rollout_prompts()
            self.history[self.history_reset] = 0
            self.previous_action[self.history_reset] = 0
            self.history = torch.cat((self.history[:, 1:], self.observations['obs'][:, None]), dim=1)
            history = self.history.clone()
            previous_action = self.previous_action.clone()
            action = self.model.actor.sample(self._history_input(history, previous_action), self.rollout_command,
                                             self.options['exploration_std'])
            current = {key: value.clone() for key, value in self.observations.items()}
            observations, reward, done, info = self.vec_env.step(action)
            observations = {key: value.clone() for key, value in observations.items()}
            done = done.bool()
            next_history = torch.cat((history[:, 1:], observations['obs'][:, None]), dim=1)
            records.append({
                'observation': current['obs'], 'state': current['states'],
                'next_observation': observations['obs'], 'next_state': observations['states'],
                'history': history.flatten(1), 'next_history': next_history.flatten(1),
                'previous_action': previous_action, 'action': action,
                'command': self.rollout_command.clone(), 'reward': reward.reshape(-1, 1).clone(),
                'auxiliary_reward': sum(info['episode_cumulative'][name] for name in self.options['auxiliary_reward_terms']).reshape(-1, 1).clone(),
                'discount': self.config['gamma'] * (~done).float().unsqueeze(-1),
                'valid': (~self.reset_before).unsqueeze(-1).clone(),
            })
            self.episode_reward += reward.reshape(-1)
            episode_returns.append(self.episode_reward[done])
            for name, value in info.get('episode_cumulative', {}).items():
                step_rewards.setdefault(name, []).append(value.mean().detach())
            for name, value in info.get('rewards_episode', {}).items():
                episode_rewards.setdefault(name, []).append(value[done].sum().detach())
            self.episode_reward.masked_fill_(done, 0)
            # Reset history again at the first observation of the new physical
            # episode, after the environment's delayed-reset transition.
            self.history_reset = self.reset_before | done
            self.reset_before = done
            self.observations = observations
            self.previous_action = action
            self.command_age += 1
            self.rollout_step += 1
        block = {name: torch.stack([record[name] for record in records]) for name in records[0]}
        self.replay.add(block)
        self.frame += self.num_actors * self.config['horizon_length'] * self.world_size
        returns = torch.cat(episode_returns)
        metrics = {'reward_step/' + name: torch.stack(values).mean() for name, values in step_rewards.items()}
        metrics.update({'reward_episode_sum/' + name: torch.stack(values).sum() for name, values in episode_rewards.items()})
        metrics.update({
            'rollout/reward': block['reward'].mean(),
            'rollout/completed_episodes': reward.new_tensor(len(returns)),
            'rollout/episode_return_sum': returns.sum(),
            'rollout/episode_return': returns.mean() if len(returns) else reward.new_zeros(()),
            'task/successes_per_completed_episode': environment.prev_episode_successes.float().mean(),
            'task/closest_keypoint_distance': environment.prev_episode_closest_keypoint_max_dist.float().mean(),
            'task/success_tolerance': reward.new_tensor(environment.success_tolerance),
            'task/lifted_fraction': environment.lifted_object.float().mean(),
            'replay/capacity_transitions_global': reward.new_tensor(self.replay.capacity * self.num_actors * self.config['horizon_length'] * self.world_size),
            'command/z_buffer_size': reward.new_tensor(self.z_size),
        })
        return metrics

    def _discriminator_update(self, expert_state, expert_command, state, rollout_command):
        real = self.model.discriminate(expert_state, expert_command)
        fake = self.model.discriminate(state, rollout_command)
        alpha = torch.rand(len(state), 1, device=self.device)
        mixed_state = (alpha * expert_state + (1 - alpha) * state).requires_grad_(True)
        mixed_command = (alpha * expert_command + (1 - alpha) * rollout_command).requires_grad_(True)
        logits = self.model.discriminate(mixed_state, mixed_command)
        gradients = torch.autograd.grad(logits.sum(), (mixed_state, mixed_command), create_graph=True)
        penalty = (torch.cat(gradients, dim=-1).norm(dim=-1) - 1).square().mean()
        loss = (functional.softplus(-real) + functional.softplus(fake)).mean() + self.options['discriminator_penalty'] * penalty
        self._synchronize_step(loss, self.discriminator_optimizer, self.model.discriminator.parameters())
        return {'loss/discriminator': loss.detach(), 'loss/discriminator_gradient_penalty': penalty.detach(),
                'discriminator/expert_logit': real.detach().mean(), 'discriminator/online_logit': fake.detach().mean()}

    def update(self):
        batch = self.replay.sample(self.config['minibatch_size'])
        for normalizer, name in ((self.observation_normalizer, 'observation'), (self.state_normalizer, 'state')):
            normalizer.train()
            normalizer(batch[name])
            normalizer(batch['next_' + name])
            normalizer.eval()
        self.action_normalizer.train()
        self.action_normalizer(batch['previous_action'])
        self.action_normalizer(batch['action'])
        self.action_normalizer.eval()
        state, next_state = self.state_normalizer(batch['state']), self.state_normalizer(batch['next_state'])
        history = self._history_input(batch['history'].reshape(-1, self.options['history_steps'], self.observation_size), batch['previous_action'])
        next_history = self._history_input(batch['next_history'].reshape(-1, self.options['history_steps'], self.observation_size), batch['action'])
        critic_state = torch.cat((state, history), -1)
        next_critic_state = torch.cat((next_state, next_history), -1)
        count, length = len(state), self.options['expert_sequence_length']
        with torch.no_grad():
            expert_sequence = self.experts.sample(count // length, length + 1)
            expert_state = self.state_normalizer(expert_sequence[:, :-1].flatten(0, 1))
            expert_next_state = self.state_normalizer(expert_sequence[:, 1:].flatten(0, 1))
            expert_features = self.model.backward(expert_next_state).reshape(-1, length, self.options['latent_size'])
            expert_command = project_task(expert_features.mean(1)).repeat_interleave(length, 0)
        # Discriminator uses actual behavior commands, before learner relabeling.
        metrics = self._discriminator_update(expert_state, expert_command, state, batch['command'])
        with torch.no_grad():
            command = project_task(torch.randn(count, self.options['latent_size'], device=self.device))
            selector = torch.rand(count, 1, device=self.device)
            permutation = torch.randperm(count, device=self.device)
            goals = project_task(self.model.backward(next_state[permutation]))
            command = torch.where(selector < .2, goals, command)
            command = torch.where((selector >= .2) & (selector < .8), expert_command[permutation], command)
            self._cache_commands(command)
            relabel = torch.rand(count, 1, device=self.device) < self.options['relabel_probability']
            command = torch.where(relabel, command, batch['command'])
            next_action = self.model.actor.sample(next_history, command, self.options['exploration_std'], self.options['target_noise_clip'])
            target_forward = self.target(next_critic_state, next_action, command)
            target_backward = self.target.backward(next_state)
        forward = self.model(critic_state, batch['action'], command)
        backward = self.model.backward(next_state)
        discount = (batch['discount'] > 0).float() * self.config['gamma']
        # Exact Gram algebra, with full FP32 matmul only for this cancellation-
        # sensitive reduction. Network matmuls retain the configured TF32 mode.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        fb_loss, orthogonal = successor_loss(forward, backward, target_forward, target_backward, discount, batch['valid'].squeeze(-1))
        # Official orth loss is half the v1 convention.
        representation_loss = fb_loss + .5 * self.options['orthogonal_coefficient'] * orthogonal
        parameters = list(self.model.forward_maps.parameters()) + list(self.model.backward.parameters())
        representation_grad = self._synchronize_step(representation_loss, self.representation_optimizer, parameters)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        with torch.no_grad():
            style_reward = self.model.discriminate(state, command).clamp(-16.118095, 16.118095)
            auxiliary_reward = self.reward_normalizer(batch['auxiliary_reward'].squeeze(-1))
        for name, reward, optimizer in (
            ('style', style_reward, self.style_optimizer),
            ('auxiliary', auxiliary_reward, self.auxiliary_optimizer),
        ):
            with torch.no_grad():
                next_control = self.model.actor.sample(next_history, command, self.options['exploration_std'], self.options['target_noise_clip'])
                target_value = getattr(self.target, name)(next_critic_state, next_control, command).min(0).values
                target_value = reward + discount.squeeze(-1) * target_value
            values = getattr(self.model, name)(critic_state, batch['action'], command)
            loss = .5 * (values - target_value).square().mean(1).sum()
            maps = self.model.style_maps if name == 'style' else self.model.auxiliary_maps
            gradient = self._synchronize_step(loss, optimizer, maps.parameters())
            metrics['loss/' + name] = loss.detach()
            metrics['gradient/' + name] = gradient
            metrics['reward_train/' + name] = reward.mean().detach()
        critics = list(self.model.forward_maps.parameters()) + list(self.model.style_maps.parameters()) + list(self.model.auxiliary_maps.parameters())
        for parameter in critics:
            parameter.requires_grad_(False)
        action = self.model.actor.sample(history, command, self.options['exploration_std'], self.options['target_noise_clip'])
        q_fb = (self.model(critic_state, action, command) * command).sum(-1).min(0).values
        q_style = self.model.style(critic_state, action, command).min(0).values
        q_auxiliary = self.model.auxiliary(critic_state, action, command).min(0).values
        scale = q_fb.detach().abs().mean()
        actor_loss = -q_fb.mean() - scale * (self.options['style_coefficient'] * q_style.mean() + self.options['auxiliary_coefficient'] * q_auxiliary.mean())
        actor_grad = self._synchronize_step(actor_loss, self.actor_optimizer, self.model.actor.parameters())
        for parameter in critics:
            parameter.requires_grad_(True)
        with torch.no_grad():
            for name in ('forward_maps', 'backward', 'style_maps', 'auxiliary_maps'):
                tau = self.options['target_tau'] if name in ('forward_maps', 'backward') else self.options['critic_target_tau']
                for target, online in zip(getattr(self.target, name).parameters(), getattr(self.model, name).parameters()):
                    target.lerp_(online, tau)
            distance = batch['next_observation'][:, self.goal_slice].reshape(-1, 4, 3).norm(dim=-1).amax(-1, keepdim=True)
            weights = torch.exp(-distance / self.options['task_reward_temperature'])
            task_vector = (self.model.backward(next_state) * weights).sum(0)
            if self.world_size > 1:
                distributed.all_reduce(task_vector)
            self.task_command = project_task(task_vector)
            moment = backward.detach().T @ backward.detach() / count
            eigenvalues = torch.linalg.eigvalsh(moment)
            mean_action = self.model.actor(history, command)
            other_action = self.model.actor(history, command.roll(1, 0))
        self.optimizer_step += 1
        metrics.update({
            'loss/fb': fb_loss.detach(), 'loss/orthogonal': .5 * orthogonal.detach(),
            'loss/actor': actor_loss.detach(), 'gradient/representation': representation_grad,
            'gradient/actor': actor_grad, 'value/fb': q_fb.detach().mean(),
            'value/style': q_style.detach().mean(), 'value/auxiliary': q_auxiliary.detach().mean(),
            'policy/action_saturation': (mean_action.abs() > .95).float().mean(),
            'policy/command_action_delta': (mean_action - other_action).norm(dim=-1).mean(),
            'representation/backward_norm': backward.detach().norm(dim=-1).mean(),
            'representation/effective_rank': moment.trace().square() / moment.square().sum().clamp_min(1e-8),
            'representation/min_eigenvalue': eigenvalues.min(), 'representation/max_eigenvalue': eigenvalues.max(),
            'task/relabel_fraction': relabel.float().mean(), 'replay/valid_fraction': batch['valid'].mean(),
        })
        return metrics


class FBcprPlayer(FBcprAgent):
    def __init__(self, params):
        super().__init__('fb_cpr_eval', params)

    @torch.no_grad()
    def _prepare_reward_command(self):
        mode = self.options.get('evaluation_command_mode', 'legacy')
        self.reward_inference = {'mode': mode}
        if mode == 'legacy':
            return
        if mode != 'full_reward':
            raise ValueError('v2 replay supports original total reward inference only')
        checkpoint = self.last_checkpoint.resolve()
        status = json.loads((checkpoint.parent.parent / 'status.json').read_text())
        if status['state'] not in ('trained', 'budget_stopped', 'iteration_limit'):
            raise ValueError('Reward inference requires a completed coordinated save')
        if status['environment_steps'] != self.frame or status['optimizer_step'] != self.optimizer_step:
            raise ValueError('Training status and checkpoint counters differ')
        world_size = self.options['inference_world_size']
        count = self.options['inference_samples']
        if world_size < 1 or count < world_size:
            raise ValueError('Invalid replay inference sample counts')
        paths = [checkpoint.with_name(checkpoint.stem + '.rank' + str(rank) + '.replay.pt')
                 for rank in range(world_size)]
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        sources = [{'path': str(path), 'bytes': path.stat().st_size,
                    'mtime_ns': path.stat().st_mtime_ns} for path in paths]
        identity = {'mode': mode, 'checkpoint_sha256': digest.hexdigest(),
                    'frame': self.frame, 'optimizer_step': self.optimizer_step,
                    'sources': sources, 'sample_count': count,
                    'seed': self.options['inference_seed'], 'world_size': world_size}
        destination = checkpoint.parent / 'reward_inference'
        destination.mkdir(exist_ok=True)
        bank_path = destination / 'full_reward.pt'
        # All evaluation workers share one frozen inference result. No model,
        # replay, optimizer or rollout RNG is modified by this preparation.
        with (destination / 'full_reward.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            bank = torch.load(bank_path, map_location='cpu') if bank_path.exists() else None
            if bank is None or bank['identity'] != identity:
                pools, populations, means = [], [], []
                for rank, path in enumerate(paths):
                    saved = torch.load(path, map_location='cpu')
                    replay = TransitionReplay(1, 1, 1, 'cpu')
                    replay.load_state_dict(saved['replay'])
                    replay.storage = {key: value for key, value in replay.storage.items()
                                      if key in ('next_state', 'reward', 'valid')}
                    del saved
                    population = sum(len(rows) for rows in replay.valid_indices[:replay.size])
                    if population == 0:
                        raise ValueError('Empty valid replay shard')
                    generator = torch.Generator().manual_seed(identity['seed'] + rank)
                    samples = count // world_size + int(rank < count % world_size)
                    pool = replay.sample(samples, generator=generator)
                    del replay
                    if not pool['valid'].bool().all() or not all(torch.isfinite(value).all() for value in pool.values()):
                        raise ValueError('Invalid reward inference samples')
                    vector = torch.zeros(self.options['latent_size'], device=self.device)
                    for start in range(0, samples, self.options['inference_batch_size']):
                        stop = start + self.options['inference_batch_size']
                        state = pool['next_state'][start:stop].to(self.device)
                        reward = pool['reward'][start:stop].to(self.device)
                        features = self.model.backward(self.state_normalizer(state))
                        vector += (features * reward).sum(0)
                    means.append(vector / samples)
                    populations.append(population)
                    pools.append({'rank': rank, 'population': population, **pool})
                weights = torch.tensor(populations, device=self.device, dtype=torch.float64)
                weights /= weights.sum()
                vector = (torch.stack(means).double() * weights[:, None]).sum(0).float()
                if not torch.isfinite(vector).all() or vector.norm() < 1e-8:
                    raise ValueError('Zero or nonfinite inferred reward direction')
                current_sources = [{'path': str(path), 'bytes': path.stat().st_size,
                                    'mtime_ns': path.stat().st_mtime_ns} for path in paths]
                if current_sources != sources:
                    raise ValueError('Replay changed during reward inference')
                metadata = {**identity, 'populations': populations,
                            'unprojected_norm': float(vector.norm()),
                            'reward_mean_per_rank': [float(pool['reward'].mean()) for pool in pools],
                            'positive_reward_samples': [int((pool['reward'] > 0).sum()) for pool in pools],
                            'definition': 'population-weighted rank means of B(next_state) times original recorded total transition reward; no reward clipping or centering',
                            'replay_epoch_validation': 'completed trainer status after coordinated save barrier; legacy v2 shards do not contain counters'}
                torch.save({'identity': identity, 'ranks': pools}, destination / 'samples.pt')
                bank = {'identity': identity, 'metadata': metadata, 'command': project_task(vector).cpu()}
                temporary = bank_path.with_suffix('.tmp')
                torch.save(bank, temporary)
                temporary.replace(bank_path)
                bank_path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
            self.task_command = bank['command'].to(self.device)
            self.reward_inference = bank['metadata']

    @torch.no_grad()
    def run(self):
        self.run_started = time.monotonic()
        self._initialize_rollout()
        self.model.eval()
        self._prepare_reward_command()
        returns = torch.zeros(self.num_actors, device=self.device)
        records = []
        environment = self.vec_env.env
        for step in range(self.options['evaluation_steps']):
            self.history[self.history_reset] = 0
            self.previous_action[self.history_reset] = 0
            self.history = torch.cat((self.history[:, 1:], self.observations['obs'][:, None]), dim=1)
            command = self.task_command.expand(self.num_actors, -1)
            action = self.model.actor(self._history_input(self.history, self.previous_action), command)
            observations, reward, done, info = self.vec_env.step(action)
            returns += reward.reshape(-1)
            for index in done.nonzero().flatten().tolist():
                records.append({'environment_id': index, 'end_step': step + 1, 'return': float(returns[index]),
                                'successes': float(environment.successes[index]),
                                'goal_fraction': float(environment.successes[index]) / environment.max_consecutive_successes,
                                'keypoint_distance': float(environment.closest_keypoint_max_dist[index])})
            returns.masked_fill_(done.bool(), 0)
            self.history_reset = self.reset_before | done.bool()
            self.reset_before = done.bool()
            self.observations = {name: value.clone() for name, value in observations.items()}
            self.previous_action = action
        summary = {'episodes': len(records), 'records': records, 'steps': self.options['evaluation_steps'],
                   'reward_inference': self.reward_inference,
                   'checkpoint': str(self.last_checkpoint), 'training_environment_steps': self.frame,
                   'object_name': environment.cfg['env']['objectName'],
                   'fixed_goal_path': environment.cfg['env'].get('fixedGoalStatesJsonPath'),
                   'evaluation_tolerance': float(environment.success_tolerance),
                   'schema': {'observation': self.observation_size, 'state': self.state_size, 'action': self.action_size}}
        for name in ('return', 'successes', 'goal_fraction'):
            summary['mean_' + name] = float(np.mean([row[name] for row in records])) if records else None
        (self.output_dir / 'evaluation.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        self._status('evaluated', episodes=len(records), mean_successes=summary['mean_successes'])
        return summary
