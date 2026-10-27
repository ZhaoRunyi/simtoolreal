# BFM-Zero / Meta FB-CPR architecture and loss adaptation.
# Reference components: CC BY-NC 4.0; see docs/licenses/bfm-zero-LICENSE.
"""FB-CPR with auxiliary critic, adapted to the existing SimToolReal interface.

BFM-Zero 05ad611 is the algorithm reference. Expert observations are collected
from the published SimToolReal PPO teacher, not synthesized from object goals.
"""

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from rl_games.algos_torch.fb_evaluation import EvaluationRecorder
import torch.distributed as distributed
from torch import nn
from torch.nn import functional
import yaml

from rl_games.algos_torch.fb_agent import FBAgent
from rl_games.algos_torch.fb_networks import project_task, successor_loss
from rl_games.algos_torch.fb_cpr_networks import (
    BatchNormalizer, CPRModel, ExpertSequences, RewardNormalizer, TransitionReplay, TemporalReplay,
)
from rl_games.algos_torch.fb_task_inference import (
    reward_components, infer_command_bank, checkpoint_digest,
    select_phase_commands,
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
        replay_class = TemporalReplay if self.options.get('temporal_replay', False) else TransitionReplay
        self.replay = replay_class(
            math.ceil(self.options['replay_capacity'] / self.world_size), self.num_actors,
            self.config['horizon_length'], self.device,
        )
        self.z_buffer = torch.empty(self.options['z_buffer_size'], self.options['latent_size'], device=self.device)
        self.z_source_buffer = torch.zeros(self.options['z_buffer_size'], dtype=torch.long, device=self.device)
        self.z_position = 0
        self.z_size = 0
        self.experts = None
        self.expert_report = None

    @torch.no_grad()
    def save(self, filename, replay=False):
        super().save(filename, replay=replay)
        if not replay or not self.options.get('record_inference_pool', False):
            return
        self._save_inference_pool(filename)

    @torch.no_grad()
    def _save_inference_pool(self, filename):
        # Snapshot the same valid replay population as the learner. Keep all
        # training RNG streams unchanged by this evaluation-only operation.
        population = sum(len(rows) for rows in self.replay.valid_indices[:self.replay.size])
        counts = torch.zeros(self.world_size, dtype=torch.long, device=self.device)
        counts[self.global_rank] = population
        if self.world_size > 1:
            distributed.all_reduce(counts)
        total = int(counts.sum())
        if total == 0:
            raise ValueError('Cannot export an empty inference population')
        requested = self.options['inference_samples']
        allocation = (counts * requested) // total
        remainder = requested - int(allocation.sum())
        order = ((counts * requested) % total).argsort(descending=True)
        allocation[order[:remainder]] += 1
        count = int(allocation[self.global_rank])
        generator = torch.Generator(device=self.replay.device)
        generator.manual_seed(self.options['inference_seed'] + self.global_rank)
        batch = self.replay.sample(count, generator=generator)
        if 'reward_components' not in batch:
            raise ValueError('Replay lacks original reward components; v3 collection is required')
        path = Path(filename).with_name(Path(filename).stem + '.rank' + str(self.global_rank) + '.inference.pt')
        payload = {
            'schema_version': 1, 'frame': self.frame, 'optimizer_step': self.optimizer_step,
            'rank': self.global_rank, 'world_size': self.world_size, 'population': population,
            'sample_count': count, 'seed': self.options['inference_seed'] + self.global_rank,
            'reward_names': list(reward_components),
            'next_state': batch['next_state'].cpu(),
            'reward_components': batch['reward_components'].cpu(),
        }
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        if self.world_size > 1:
            distributed.barrier()

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
            'z_source_buffer': self.z_source_buffer,
            'z_buffer': self.z_buffer, 'z_position': self.z_position, 'z_size': self.z_size,
            'expert_report': self.expert_report,
            'training_world_size': self.world_size,
        }

    def _restore_extra(self, checkpoint):
        self.training_world_size = checkpoint.get('training_world_size')
        if 'z_source_buffer' in checkpoint:
            self.z_source_buffer.copy_(checkpoint['z_source_buffer'])
        elif self.options.get('log_command_sources', False):
            raise ValueError('Source-labelled rollout requires a v3 checkpoint')
        for name in ('style_optimizer', 'discriminator_optimizer', 'action_normalizer', 'reward_normalizer'):
            getattr(self, name).load_state_dict(checkpoint[name])
        for name in ('z_buffer', 'z_position', 'z_size', 'expert_report'):
            setattr(self, name, checkpoint[name])

    @torch.no_grad()
    def _prepare_experts(self):
        path = self.output_dir / ('expert.rank' + str(self.global_rank) + '.pt')
        scene_identity = self._reference_scene_identity()
        if path.exists():
            data = torch.load(path, map_location=self.device)
            if scene_identity is not None:
                if data.get('scene_identity') != scene_identity or 'quality_values' not in data:
                    raise ValueError('PPO reference cache scene identity or physical quality metadata mismatch')
                expected_shape = (self.options['expert_steps'], min(self.num_actors, self.options['expert_environments']))
                if data['states'].shape != (*expected_shape, self.state_size) or data['valid'].shape != expected_shape:
                    raise ValueError('PPO reference cache collection shape mismatch')
                selected = data.get('selected_environments')
                if (selected is None or selected.shape != (expected_shape[1],)
                        or selected.unique().numel() != selected.numel()
                        or selected.min() < 0 or selected.max() >= self.num_actors):
                    raise ValueError('PPO reference cache selected environment identity mismatch')
                for length in (self.options['expert_sequence_length'], self.options['tracking_length']):
                    if not data['valid'].unfold(0, length, 1).all(-1).any():
                        raise ValueError('PPO reference cache has no valid reference windows')
                self.reference_quality_values = data['quality_values']
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
        quality_values = []
        initial_scene = None
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
            if scene_identity is not None:
                quality_values.append(self._reference_physical_values(observations['states'][selected]).clone())
                if step == 0:
                    initial_scene = self._reference_initial_scene()
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
        if scene_identity is not None:
            self.reference_quality_values = torch.stack(quality_values)
            data.update(scene_identity=scene_identity, initial_scene=initial_scene,
                        quality_values=self.reference_quality_values, selected_environments=selected)
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
        self._check_reference_quality()
        return super().train()

    def _reference_scene_identity(self):
        if not self.options.get('reference_min_qualified_environments', 0):
            return None
        environment = self.vec_env.env
        asset_hashes = environment.object_asset_sha256
        if len(asset_hashes) != 1:
            raise ValueError('Single-task reference collection requires exactly one object asset')
        scales = environment.object_scales
        if not torch.equal(scales, scales[0].expand_as(scales)):
            raise ValueError('Single-task reference collection found different object scales')
        physical_properties = []
        for scene, actor in zip(environment.envs, environment.objects):
            bodies = environment.gym.get_actor_rigid_body_properties(scene, actor)
            shapes = environment.gym.get_actor_rigid_shape_properties(scene, actor)
            body_values = []
            for body in bodies:
                body_values.append([body.mass, body.com.x, body.com.y, body.com.z] +
                    [getattr(getattr(body.inertia, row), column) for row in 'xyz' for column in 'xyz'])
            physical_properties.append({'bodies': body_values, 'friction': [shape.friction for shape in shapes]})
        if any(properties != physical_properties[0] for properties in physical_properties):
            raise ValueError('Single-task reference collection found different object physics')
        teacher_path = Path(self.options['teacher_path']) / 'model.pth'
        digest = hashlib.sha256()
        with teacher_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return {'asset_sha256': asset_hashes, 'scale': scales[0].cpu().tolist(),
                'physics': physical_properties[0], 'environment_count': len(environment.envs),
                'settings': copy.deepcopy(environment.cfg['env']),
                'teacher_sha256': digest.hexdigest(), 'state_list': list(environment.state_list),
                'state_size': self.state_size, 'rank': self.global_rank, 'seed': torch.initial_seed(),
                'collection': {name: self.options[name] for name in
                    ('expert_steps', 'expert_environments', 'reference_stride')}}

    def _reference_initial_scene(self):
        environment = self.vec_env.env
        objects = environment.root_state_tensor[environment.object_indices, :7]
        goals = environment.goal_states[:, :7]
        expected_goal = goals.new_tensor(environment.cfg['env']['goalObjectPose'])
        if not torch.allclose(goals, expected_goal.expand_as(goals), atol=1e-6, rtol=0):
            raise ValueError('Single-task physical reset produced inconsistent goals')
        if not torch.equal(environment.object_scale_noise_multiplier,
                           torch.ones_like(environment.object_scale_noise_multiplier)):
            raise ValueError('Single-task physical reset changed object scale')
        reset = environment.reference_initial_reset
        reset_pose = reset['object_pose']
        if not torch.allclose(reset_pose, reset_pose[0].expand_as(reset_pose), atol=1e-6, rtol=0):
            raise ValueError('Single-task reset commands have different initial object poses')
        if not torch.equal(reset['scale_noise'], torch.ones_like(reset['scale_noise'])):
            raise ValueError('Single-task reset command changed object scale')
        report = {'reset_command_object_pose': reset_pose[0].cpu().tolist(),
                  'object_pose': objects[0].cpu().tolist(), 'goal_pose': goals[0].cpu().tolist(),
                  'object_pose_max_difference': float((objects - objects[0]).abs().max()),
                  'scale_noise_min': float(environment.object_scale_noise_multiplier.min()),
                  'scale_noise_max': float(environment.object_scale_noise_multiplier.max()),
                  'after_first_physics_step': True}
        path = self.output_dir / ('reference_scene.rank' + str(self.global_rank) + '.json')
        path.write_text(json.dumps(report, indent=2))
        return report

    def _reference_physical_values(self, states):
        environment = self.vec_env.env
        slices = {}
        offset = 0
        for name in environment.state_list:
            size = environment.obs_type_size_dict[name]
            slices[name] = slice(offset, offset + size)
            offset += size
        if states.shape[-1] != offset:
            raise ValueError('Reference state schema does not match the current environment')
        keypoints = states[..., slices['keypoints_rel_palm']].reshape(*states.shape[:-1], 4, 3)
        center = states[..., slices['palm_pos']] + keypoints.mean(-2)
        goal_error = states[..., slices['keypoints_rel_goal']].reshape(*states.shape[:-1], 4, 3).norm(dim=-1).max(-1).values
        velocity = states[..., slices['object_vel']]
        return torch.stack((center[..., 2], goal_error, velocity[..., :3].norm(dim=-1),
                            velocity[..., 3:].norm(dim=-1)), dim=-1)

    @torch.no_grad()
    def _check_reference_quality(self):
        minimum = self.options.get('reference_min_qualified_environments', 0)
        if not minimum:
            return
        values = self.reference_quality_values
        if values.shape != (*self.experts.valid.shape, 4) or values.dtype != torch.float32:
            raise ValueError('Reference qualification requires aligned float32 physical measurements')
        qualified = self.experts.valid.bool() & torch.isfinite(values).all(-1)
        qualified &= values[..., 0] >= self.options['reference_min_height']
        qualified &= values[..., 1] <= self.options['reference_max_goal_error']
        qualified &= values[..., 2] <= self.options['reference_max_linear_speed']
        qualified &= values[..., 3] <= self.options['reference_max_angular_speed']
        length = self.options['reference_quality_steps']
        windows = qualified.unfold(0, length, 1).all(-1)
        counts = windows.sum(0)
        passed = (counts > 0).sum() >= minimum
        report = {
            'rank': self.global_rank, 'qualified_environments': int((counts > 0).sum()),
            'required_environments': minimum, 'window_steps': length,
            'qualified_windows_per_environment': counts.cpu().tolist(),
            'minimum_height': self.options['reference_min_height'],
            'maximum_goal_error': self.options['reference_max_goal_error'],
            'maximum_linear_speed': self.options['reference_max_linear_speed'],
            'maximum_angular_speed': self.options['reference_max_angular_speed'],
            'measurement_dtype': str(values.dtype),
            'reference_rows_filtered_for_training': False, 'passed': bool(passed),
        }
        (self.output_dir / ('reference_quality.rank' + str(self.global_rank) + '.json')).write_text(json.dumps(report, indent=2))
        decision = torch.tensor(int(passed), device=self.device)
        if self.world_size > 1:
            distributed.all_reduce(decision, op=distributed.ReduceOp.MIN)
        if not decision.item():
            raise ValueError('PPO reference cache failed the registered physical reachability gate; cache retained')

    @torch.no_grad()
    def _sample_commands(self, count):
        if getattr(self, 'z_size', 0):
            indices = torch.randint(self.z_size, (count,), device=self.device)
            self.sampled_command_source = self.z_source_buffer[indices].clone()
            return self.z_buffer[indices].clone()
        self.sampled_command_source = torch.zeros(count, dtype=torch.long, device=self.device)
        return project_task(torch.randn(count, self.options['latent_size'], device=self.device))

    @torch.no_grad()
    def _cache_commands(self, commands, sources=None):
        commands = commands[-len(self.z_buffer):]
        if sources is None:
            sources = torch.zeros(len(commands), dtype=torch.long, device=self.device)
        sources = sources[-len(commands):]
        indices = (torch.arange(len(commands), device=self.device) + self.z_position) % len(self.z_buffer)
        self.z_buffer[indices] = commands
        self.z_source_buffer[indices] = sources
        self.z_position = (self.z_position + len(commands)) % len(self.z_buffer)
        self.z_size = min(len(self.z_buffer), self.z_size + len(commands))

    def _initialize_rollout(self):
        self.vec_env.env.reset_buf.fill_(1)
        self.observations = {key: value.clone() for key, value in self.vec_env.reset().items()}
        self.history = torch.zeros(self.num_actors, self.options['history_steps'], self.observation_size, device=self.device)
        self.previous_action = torch.zeros(self.num_actors, self.action_size, device=self.device)
        self.rollout_command = self._sample_commands(self.num_actors)
        self.rollout_command_source = self.sampled_command_source.clone()
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

    def _sample_action(self, history, state, command, clip=None):
        prompt = self.model.actor_prompt(state, command)
        return self.model.actor.sample(history, prompt, self.options['exploration_std'], clip)

    def _actor_objective(self, state, history, command):
        action = self._sample_action(history, state, command, self.options['target_noise_clip'])
        critic_state = torch.cat((state, history), -1)
        q_fb = (self.model(critic_state, action, command) * command).sum(-1).min(0).values
        q_style = self.model.style(critic_state, action, command).min(0).values
        q_auxiliary = self.model.auxiliary(critic_state, action, command).min(0).values
        scale = q_fb.detach().abs().mean()
        loss = -q_fb.mean() - scale * (self.options['style_coefficient'] * q_style.mean()
                                     + self.options['auxiliary_coefficient'] * q_auxiliary.mean())
        return loss, q_fb, q_style, q_auxiliary

    def _actor_gradient_probe(self, q_fb, q_style, q_auxiliary):
        """Measure weighted branch gradients on the actual sampled actor graph.

        This does not populate .grad, draw samples, or step an optimizer. Average
        vectors across ranks before comparing them, as in the production update.
        """
        parameters = tuple(self.model.actor.parameters())
        scale = q_fb.detach().abs().mean()
        losses = {
            'fb': -q_fb.mean(),
            'style': -scale * self.options['style_coefficient'] * q_style.mean(),
            'auxiliary': -scale * self.options['auxiliary_coefficient'] * q_auxiliary.mean(),
        }
        vectors = {}
        for name, loss in losses.items():
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
            vector = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
            if self.world_size > 1:
                distributed.all_reduce(vector)
                vector.div_(self.world_size)
            vectors[name] = vector
        before = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
        return vectors, before

    @torch.no_grad()
    def _actor_gradient_metrics(self, vectors, before):
        """Compare branches with the total gradient and actual Adam change."""
        parameters = tuple(self.model.actor.parameters())
        actual_update = torch.cat([parameter.detach().reshape(-1) for parameter in parameters]) - before
        # .grad has already been averaged and clipped by the production step.
        clipped_gradient = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        metrics = {'actor_grad/update_norm': actual_update.norm()}
        for name, vector in vectors.items():
            metrics['actor_grad/' + name + '_norm'] = vector.norm()
            metrics['actor_grad/' + name + '_update_cosine'] = functional.cosine_similarity(
                -vector, actual_update, dim=0)
        for left, right in (('fb', 'style'), ('fb', 'auxiliary'), ('style', 'auxiliary')):
            metrics['actor_grad/' + left + '_' + right + '_cosine'] = functional.cosine_similarity(
                vectors[left], vectors[right], dim=0)
        metrics['actor_grad/fb_total_cosine'] = functional.cosine_similarity(
            vectors['fb'], clipped_gradient, dim=0)
        return metrics

    def _log(self, metrics):
        # Probes need not coincide with log_interval; preserve their exact step.
        metrics = {**metrics, **getattr(self, 'pending_actor_gradient_metrics', {})}
        result = super()._log(metrics)
        self.pending_actor_gradient_metrics = {}
        return result

    @torch.no_grad()
    def _rollout_prompts(self):
        refresh = self.reset_before | (self.command_age >= self.options['command_steps'])
        candidates = self._sample_commands(self.num_actors)
        self.rollout_command = torch.where(refresh[:, None], candidates, self.rollout_command)
        self.rollout_command_source = torch.where(refresh, self.sampled_command_source, self.rollout_command_source)
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
        self.rollout_command_source[self.tracking_indices] = 3

    def _auxiliary_reward(self, environment, components):
        reward = sum(components[name] for name in self.options['auxiliary_reward_terms'])
        limit_reward = torch.zeros_like(reward)
        coefficient = self.options.get('joint_limit_coefficient', 0.0)
        if coefficient:
            fraction = self.options.get('soft_joint_limit_fraction', 0.95)
            if coefficient < 0 or not 0 < fraction <= 1:
                raise ValueError('Joint limit coefficient must be nonnegative and soft fraction in (0, 1]')
            lower = environment.arm_hand_dof_lower_limits
            upper = environment.arm_hand_dof_upper_limits
            center = (lower + upper) * 0.5
            radius = (upper - lower) * (0.5 * fraction)
            position = environment.arm_hand_dof_pos
            violation = (center - radius - position).clamp_min(0)
            violation += (position - center - radius).clamp_min(0)
            limit_reward = -coefficient * violation.sum(-1)
        return reward + limit_reward, limit_reward

    @torch.no_grad()
    def collect(self):
        records, episode_returns = [], []
        step_rewards, episode_rewards = {}, {}
        command_metrics = {}
        environment = self.vec_env.env
        for _ in range(self.config['horizon_length']):
            self._rollout_prompts()
            self.history[self.history_reset] = 0
            self.previous_action[self.history_reset] = 0
            self.history = torch.cat((self.history[:, 1:], self.observations['obs'][:, None]), dim=1)
            history = self.history.clone()
            previous_action = self.previous_action.clone()
            action = self._sample_action(self._history_input(history, previous_action),
                                         self.state_normalizer(self.observations['states']), self.rollout_command)
            current = {key: value.clone() for key, value in self.observations.items()}
            observations, reward, done, info = self.vec_env.step(action)
            observations = {key: value.clone() for key, value in observations.items()}
            done = done.bool()
            next_history = torch.cat((history[:, 1:], observations['obs'][:, None]), dim=1)
            auxiliary_reward, joint_limit_reward = self._auxiliary_reward(environment, info['episode_cumulative'])
            step_rewards.setdefault('aux_joint_limit', []).append(joint_limit_reward.mean().detach())
            records.append({
                'observation': current['obs'], 'state': current['states'],
                'next_observation': observations['obs'], 'next_state': observations['states'],
                'history': history.flatten(1), 'next_history': next_history.flatten(1),
                'previous_action': previous_action, 'action': action,
                'command': self.rollout_command.clone(), 'reward': reward.reshape(-1, 1).clone(),
                'auxiliary_reward': auxiliary_reward.reshape(-1, 1).clone(),
                'discount': self.config['gamma'] * (~done).float().unsqueeze(-1),
                'valid': (~self.reset_before).unsqueeze(-1).clone(),
            })
            if self.options.get('record_inference_pool', False):
                records[-1]['reward_components'] = torch.stack(
                    [info['episode_cumulative'][name] for name in reward_components], dim=-1
                ).clone()
            if self.options.get('log_command_sources', False):
                for source, name in enumerate(('random', 'replay_goal', 'expert_mean', 'expert_sequence')):
                    mask = (self.rollout_command_source == source) & ~self.reset_before
                    count_name = 'command_count/' + name
                    command_metrics[count_name] = command_metrics.get(count_name, 0) + mask.sum()
                    for term, value in info['episode_cumulative'].items():
                        key = 'command_reward_sum/' + name + '/' + term
                        command_metrics[key] = command_metrics.get(key, 0) + (value.reshape(-1) * mask).sum()
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
        metrics.update(command_metrics)
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
            'learning/sample_reuse': reward.new_tensor(self.options['updates_per_iteration'] * self.config['minibatch_size']
                                                       / (self.num_actors * self.config['horizon_length'])),
            'replay/steps_per_environment': reward.new_tensor(self.replay.capacity * self.config['horizon_length']),
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
            sources = torch.where(selector.squeeze(-1) < .2, 1,
                                  torch.where(selector.squeeze(-1) < .8, 2, 0))
            self._cache_commands(command, sources)
            relabel = torch.rand(count, 1, device=self.device) < self.options['relabel_probability']
            command = torch.where(relabel, command, batch['command'])
            next_action = self._sample_action(next_history, next_state, command, self.options['target_noise_clip'])
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
                next_control = self._sample_action(next_history, next_state, command, self.options['target_noise_clip'])
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
        actor_loss, q_fb, q_style, q_auxiliary = self._actor_objective(state, history, command)
        probe_every = self.options.get('actor_gradient_probe_every', 0)
        probe = self._actor_gradient_probe(q_fb, q_style, q_auxiliary) if (
            probe_every > 0 and self.optimizer_step % probe_every == 0) else None
        self.actor_optimizer.zero_grad(set_to_none=True)
        self._backward_loss(actor_loss)
        actor_loss = actor_loss.detach()
        q_fb, q_style, q_auxiliary = q_fb.detach(), q_style.detach(), q_auxiliary.detach()
        if self.options.get('logo_coefficient', 0) > 0:
            window = self.replay.sample_window(count, self.options['logo_horizon'])
            available = torch.tensor(int(window is not None), device=self.device)
            if self.world_size > 1:
                distributed.all_reduce(available, op=distributed.ReduceOp.MIN)
            metrics['logo/available'] = available.float()
            if available.item():
                local_state = self.state_normalizer(window['state'])
                local_history = self._history_input(window['history'].reshape(
                    -1, self.options['history_steps'], self.observation_size), window['previous_action'])
                with torch.no_grad():
                    local_command = self.model.backward(self.state_normalizer(window['local_state']))
                    goal_command = self.model.backward(self.state_normalizer(window['goal_state']))
                local_loss = self._actor_objective(local_state, local_history, local_command)[0]
                # Backpropagate each term before constructing the next graph.
                # Accumulate the same weighted sum, then synchronize/clip/step once.
                self._backward_loss(self.options['logo_coefficient'] * self.options['logo_local_weight'] * local_loss)
                local_loss = local_loss.detach()
                goal_loss = self._actor_objective(local_state, local_history, goal_command)[0]
                self._backward_loss(self.options['logo_coefficient'] * self.options['logo_goal_weight'] * goal_loss)
                goal_loss = goal_loss.detach()
                pair_loss = self.options['logo_local_weight'] * local_loss + self.options['logo_goal_weight'] * goal_loss
                actor_loss = actor_loss + self.options['logo_coefficient'] * pair_loss
                metrics.update({'loss/logo_local': local_loss.detach(), 'loss/logo_goal': goal_loss.detach(),
                                'loss/logo_scaled': self.options['logo_coefficient'] * pair_loss.detach()})
        actor_grad = self._step_accumulated_gradients(self.actor_optimizer, self.model.actor.parameters())
        if probe is not None:
            probe_metrics = self._actor_gradient_metrics(*probe)
            probe_metrics['actor_grad/probe_optimizer_step'] = actor_grad.new_tensor(self.optimizer_step + 1)
            probe_metrics['actor_grad/probe_environment_steps'] = actor_grad.new_tensor(self.frame)
            self.pending_actor_gradient_metrics = probe_metrics
            metrics.update(probe_metrics)
        for parameter in critics:
            parameter.requires_grad_(True)
        with torch.no_grad():
            for name in ('forward_maps', 'backward', 'style_maps', 'auxiliary_maps'):
                tau = self.options['target_tau'] if name in ('forward_maps', 'backward') else self.options['critic_target_tau']
                for target, online in zip(getattr(self.target, name).parameters(), getattr(self.model, name).parameters()):
                    target.lerp_(online, tau)
            if self.options.get('cache_tracking_command', True):
                distance = batch['next_observation'][:, self.goal_slice].reshape(-1, 4, 3).norm(dim=-1).amax(-1, keepdim=True)
                weights = torch.exp(-distance / self.options['task_reward_temperature'])
                task_vector = (self.model.backward(next_state) * weights).sum(0)
                if self.world_size > 1:
                    distributed.all_reduce(task_vector)
                self.task_command = project_task(task_vector)
            moment = backward.detach().T @ backward.detach() / count
            eigenvalues = torch.linalg.eigvalsh(moment)
            prompt = self.model.actor_prompt(state, command)
            mean_action = self.model.actor(history, prompt)
            other_action = self.model.actor(history, self.model.actor_prompt(state, command.roll(1, 0)))
            metrics['policy/prompt_norm'] = prompt.norm(dim=-1).mean()
            metrics['policy/arm_saturation'] = (mean_action[:, :7].abs() > .95).float().mean()
            metrics['policy/hand_saturation'] = ((mean_action[:, 7:].abs() > .95).float().mean()
                                                 if self.action_size > 7 else mean_action.new_zeros(()))
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
    def _prepare_task_commands(self):
        mode = self.options.get('evaluation_command_mode', 'legacy')
        if mode == 'legacy':
            if not self.options.get('cache_tracking_command', True):
                raise ValueError('Legacy task_command is disabled; infer a command bank')
            return
        if mode not in ('phased', 'full', 'grasp_lift', 'track'):
            raise ValueError('Unknown evaluation command mode: ' + mode)
        digest = checkpoint_digest(self.last_checkpoint)
        bank_path = self.options.get('command_bank_path')
        if bank_path:
            bank = torch.load(bank_path, map_location='cpu')
            if bank['checkpoint_sha256'] != digest:
                raise ValueError('Command bank was encoded with a different checkpoint')
        else:
            import glob
            pattern = self.options.get('inference_pool_glob') or str(
                self.last_checkpoint.with_name(self.last_checkpoint.stem + '.rank*.inference.pt'))
            bank = infer_command_bank(
                self.model.backward, self.state_normalizer, sorted(glob.glob(pattern)),
                self.frame, self.optimizer_step, self.device,
                self.options['inference_method'], self.options['inference_temperature'],
                self.options['inference_batch_size'],
                self.training_world_size or self.options['inference_world_size'],
            )
            bank['checkpoint_sha256'] = digest
            torch.save(bank, self.output_dir / 'command_bank.pt')
        if bank['stage_names'] != ['grasp_lift', 'track', 'full']:
            raise ValueError('Unexpected command bank phase ordering')
        commands = bank['commands']
        if commands.shape != (3, self.options['latent_size']) or not torch.isfinite(commands).all():
            raise ValueError('Invalid command bank shape or values')
        expected_norm = math.sqrt(self.options['latent_size'])
        if not torch.allclose(commands.norm(dim=-1), commands.new_full((3,), expected_norm)):
            raise ValueError('Command bank directions are not projected to the latent sphere')
        self.command_bank = commands.to(self.device)
        self.command_bank_metadata = {key: value for key, value in bank.items() if key != 'commands'}
        (self.output_dir / 'command_inference.json').write_text(
            json.dumps(self.command_bank_metadata, indent=2, allow_nan=False))

    def _evaluation_commands(self, environment):
        mode = self.options.get('evaluation_command_mode', 'legacy')
        if mode == 'legacy':
            return self.task_command.expand(self.num_actors, -1)
        return select_phase_commands(self.command_bank, environment.lifted_object, mode)

    @torch.no_grad()
    def run(self):
        self.vec_env.env.cfg["env"]["capture_viewer"] = False
        self.run_started = time.monotonic()
        self._initialize_rollout()
        self.model.eval()
        self._prepare_task_commands()
        returns = torch.zeros(self.num_actors, device=self.device)
        records = []
        environment = self.vec_env.env
        recorder = EvaluationRecorder(self)
        for step in range(self.options['evaluation_steps']):
            self.history[self.history_reset] = 0
            self.previous_action[self.history_reset] = 0
            self.history = torch.cat((self.history[:, 1:], self.observations['obs'][:, None]), dim=1)
            command = self._evaluation_commands(environment)
            prompt = self.model.actor_prompt(self.state_normalizer(self.observations['states']), command)
            action = self.model.actor(self._history_input(self.history, self.previous_action), prompt)
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
                   'checkpoint': str(self.last_checkpoint), 'training_environment_steps': self.frame,
                   'object_name': environment.cfg['env']['objectName'],
                   'fixed_goal_path': environment.cfg['env'].get('fixedGoalStatesJsonPath'),
                   'evaluation_tolerance': float(environment.success_tolerance),
                   'schema': {'observation': self.observation_size, 'state': self.state_size, 'action': self.action_size},
                   'command_mode': self.options.get('evaluation_command_mode', 'legacy'),
                   'command_bank': getattr(self, 'command_bank_metadata', None)}
        for name in ('return', 'successes', 'goal_fraction'):
            summary['mean_' + name] = float(np.mean([row[name] for row in records])) if records else None
        recorder.finish(summary)
        (self.output_dir / 'evaluation.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        self._status('evaluated', episodes=len(records), mean_successes=summary['mean_successes'])
        return summary
