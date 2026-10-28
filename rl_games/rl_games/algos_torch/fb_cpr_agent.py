# BFM-Zero / Meta FB-CPR architecture and loss adaptation.
# Reference components: CC BY-NC 4.0; see docs/licenses/bfm-zero-LICENSE.
"""FB-CPR with auxiliary critic, adapted to the existing SimToolReal interface.

BFM-Zero 05ad611 is the algorithm reference. Expert observations are collected
from the published SimToolReal PPO teacher, not synthesized from object goals.
"""

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
    select_phase_commands, mixed_reward_commands,
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
        self.mixed_reward_bank = None
        self.reward_sidecar = None
        self.reward_sidecar_identity = None
        self.reward_generator = None
        if self.options.get('mixed_reward_commands', False):
            self.reward_generator = torch.Generator(device=self.replay.device)
            self.reward_generator.manual_seed(self.options['reward_sampling_seed'] + self.global_rank)

    @torch.no_grad()
    def save(self, filename, replay=False):
        super().save(filename, replay=replay)
        if replay and self.options.get('mixed_reward_commands', False):
            path = Path(filename).with_name(Path(filename).stem + '.rank' + str(self.global_rank) + '.commands.pt')
            payload = {
                'schema_version': 2,
                'frame': self.frame, 'optimizer_step': self.optimizer_step, 'rank': self.global_rank,
                'world_size': self.world_size, 'bank': self.mixed_reward_bank,
                'sidecar_identity': self.reward_sidecar_identity,
                'generator_state': self.reward_generator.get_state(),
                'z_buffer': self.z_buffer, 'z_source_buffer': self.z_source_buffer,
                'z_position': self.z_position, 'z_size': self.z_size,
                'model_sha256': checkpoint_digest(filename),
                'replay_identity': self._reward_replay_identity(filename),
                'replay_state': self._reward_replay_state(),
                'warmup_complete': self.replay.size >= self.options['warmup_blocks'],
                'rollout_step': self.rollout_step,
                'resume_semantics': 'same replay file; fresh physical episodes; parent restores per-rank global RNG',
            }
            temporary = path.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.replace(path)
            if self.world_size > 1:
                distributed.barrier()
        if replay and self.options.get('record_inference_pool', False):
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
        if path.exists():
            data = torch.load(path, map_location=self.device)
            self.experts = ExpertSequences(data['states'], data['valid'])
            self.expert_report = data['report']
            self.vec_env.env.reference_reset_bank = data['resets']
            return
        teacher, checkpoint, teacher_root = self._load_reference_policy()
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

    def _load_reference_policy(self):
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
        return teacher, checkpoint, teacher_root

    def _command_source_names(self):
        if self.options.get('mixed_reward_commands', False):
            return ('random', 'replay_goal', 'reward_full', 'reward_phased')
        return ('random', 'replay_goal', 'expert_mean', 'expert_sequence')

    def _reward_replay_identity(self, filename):
        path = Path(filename)
        path = path.with_name(path.stem + '.rank' + str(self.global_rank) + '.replay.pt')
        metadata = path.stat()
        # This identity deliberately rejects copied/replaced shards. It is not
        # a content hash or a portable checkpoint manifest.
        return {'device': metadata.st_dev, 'inode': metadata.st_ino,
                'size': metadata.st_size, 'mtime_ns': metadata.st_mtime_ns,
                'ctime_ns': metadata.st_ctime_ns}

    def _reward_replay_state(self):
        return {'size': self.replay.size, 'position': self.replay.position,
                'next_time': getattr(self.replay, 'next_time', None),
                'fields': sorted(self.replay.storage or {}),
                'capacity': self.replay.capacity}

    def _reward_gate(self, error, phase):
        success = torch.tensor(int(error is None), device=self.device)
        if self.world_size > 1:
            distributed.all_reduce(success, op=distributed.ReduceOp.MIN)
        if not success.item():
            if error is not None:
                raise error
            raise RuntimeError('Another rank failed ' + phase)

    @staticmethod
    def _reward_json_digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _reward_plain_config(value):
        from omegaconf import OmegaConf
        # The production omegaconf_to_dict leaves ListConfig nodes nested in
        # ordinary dicts. Resolve each node with its original parent context.
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return {name: FBcprAgent._reward_plain_config(item) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FBcprAgent._reward_plain_config(item) for item in value]
        return value

    def _reward_scene_mapping(self):
        environment = self.vec_env.env
        # Procedural URDF directories are process-local and already deleted
        # after scene creation. Their filenames encode sampled geometry and
        # densities; bind their ordered names, scales and generator source.
        procedural = environment.cfg['env'].get('objectName') == 'handle_head_primitives'
        assets = [Path(path).name if procedural else str(path) for path in environment.object_asset_files]
        source_root = Path(__file__).resolve().parents[3] / 'isaacgymenvs/tasks/simtoolreal'
        return {'asset_files': assets,
                'asset_index': (torch.arange(self.num_actors) % len(assets)).tolist(),
                'object_scales': environment.object_scales.cpu().tolist(),
                'generator_source_sha256': {name: checkpoint_digest(source_root / name) for name in
                                            ('env.py', 'generate_objects.py', 'object_size_distributions.py')}}

    @torch.no_grad()
    def _collect_reward_sidecar(self):
        """Independent PPO process: retain actual rewards and terminal transitions."""
        path = self.output_dir / ('reward_sidecar.rank' + str(self.global_rank) + '.pt')
        if path.exists():
            raise FileExistsError('Preserve existing PPO sidecar; validate it before reuse')
        teacher, checkpoint, teacher_root = self._load_reference_policy()
        environment = self.vec_env.env
        if (self.num_actors, self.options['expert_steps'], self.options['expert_environments']) != (6144, 1500, 256):
            raise ValueError('PPO reward collection requires 6144 scenes, 1500 steps and 256 fixed streams per rank')
        if getattr(environment, 'reference_reset_bank', None) is not None:
            raise ValueError('PPO reward collection must begin without a reference reset bank')
        configuration = self._reward_plain_config(environment.cfg)
        initial_curriculum = {name: getattr(environment, name) for name in
                              ('success_tolerance', 'frame_since_restart', 'last_curriculum_update')}
        if (initial_curriculum['frame_since_restart'] != 0
                or initial_curriculum['last_curriculum_update'] != 0
                or initial_curriculum['success_tolerance'] != environment.initial_tolerance
                or configuration.get('test', False)
                or configuration['env'].get('useFixedGoalStates', False)
                or configuration['env'].get('evalSuccessTolerance') is not None):
            raise ValueError('PPO reward collection requires original training curriculum and goals')
        scene_mapping = self._reward_scene_mapping()
        collection_seed = torch.initial_seed()
        # Fail before reset, stream selection or rollout if static metadata
        # cannot be serialized. Keep the predeclared plain-JSON hash semantics.
        configuration_sha256 = self._reward_json_digest(configuration)
        scene_mapping_sha256 = self._reward_json_digest(scene_mapping)
        json.dumps(initial_curriculum, allow_nan=False)
        teacher_sha256 = checkpoint_digest(teacher_root / 'model.pth')
        teacher_config_sha256 = checkpoint_digest(teacher_root / 'config.yaml')
        memory = [value.to(self.device) for value in teacher.get_default_rnn_state()]
        observations = {key: value.clone() for key, value in self.vec_env.reset().items()}
        selected = torch.randperm(self.num_actors, device=self.device)[:self.options['expert_environments']]
        reset_before = torch.ones(self.num_actors, dtype=torch.bool, device=self.device)
        episode = torch.zeros(self.num_actors, dtype=torch.long, device=self.device)
        records = []
        started = time.monotonic()
        for step in range(self.options['expert_steps']):
            inputs = torch.cat((observations['obs'], observations['obs'].new_full((self.num_actors, 1), 50.)), -1)
            result = teacher({'is_train': False, 'prev_actions': None, 'obs': inputs, 'rnn_states': memory})
            action, memory = result['mus'].clamp(-1, 1), result['rnn_states']
            state = observations['states'][selected].clone()
            goal_reset_before = environment.reset_goal_buf[selected].bool().clone()
            next_observations, reward, done, info = self.vec_env.step(action)
            done = done.bool()
            episode += reset_before.long()
            components = torch.stack([info['episode_cumulative'][name] for name in reward_components], -1)
            if not torch.allclose(components.sum(-1), reward.reshape(-1), atol=1e-5, rtol=1e-5):
                raise ValueError('PPO sidecar component sum does not equal original reward')
            row = {
                'state': state, 'next_state': next_observations['states'][selected].clone(),
                'action': action[selected].clone(), 'reward': reward.reshape(-1)[selected].clone(),
                'reward_components': components[selected].clone(),
                'valid': (~reset_before)[selected].clone(), 'reset_before': reset_before[selected].clone(),
                'done': done[selected].clone(), 'discount': self.config['gamma'] * (~done[selected]).float(),
                'episode': episode[selected].clone(), 'goal_reset_before': goal_reset_before,
                'goal_reset_after': environment.reset_goal_buf[selected].bool().clone(),
                'source_step': torch.full_like(selected, step), 'source_environment': selected.clone(),
                'goal': environment.goal_states[selected].clone(),
                'object': environment.root_state_tensor[environment.object_indices[selected]].clone(),
                'object_initial': environment.object_init_state[selected].clone(),
                'table': environment.root_state_tensor[environment.table_indices[selected]].clone(),
                'lifted': environment.lifted_object[selected].clone(),
                'successes': environment.successes[selected].clone(),
                'success_tolerance': state.new_full((len(selected),), environment.success_tolerance),
            }
            records.append({name: value.cpu() for name, value in row.items()})
            for value in memory:
                value[:, done] = 0
            reset_before = done
            observations = {key: value.clone() for key, value in next_observations.items()}
            if step % 100 == 0:
                self._status('reward_collection', collection_step=step)
        payload = {
            'schema_version': 2, 'rank': self.global_rank, 'world_size': self.world_size,
            'reward_names': list(reward_components), 'selected_environments': selected.cpu(),
            'steps': self.options['expert_steps'], 'simulation_environments': self.num_actors,
            'state_size': self.state_size, 'collection_environment_config': configuration,
            'collection_environment_sha256': configuration_sha256,
            'seed': collection_seed, 'initial_curriculum': initial_curriculum,
            'scene_mapping': scene_mapping, 'scene_mapping_sha256': scene_mapping_sha256,
            'simulation_transitions': self.num_actors * self.options['expert_steps'],
            'stored_transitions': len(selected) * self.options['expert_steps'],
            'teacher_sha256': teacher_sha256,
            'teacher_config_sha256': teacher_config_sha256,
            'seconds': time.monotonic() - started,
            'timing': 'state/action before step; next_state, components, goal and object after reward; reset_before excluded, terminal retained',
            'rows': {name: torch.stack([row[name] for row in records]) for name in records[0]},
        }
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        report = {name: value for name, value in payload.items() if name not in ('rows', 'selected_environments', 'collection_environment_config', 'scene_mapping')}
        report.update(sha256=checkpoint_digest(path), valid_transitions=int(payload['rows']['valid'].sum()),
                      terminal_valid_transitions=int((payload['rows']['valid'] & payload['rows']['done']).sum()))
        path.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        self._status('reward_collection_complete', collection_steps=payload['steps'])

    def _load_reward_sidecar(self):
        path = Path(self.options['reward_sidecar_path'].format(rank=self.global_rank))
        expected = self.options['reward_sidecar_sha256'][self.global_rank]
        actual = checkpoint_digest(path)
        if actual != expected:
            raise ValueError('PPO reward sidecar identity changed')
        data = torch.load(path, map_location='cpu')
        if (data['schema_version'] != 2 or data['rank'] != self.global_rank
                or data['world_size'] != self.world_size or data['state_size'] != self.state_size
                or tuple(data['reward_names']) != reward_components):
            raise ValueError('PPO reward sidecar schema/rank mismatch')
        teacher_root = Path(self.options['teacher_path'])
        if (data['teacher_sha256'] != checkpoint_digest(teacher_root / 'model.pth')
                or data['teacher_config_sha256'] != checkpoint_digest(teacher_root / 'config.yaml')):
            raise ValueError('PPO reward sidecar teacher mismatch')
        expected_environment = self.options['reward_sidecar_environment_sha256'][self.global_rank]
        if (data['collection_environment_sha256'] != expected_environment
                or self._reward_json_digest(data['collection_environment_config']) != expected_environment
                or data['seed'] != self.options['reward_sidecar_seed'][self.global_rank]):
            raise ValueError('PPO reward sidecar collection configuration or seed mismatch')
        if (self._reward_json_digest(data['scene_mapping']) != data['scene_mapping_sha256']
                or data['scene_mapping'] != self._reward_scene_mapping()):
            raise ValueError('PPO reward sidecar scene mapping mismatch')
        if (data['steps'], data['simulation_environments'], len(data['selected_environments']),
                data['simulation_transitions'], data['stored_transitions']) != (1500, 6144, 256, 9216000, 384000):
            raise ValueError('PPO reward sidecar collection budget mismatch')
        curriculum = data['initial_curriculum']
        if (curriculum['frame_since_restart'] != 0 or curriculum['last_curriculum_update'] != 0
                or curriculum['success_tolerance'] != self.vec_env.env.initial_tolerance):
            raise ValueError('PPO reward sidecar initial curriculum mismatch')
        valid = data['rows']['valid'].bool()
        rows = data['rows']
        if (valid.shape != (1500, 256) or not valid.any()
                or not torch.equal(valid, ~rows['reset_before'].bool())
                or not torch.equal(rows['reset_before'][1:], rows['done'][:-1])
                or not rows['reset_before'][0].all()
                or not torch.equal(rows['source_step'], torch.arange(1500)[:, None].expand(1500, 256))
                or not torch.equal(rows['source_environment'], data['selected_environments'].expand(1500, 256))
                or data['selected_environments'].unique().numel() != 256
                or data['selected_environments'].min() < 0 or data['selected_environments'].max() >= 6144
                or not torch.allclose(rows['reward_components'].sum(-1), rows['reward'], atol=1e-5, rtol=1e-5)
                or not torch.allclose(rows['discount'], self.config['gamma'] * (~rows['done'].bool()).float())):
            raise ValueError('PPO reward sidecar valid population is not replay-compatible')
        self.reward_sidecar = {name: data['rows'][name][valid].to(self.device)
                               for name in ('next_state', 'reward_components')}
        if not all(torch.isfinite(value).all() for value in self.reward_sidecar.values()):
            raise ValueError('Nonfinite PPO reward sidecar')
        self.reward_sidecar_rows = valid.nonzero()
        self.reward_sidecar_identity = {'path': str(path.resolve()), 'sha256': actual,
                                       'population': int(valid.sum()), 'rank': self.global_rank,
                                       'environment_sha256': expected_environment, 'seed': data['seed'],
                                       'scene_mapping_sha256': data['scene_mapping_sha256']}

    @torch.no_grad()
    def _refresh_mixed_reward_bank(self):
        count = self.options['reward_samples_per_source']
        evidence = {'frame': self.frame, 'optimizer_step': self.optimizer_step,
                    'rank': self.global_rank, 'sidecar_identity': self.reward_sidecar_identity}
        error = None
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if self.state_normalizer.training:
                raise ValueError('Reward inference requires a frozen normalizer')
            if not isinstance(self.replay, TemporalReplay):
                raise ValueError('Mixed reward commands require the original v4 temporal replay')
            indices = torch.randint(len(self.reward_sidecar['next_state']), (count,),
                                    generator=self.reward_generator, device=self.replay.device)
            blocks, rows = self.replay._indices(count, generator=self.reward_generator)
            online = self.replay._gather(blocks, rows)
            evidence.update(ppo_indices=indices, online_blocks=blocks, online_rows=rows,
                            online_time_ids=self.replay.time_ids[blocks])
            if hasattr(self, 'reward_sidecar_rows'):
                evidence['ppo_step_stream'] = self.reward_sidecar_rows[indices]
            indices = indices.to(self.device)
            states = torch.cat((self.reward_sidecar['next_state'][indices], online['next_state']))
            components = torch.cat((self.reward_sidecar['reward_components'][indices], online['reward_components']))
            features = self.model.backward(self.state_normalizer(states))
            evidence.update(states=states, components=components, features=features)
            torch.backends.cuda.matmul.allow_tf32 = False
            bank, statistics, valid = mixed_reward_commands(features, components, count, evidence)
            evidence.update(statistics=statistics, local_valid=bool(valid))
            if not valid.item():
                error = FloatingPointError('Mixed reward bank lacks finite positive task evidence')
        except Exception as caught:
            error = caught
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        globally_valid = torch.tensor(int(error is None), device=self.device)
        if self.world_size > 1:
            distributed.all_reduce(globally_valid, op=distributed.ReduceOp.MIN)
        if not globally_valid.item():
            evidence['error'] = repr(error)
            path = self.output_dir / ('reward_bank_failure.rank' + str(self.global_rank) + '.pt')
            torch.save(evidence, path)
            if error is not None:
                raise error
            raise FloatingPointError('Mixed reward bank failed on another rank')
        if self.mixed_reward_bank is not None:
            cosines = functional.cosine_similarity(bank, self.mixed_reward_bank['commands'], dim=-1)
            for index, name in enumerate(('grasp_lift', 'track', 'full')):
                statistics[name + '/previous_cosine'] = cosines[index]
                statistics[name + '/previous_angle_radians'] = cosines[index].clamp(-1, 1).acos()
        self.mixed_reward_bank = {
            'commands': bank, 'frame': self.frame, 'before_optimizer_step': self.optimizer_step,
            'backward_optimizer_step': self.optimizer_step,
            'normalizer_online_updates': 2 * (self.optimizer_step + 1),
            'sidecar_identity': self.reward_sidecar_identity,
            'ppo_contribution': evidence['ppo_contribution'],
            'online_contribution': evidence['online_contribution'],
            'ppo_indices': evidence['ppo_indices'], 'online_blocks': blocks,
            'online_rows': rows, 'online_time_ids': self.replay.time_ids[blocks],
        }
        return {'reward_bank/' + name: value for name, value in statistics.items()}

    def train(self):
        if self.options.get('reference_reward_collection', False):
            error = None
            try:
                self._collect_reward_sidecar()
            except Exception as caught:
                error = caught
            self._reward_gate(error, 'reward sidecar collection')
            return
        if self.options.get('mixed_reward_commands', False):
            error = None
            try:
                self._load_reward_sidecar()
            except Exception as caught:
                error = caught
            self._reward_gate(error, 'reward sidecar loading')
        self._prepare_experts()
        return super().train()

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

    def _initialize_rollout(self, restore_training=True):
        resumed_rollout_step = 0
        if restore_training and self.options.get('mixed_reward_commands', False) and self.last_checkpoint is not None:
            error = None
            try:
                path = self.last_checkpoint.with_name(self.last_checkpoint.stem + '.rank' + str(self.global_rank) + '.commands.pt')
                saved = torch.load(path, map_location=self.device)
                expected = (self.frame, self.optimizer_step, self.global_rank, self.world_size)
                actual = tuple(saved[name] for name in ('frame', 'optimizer_step', 'rank', 'world_size'))
                if (saved['schema_version'] != 2 or actual != expected
                        or saved['sidecar_identity'] != self.reward_sidecar_identity
                        or saved['model_sha256'] != checkpoint_digest(self.last_checkpoint)
                        or saved['replay_identity'] != self._reward_replay_identity(self.last_checkpoint)
                        or saved['warmup_complete'] != (saved['replay_state']['size'] >= self.options['warmup_blocks'])):
                    raise ValueError('Mixed reward snapshot does not match model/replay/rank/sidecar/warmup')
                bank = saved['bank']
                if self.optimizer_step == 0:
                    if bank is not None:
                        raise ValueError('Warmup checkpoint cannot contain a reward bank')
                elif (bank is None or bank['before_optimizer_step'] != self.optimizer_step - 1
                      or bank['backward_optimizer_step'] != self.optimizer_step - 1
                      or bank['normalizer_online_updates'] != 2 * self.optimizer_step
                      or bank['frame'] != self.frame or bank['sidecar_identity'] != self.reward_sidecar_identity
                      or bank['commands'].shape != (3, self.options['latent_size'])
                      or not torch.isfinite(bank['commands']).all()):
                    raise ValueError('Mixed reward bank version does not match the saved learner')
                self.mixed_reward_bank = bank
                self.reward_generator.set_state(saved['generator_state'].cpu())
                for name in ('z_buffer', 'z_source_buffer', 'z_position', 'z_size'):
                    setattr(self, name, saved[name])
                self.reward_resume_replay_state = saved['replay_state']
                resumed_rollout_step = saved['rollout_step']
            except Exception as caught:
                error = caught
            self._reward_gate(error, 'reward command restoration')
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
        self.rollout_step = resumed_rollout_step

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

    def _actor_gradient_probe(self, q_fb, q_style, q_auxiliary, command_sources=None):
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
        if command_sources is not None:
            for source, name in enumerate(self._command_source_names()):
                mask = (command_sources == source).to(q_fb.dtype)
                # Contributions to the actual global-batch objective; empty
                # sources contribute exactly zero, without drawing new actions.
                losses['fb_' + name] = -(q_fb * mask).mean()
                losses['style_' + name] = -scale * self.options['style_coefficient'] * (q_style * mask).mean()
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
        for source in self._command_source_names():
            if 'fb_' + source in vectors:
                metrics['actor_grad/' + source + '_fb_style_cosine'] = functional.cosine_similarity(
                    vectors['fb_' + source], vectors['style_' + source], dim=0)
        metrics['actor_grad/fb_total_cosine'] = functional.cosine_similarity(
            vectors['fb'], clipped_gradient, dim=0)
        return metrics

    def _log(self, metrics):
        # Probes need not coincide with log_interval; preserve their exact step.
        metrics = {**metrics, **getattr(self, 'pending_actor_gradient_metrics', {}),
                   **getattr(self, 'pending_command_diagnostics', {})}
        result = super()._log(metrics)
        self.pending_actor_gradient_metrics = {}
        self.pending_command_diagnostics = {}
        return result

    @torch.no_grad()
    def _rollout_prompts(self):
        refresh = self.reset_before | (self.command_age >= self.options['command_steps'])
        candidates = self._sample_commands(self.num_actors)
        length = self.options['tracking_length']
        if self.options.get('mixed_reward_commands', False):
            if not hasattr(self, 'held_command'):
                self.held_command = self.rollout_command.clone()
                self.held_command_source = self.rollout_command_source.clone()
            self.held_command = torch.where(refresh[:, None], candidates, self.held_command)
            self.held_command_source = torch.where(refresh, self.sampled_command_source, self.held_command_source)
            self.rollout_command = self.held_command.clone()
            self.rollout_command_source = self.held_command_source.clone()
            self.command_age = torch.where(refresh, 0, self.command_age)
            if self.mixed_reward_bank is None:
                return
            if self.rollout_step % length == 0 or self.tracking_prompts is None:
                self.tracking_indices = torch.randperm(self.num_actors, device=self.device)[:len(self.tracking_indices)]
                self.tracking_prompts = True
            lifted = self.observations['states'][:, self.options['lifted_state_index']].bool()
            commands = select_phase_commands(self.mixed_reward_bank['commands'], lifted, 'phased')
            self.rollout_command[self.tracking_indices] = commands[self.tracking_indices]
            self.rollout_command_source[self.tracking_indices] = 3
            return
        self.rollout_command = torch.where(refresh[:, None], candidates, self.rollout_command)
        self.rollout_command_source = torch.where(refresh, self.sampled_command_source, self.rollout_command_source)
        self.command_age = torch.where(refresh, 0, self.command_age)
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
        if hasattr(self, 'reward_resume_replay_state'):
            error = None
            if self._reward_replay_state() != self.reward_resume_replay_state:
                error = ValueError('Loaded replay logical counters/schema differ from command snapshot')
            self._reward_gate(error, 'loaded replay binding')
            del self.reward_resume_replay_state
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
            if self.options.get('mixed_reward_commands', False):
                records[-1]['command_source'] = self.rollout_command_source[:, None].clone()
            if self.options.get('record_inference_pool', False) or self.options.get('mixed_reward_commands', False):
                records[-1]['reward_components'] = torch.stack(
                    [info['episode_cumulative'][name] for name in reward_components], dim=-1
                ).clone()
            if self.options.get('log_command_sources', False):
                for source, name in enumerate(self._command_source_names()):
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

    def _discriminator_update(self, expert_state, expert_command, state, rollout_command, rollout_sources=None):
        real = self.model.discriminate(expert_state, expert_command)
        fake = self.model.discriminate(state, rollout_command)
        diagnostics = {}
        if rollout_sources is not None:
            with torch.no_grad():
                diagnostics['discriminator_source/expert_positive/count'] = real.new_tensor(real.numel())
                diagnostics['discriminator_source/expert_positive/logit_sum'] = real.sum()
                diagnostics['discriminator_source/expert_positive/saturated_count'] = (real.abs() > math.log(99)).float().sum()
                for source, name in enumerate(self._command_source_names()):
                    mask = rollout_sources == source
                    prefix = 'discriminator_source/' + name + '/'
                    diagnostics[prefix + 'count'] = mask.float().sum()
                    diagnostics[prefix + 'logit_sum'] = fake[mask].sum()
                    diagnostics[prefix + 'saturated_count'] = (fake[mask].abs() > math.log(99)).float().sum()
                # Same states, fixed replacement commands, no new sampling.
                replacements = [('expert_mean', expert_command)] + [
                    (name, self.mixed_reward_bank['commands'][index].expand_as(rollout_command))
                    for index, name in enumerate(('reward_grasp_lift', 'reward_track', 'reward_full'))]
                probe_every = self.options.get('actor_gradient_probe_every', 0)
                if probe_every <= 0 or self.optimizer_step % probe_every:
                    replacements = []
                for name, replacement in replacements:
                    swapped = self.model.discriminate(state, replacement)
                    prefix = 'discriminator_fixed_state_swap/' + name + '/'
                    diagnostics[prefix + 'logit'] = swapped.mean()
                    diagnostics[prefix + 'delta_from_behavior'] = (swapped - fake).mean()
                    diagnostics[prefix + 'saturated_fraction'] = (swapped.abs() > math.log(99)).float().mean()
                if replacements:
                    self.pending_command_diagnostics = {
                        name: value for name, value in diagnostics.items()
                        if name.startswith('discriminator_fixed_state_swap/')}
                    self.pending_command_diagnostics['discriminator_fixed_state_swap/before_optimizer_step'] = real.new_tensor(self.optimizer_step)
        alpha = torch.rand(len(state), 1, device=self.device)
        mixed_state = (alpha * expert_state + (1 - alpha) * state).requires_grad_(True)
        mixed_command = (alpha * expert_command + (1 - alpha) * rollout_command).requires_grad_(True)
        logits = self.model.discriminate(mixed_state, mixed_command)
        gradients = torch.autograd.grad(logits.sum(), (mixed_state, mixed_command), create_graph=True)
        penalty = (torch.cat(gradients, dim=-1).norm(dim=-1) - 1).square().mean()
        loss = (functional.softplus(-real) + functional.softplus(fake)).mean() + self.options['discriminator_penalty'] * penalty
        self._synchronize_step(loss, self.discriminator_optimizer, self.model.discriminator.parameters())
        return {'loss/discriminator': loss.detach(), 'loss/discriminator_gradient_penalty': penalty.detach(),
                'discriminator/expert_logit': real.detach().mean(), 'discriminator/online_logit': fake.detach().mean(),
                **diagnostics}

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
        mixed_metrics = {}
        if self.options.get('mixed_reward_commands', False):
            mixed_metrics = self._refresh_mixed_reward_bank()
        with torch.no_grad():
            expert_sequence = self.experts.sample(count // length, length + 1)
            expert_state = self.state_normalizer(expert_sequence[:, :-1].flatten(0, 1))
            expert_next_state = self.state_normalizer(expert_sequence[:, 1:].flatten(0, 1))
            expert_features = self.model.backward(expert_next_state).reshape(-1, length, self.options['latent_size'])
            expert_command = project_task(expert_features.mean(1)).repeat_interleave(length, 0)
        # Discriminator uses actual behavior commands, before learner relabeling.
        rollout_sources = batch['command_source'].squeeze(-1).long() if self.options.get('mixed_reward_commands', False) else None
        metrics = self._discriminator_update(expert_state, expert_command, state, batch['command'], rollout_sources)
        metrics.update(mixed_metrics)
        with torch.no_grad():
            command = project_task(torch.randn(count, self.options['latent_size'], device=self.device))
            selector = torch.rand(count, 1, device=self.device)
            permutation = torch.randperm(count, device=self.device)
            goals = project_task(self.model.backward(next_state[permutation]))
            command = torch.where(selector < .2, goals, command)
            reference_command = expert_command[permutation]
            if self.options.get('mixed_reward_commands', False):
                reference_command = self.mixed_reward_bank['commands'][2].expand_as(command)
            command = torch.where((selector >= .2) & (selector < .8), reference_command, command)
            sources = torch.where(selector.squeeze(-1) < .2, 1,
                                  torch.where(selector.squeeze(-1) < .8, 2, 0))
            self._cache_commands(command, sources)
            relabel = torch.rand(count, 1, device=self.device) < self.options['relabel_probability']
            command = torch.where(relabel, command, batch['command'])
            learner_sources = torch.where(relabel.squeeze(-1), sources, rollout_sources) if rollout_sources is not None else None
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
        probe = self._actor_gradient_probe(q_fb, q_style, q_auxiliary, learner_sources) if (
            probe_every > 0 and self.optimizer_step % probe_every == 0) else None
        if probe is not None and learner_sources is not None:
            for source, name in enumerate(self._command_source_names()):
                source_count = (learner_sources == source).float().sum()
                if self.world_size > 1:
                    distributed.all_reduce(source_count)
                metrics['actor_grad/' + name + '_sample_count'] = source_count
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
            probe_metrics.update({name: value for name, value in metrics.items()
                                  if name.startswith('actor_grad/')})
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
        self._initialize_rollout(restore_training=False)
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
