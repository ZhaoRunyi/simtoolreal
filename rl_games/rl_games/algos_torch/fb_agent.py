"""Recurrent online FB learner for the unchanged rl_games vector environment.

The environment remains authoritative for physics, rewards, targets and resets.
Only the learner, replay and task-conditioned networks differ from PPO/SAPG.
"""

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from rl_games.common import vecenv
from rl_games.algos_torch.fb_networks import FBModel, SequenceReplay, project_task, successor_loss
from rl_games.algos_torch.running_mean_std import RunningMeanStd


class FBAgent:
    def __init__(self, base_name, params):
        self.params = params
        self.config = params["config"]
        self.options = self.config["fb"]
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(self.config.get("device", "cuda:0"))
        if self.world_size > 1:
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
            if not distributed.is_initialized():
                distributed.init_process_group("nccl")
        torch.backends.cuda.matmul.allow_tf32 = self.options.get("tf32", True)
        torch.backends.cudnn.allow_tf32 = self.options.get("tf32", True)
        torch.set_num_threads(self.options.get("cpu_threads", 2))
        self.num_actors = self.config["num_actors"]
        self.vec_env = self.config.get("vec_env")
        if self.vec_env is None:
            self.vec_env = vecenv.create_vec_env(
                self.config["env_name"], self.num_actors, **self.config.get("env_config", {})
            )
        self.env_info = self.vec_env.get_env_info()
        self.observation_size = self.env_info["observation_space"].shape[0]
        self.state_size = self.env_info["state_space"].shape[0]
        self.action_size = self.env_info["action_space"].shape[0]
        environment = self.vec_env.env
        observation_offset = 0
        self.goal_slice = None
        for name in environment.obs_list:
            width = environment.obs_type_size_dict[name]
            if name == "keypoints_rel_goal":
                self.goal_slice = slice(observation_offset, observation_offset + width)
            observation_offset += width
        if self.goal_slice is None:
            raise ValueError("FB tracking reward requires the existing keypoints_rel_goal observation")
        if self.config["seq_length"] != self.config["horizon_length"]:
            raise ValueError("FB replay uses one rollout horizon for burn-in and one for learning")
        if self.config["minibatch_size"] % self.config["seq_length"]:
            raise ValueError("FB minibatch_size must be divisible by seq_length")
        self.model = FBModel(
            self.observation_size, self.state_size, self.action_size,
            self.options, params["network"],
        ).to(self.device)
        if self.world_size > 1:
            for parameter in self.model.parameters():
                distributed.broadcast(parameter.data, 0)
        self.target = copy.deepcopy(self.model).requires_grad_(False)
        self.observation_normalizer = RunningMeanStd((self.observation_size,)).to(self.device).eval()
        self.state_normalizer = RunningMeanStd((self.state_size,)).to(self.device).eval()
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.config["learning_rate"]
        )
        representation_parameters = list(self.model.forward_maps.parameters())
        self.representation_optimizer = torch.optim.Adam([
            {"params": representation_parameters},
            {"params": self.model.backward.parameters(), "lr": self.options["backward_learning_rate"]},
        ], lr=self.config["learning_rate"])
        self.auxiliary_optimizer = torch.optim.Adam(
            self.model.auxiliary_maps.parameters(), lr=self.config["learning_rate"]
        )
        self.frame = 0
        self.epoch_num = 0
        self.optimizer_step = 0
        self.elapsed_before_resume = 0.0
        self.task_command = project_task(torch.randn(self.options["latent_size"], device=self.device))
        self.command_cache = None
        self.replay = SequenceReplay(self.options["replay_blocks"], self.device)
        self.output_dir = Path(self.config.get("train_dir", "runs")) / self.config["name"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nn_dir = self.output_dir / "nn"
        self.nn_dir.mkdir(exist_ok=True)
        self.writer = None
        self.run_started = None
        self.training_initialized = False
        self.last_checkpoint = None

    def _initialize_logging(self):
        if self.global_rank != 0:
            return
        observer = self.config.get("features", {}).get("observer")
        if observer is not None:
            observer.before_init("fb", self.config, self.config["name"])
        self.writer = SummaryWriter(str(self.output_dir / "summaries"))
        (self.output_dir / "learner_config.json").write_text(json.dumps(self.params, default=str, indent=2))

    def _status(self, state, **metrics):
        if self.global_rank != 0:
            return
        report = {
            **metrics,
            "state": state, "epoch": self.epoch_num,
            "optimizer_step": self.optimizer_step, "environment_steps": self.frame,
            "checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "updated_at": time.time(),
            "metrics_optimizer_step": metrics.get("optimizer_step"),
        }
        path = self.output_dir / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, allow_nan=False, indent=2))
        temporary.replace(path)

    def _synchronize_step(self, loss, optimizer, parameters):
        parameters = list(parameters)
        finite = torch.isfinite(loss.detach()).float()
        if self.world_size > 1:
            distributed.all_reduce(finite, op=distributed.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Nonfinite FB training loss on one or more ranks")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad.reshape(-1) for parameter in parameters if parameter.grad is not None]
        packed = torch.cat(gradients)
        if self.world_size > 1:
            distributed.all_reduce(packed)
            packed.div_(self.world_size)
            offset = 0
            for parameter in parameters:
                if parameter.grad is not None:
                    count = parameter.numel()
                    parameter.grad.copy_(packed[offset:offset + count].view_as(parameter))
                    offset += count
        norm = nn.utils.clip_grad_norm_(parameters, self.config.get("grad_norm", 1.0))
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite FB gradient norm")
        optimizer.step()
        return norm.detach()

    @torch.no_grad()
    def _sample_commands(self, count):
        commands = project_task(torch.randn(count, self.options["latent_size"], device=self.device))
        if self.command_cache is not None:
            indices = torch.randint(len(self.command_cache), (count,), device=self.device)
            replay_commands = self.command_cache[indices]
            source = torch.rand(count, 1, device=self.device)
            goal_probability = self.options["goal_mix_probability"]
            commands = torch.where(source < goal_probability, replay_commands, commands)
            task_mask = (source >= goal_probability) & (source < goal_probability + self.options["task_mix_probability"])
            commands = torch.where(task_mask, self.task_command.expand_as(commands), commands)
        return commands

    @torch.no_grad()
    def _update_statistics(self, observations):
        self.observation_normalizer.train()
        self.state_normalizer.train()
        self.observation_normalizer(observations["obs"])
        self.state_normalizer(observations["states"])
        self.observation_normalizer.eval()
        self.state_normalizer.eval()

    def _initialize_rollout(self):
        self.observations = {key: value.clone() for key, value in self.vec_env.reset().items()}
        self.actor_state = self.model.actor.initial_state(self.num_actors, self.device)
        self.rollout_command = self._sample_commands(self.num_actors)
        self.reset_before = torch.ones(self.num_actors, dtype=torch.bool, device=self.device)
        self.command_age = torch.zeros(self.num_actors, dtype=torch.long, device=self.device)
        self.episode_reward = torch.zeros(self.num_actors, device=self.device)
        self.previous_block = None
        self.previous_initial_state = None

    @torch.no_grad()
    def collect(self):
        initial_state = tuple(value.clone() for value in self.actor_state)
        records = []
        episode_returns = []
        reward_steps = {}
        reward_episodes = {}
        environment = self.vec_env.env
        for _ in range(self.config["horizon_length"]):
            refresh = self.reset_before | (self.command_age >= self.options["command_steps"])
            candidates = self._sample_commands(self.num_actors)
            self.rollout_command = torch.where(refresh[:, None], candidates, self.rollout_command)
            self.command_age = torch.where(refresh, 0, self.command_age)
            self._update_statistics(self.observations)
            normalized = self.observation_normalizer(self.observations["obs"])
            memory, self.actor_state = self.model.actor.encode(
                normalized.unsqueeze(0), self.actor_state, self.reset_before.float().unsqueeze(0)
            )
            action = self.model.actor.act(memory[0], self.rollout_command)
            action = action + self.options["exploration_std"] * torch.randn_like(action)
            action = action.clamp(-1, 1)
            current = {key: value.clone() for key, value in self.observations.items()}
            next_observations, reward, done, info = self.vec_env.step(action)
            next_observations = {key: value.clone() for key, value in next_observations.items()}
            # SimToolReal resets physics in the following pre_physics_step. That
            # boundary row has a pre-reset observation and is excluded from TD.
            record = {
                "observation": current["obs"], "state": current["states"],
                "next_observation": next_observations["obs"], "next_state": next_observations["states"],
                "action": action, "command": self.rollout_command.clone(),
                "reward": reward.reshape(-1, 1).clone(),
                "discount": (self.config["gamma"] * (~done.bool()).float()).unsqueeze(-1),
                "reset": self.reset_before.unsqueeze(-1).clone(),
                "valid": (~self.reset_before).unsqueeze(-1).clone(),
            }
            records.append(record)
            self.episode_reward += reward.reshape(-1)
            finished = done.bool()
            # Environment supplies weighted step components and cumulative raw /
            # weighted episode components. Read before the next reset clears them.
            for name, value in info.get("episode_cumulative", {}).items():
                reward_steps.setdefault(name, []).append(value.detach().mean())
            for name, value in info.get("rewards_episode", {}).items():
                reward_episodes.setdefault(name, []).append(value.detach()[finished].sum())
            episode_returns.append(self.episode_reward[finished])
            self.episode_reward.masked_fill_(finished, 0)
            self.reset_before = finished
            self.observations = next_observations
            self.command_age += 1
        block = {key: torch.stack([record[key] for record in records]) for key in records[0]}
        if self.previous_block is not None:
            combined = {key: torch.cat((self.previous_block[key], block[key])) for key in block}
            combined["hidden"] = self.previous_initial_state[0]
            combined["cell"] = self.previous_initial_state[1]
            self.replay.add(combined)
        self.previous_block = block
        self.previous_initial_state = initial_state
        self.frame += self.num_actors * self.config["horizon_length"] * self.world_size
        valid_returns = torch.cat(episode_returns)
        reward_metrics = {
            "reward_step/" + name: torch.stack(values).mean()
            for name, values in reward_steps.items()
        }
        reward_metrics.update({
            "reward_episode_sum/" + name: torch.stack(values).sum()
            for name, values in reward_episodes.items()
        })
        return {
            **reward_metrics,
            "rollout/reward": block["reward"].mean(),
            "rollout/completed_episodes": torch.tensor(valid_returns.numel(), device=self.device),
            "rollout/episode_return_sum": valid_returns.sum(),
            "rollout/episode_return": valid_returns.mean() if valid_returns.numel() else torch.tensor(0., device=self.device),
            "task/successes_per_completed_episode": environment.prev_episode_successes.float().mean(),
            "task/closest_keypoint_distance": environment.prev_episode_closest_keypoint_max_dist.float().mean(),
            "task/success_tolerance": torch.as_tensor(environment.success_tolerance, device=self.device),
            "task/lifted_fraction": environment.lifted_object.float().mean(),
        }

    def _training_memory(self, batch, gradients):
        burn_in = self.config["horizon_length"]
        inputs = self.observation_normalizer(batch["observation"].flatten(0, 1)).view_as(batch["observation"])
        initial = (batch["hidden"], batch["cell"])
        resets = batch["reset"].squeeze(-1)
        with torch.no_grad():
            _, state = self.model.actor.encode(inputs[:burn_in], initial, resets[:burn_in])
        suffix = inputs[burn_in:]
        final_next = self.observation_normalizer(batch["next_observation"][-1]).unsqueeze(0)
        inputs_with_next = torch.cat((suffix, final_next))
        reset_with_next = torch.cat((resets[burn_in:], torch.zeros_like(resets[-1:])))
        with torch.set_grad_enabled(gradients):
            memory, _ = self.model.actor.encode(inputs_with_next, state, reset_with_next)
        return memory[:-1].flatten(0, 1), memory[1:].flatten(0, 1)

    def update(self):
        sequence_count = self.config["minibatch_size"] // self.config["seq_length"]
        batch = self.replay.sample(sequence_count)
        burn_in = self.config["horizon_length"]
        suffix = {key: value[burn_in:].flatten(0, 1) for key, value in batch.items() if key not in ("hidden", "cell")}
        # Decode terminal membership, retaining the exact configured discount
        # even when restoring older half-precision replay shards.
        suffix["discount"] = (suffix["discount"] > 0).float() * self.config["gamma"]
        observation = self.observation_normalizer(suffix["observation"])
        next_observation = self.observation_normalizer(suffix["next_observation"])
        state = self.state_normalizer(suffix["state"])
        next_state = self.state_normalizer(suffix["next_state"])
        valid = suffix["valid"].squeeze(-1)
        with torch.no_grad():
            cached = project_task(self.model.backward(next_observation))
            self.command_cache = cached
            mixed = self._sample_commands(sequence_count)
            mixed = mixed.unsqueeze(0).expand(self.config["seq_length"], -1, -1).reshape_as(suffix["command"])
            relabel = torch.rand(sequence_count, 1, device=self.device) < self.options["relabel_probability"]
            relabel = relabel.unsqueeze(0).expand(self.config["seq_length"], -1, -1).reshape(-1, 1)
            command = torch.where(relabel, mixed, suffix["command"])
            _, next_memory = self._training_memory(batch, False)
            next_action = self.model.actor.act(next_memory, command)
            noise = (self.options["exploration_std"] * torch.randn_like(next_action)).clamp(
                -self.options["target_noise_clip"], self.options["target_noise_clip"]
            )
            next_action = (next_action + noise).clamp(-1, 1)
            target_forward = self.target(next_state, next_action, command)
            target_backward = self.target.backward(next_observation)
        forward = self.model(state, suffix["action"], command)
        backward = self.model.backward(next_observation)
        fb_loss, orthogonal_loss = successor_loss(
            forward, backward, target_forward, target_backward,
            suffix["discount"], valid,
        )
        representation_loss = fb_loss + self.options["orthogonal_coefficient"] * orthogonal_loss
        representation_parameters = list(self.model.forward_maps.parameters()) + list(self.model.backward.parameters())
        representation_grad = self._synchronize_step(
            representation_loss, self.representation_optimizer, representation_parameters
        )
        with torch.no_grad():
            goal_error = suffix["next_observation"][:, self.goal_slice].reshape(-1, 4, 3)
            goal_distance = goal_error.norm(dim=-1).amax(dim=-1, keepdim=True)
            task_weights = torch.exp(-goal_distance / self.options["task_reward_temperature"]) * valid[:, None]
            task_vector = (self.model.backward(next_observation) * task_weights).sum(0)
            if self.world_size > 1:
                distributed.all_reduce(task_vector)
            if task_vector.norm() > 1e-8:
                self.task_command = project_task(task_vector)
        auxiliary_loss = state.new_zeros(())
        auxiliary_grad = state.new_zeros(())
        if self.options["auxiliary_coefficient"] > 0:
            with torch.no_grad():
                target_value = self.target.auxiliary(next_state, next_action, command).min(0).values
                target_value = suffix["reward"].squeeze(-1) * self.options["reward_scale"] + suffix["discount"].squeeze(-1) * target_value
            auxiliary = self.model.auxiliary(state, suffix["action"], command)
            auxiliary_loss = ((auxiliary - target_value).square() * valid).sum() / valid.sum().clamp_min(1)
            auxiliary_grad = self._synchronize_step(
                auxiliary_loss, self.auxiliary_optimizer, self.model.auxiliary_maps.parameters()
            )
        # Critic weights are fixed during actor optimization, while action
        # derivatives are retained. Memory is recomputed using current weights.
        critics = list(self.model.forward_maps.parameters()) + list(self.model.auxiliary_maps.parameters())
        for parameter in critics:
            parameter.requires_grad_(False)
        memory, _ = self._training_memory(batch, True)
        action_logits = self.model.actor.action_logits(memory, command)
        proposed_action = action_logits.tanh()
        q_values = (self.model(state, proposed_action, command) * command).sum(-1).min(0).values
        auxiliary_values = self.model.auxiliary(state, proposed_action, command).min(0).values
        q_scale = (q_values.detach().abs() * valid).sum() / valid.sum().clamp_min(1)
        auxiliary_scale = (auxiliary_values.detach().abs() * valid).sum() / valid.sum().clamp_min(1)
        objective = q_values / q_scale.clamp_min(1)
        objective = objective + self.options["auxiliary_coefficient"] * auxiliary_values / auxiliary_scale.clamp_min(1)
        bounds_penalty = (action_logits.abs() - 1.1).clamp_min(0).square().sum(-1)
        bounds_loss = (bounds_penalty * valid).sum() / valid.sum().clamp_min(1)
        actor_loss = -(objective * valid).sum() / valid.sum().clamp_min(1)
        actor_loss = actor_loss + self.config.get("bounds_loss_coef", 0.0001) * bounds_loss
        with torch.no_grad():
            other_action = self.model.actor.act(memory.detach(), command.roll(1, 0))
            command_action_delta = (proposed_action.detach() - other_action).norm(dim=-1).mean()
        actor_grad = self._synchronize_step(actor_loss, self.actor_optimizer, self.model.actor.parameters())
        for parameter in critics:
            parameter.requires_grad_(True)
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.model.parameters()):
                target_parameter.lerp_(parameter, self.options["target_tau"])
        self.optimizer_step += 1
        with torch.no_grad():
            moment = backward.detach().T @ (backward.detach() * valid[:, None]) / valid.sum().clamp_min(1)
            eigenvalues = torch.linalg.eigvalsh(moment)
        return {
            "loss/fb": fb_loss.detach(), "loss/orthogonal": orthogonal_loss.detach(),
            "loss/auxiliary": auxiliary_loss.detach(), "loss/actor": actor_loss.detach(),
            "loss/bounds": bounds_loss.detach(),
            "gradient/representation": representation_grad, "gradient/actor": actor_grad,
            "gradient/auxiliary": auxiliary_grad, "value/fb": q_values.detach().mean(),
            "value/auxiliary": auxiliary_values.detach().mean(),
            "representation/backward_norm": backward.detach().norm(dim=-1).mean(),
            "representation/min_eigenvalue": eigenvalues.min(),
            "representation/max_eigenvalue": eigenvalues.max(),
            "representation/effective_rank": moment.trace().square() / moment.square().sum().clamp_min(1e-8),
            "replay/valid_fraction": valid.mean(), "task/relabel_fraction": relabel.float().mean(),
            "policy/action_saturation": (proposed_action.detach().abs() > .95).float().mean(),
            "policy/command_action_delta": command_action_delta,
        }

    def _log(self, metrics):
        names = sorted(metrics)
        values = torch.stack([torch.as_tensor(metrics[name], device=self.device).float() for name in names])
        if self.world_size > 1:
            distributed.all_reduce(values)
            for index, name in enumerate(names):
                if name not in ("rollout/completed_episodes", "rollout/episode_return_sum") and not name.startswith("reward_episode_sum/"):
                    values[index].div_(self.world_size)
        result = {name: float(value) for name, value in zip(names, values.cpu())}
        if result.get("rollout/completed_episodes", 0) > 0:
            result["rollout/episode_return"] = result["rollout/episode_return_sum"] / result["rollout/completed_episodes"]
        if not all(np.isfinite(value) for value in result.values()):
            raise FloatingPointError("Nonfinite diagnostic metric")
        for name, value in list(result.items()):
            if name.startswith("reward_episode_sum/") and result.get("rollout/completed_episodes", 0) > 0:
                result[name.replace("reward_episode_sum/", "reward_episode/")] = value / result["rollout/completed_episodes"]
        if self.global_rank == 0:
            result.update(environment_steps=self.frame, optimizer_step=self.optimizer_step, epoch=self.epoch_num)
            with (self.output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            for name, value in result.items():
                self.writer.add_scalar(name, value, self.optimizer_step)
            import wandb
            if wandb.run is not None:
                wandb.log({**result, "global_step": self.optimizer_step})
            print(json.dumps(result, allow_nan=False), flush=True)
        return result

    def save(self, filename, replay=False):
        filename = Path(filename)
        if self.global_rank == 0:
            checkpoint = {
                "model": self.model.state_dict(), "target": self.target.state_dict(),
                "observation_normalizer": self.observation_normalizer.state_dict(),
                "state_normalizer": self.state_normalizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "representation_optimizer": self.representation_optimizer.state_dict(),
                "auxiliary_optimizer": self.auxiliary_optimizer.state_dict(),
                "frame": self.frame, "epoch": self.epoch_num, "optimizer_step": self.optimizer_step,
                "task_command": self.task_command, "options": self.options,
                "elapsed": self.elapsed_before_resume + time.monotonic() - self.run_started,
                "schema": {"observation": self.observation_size, "state": self.state_size, "action": self.action_size},
            }
            temporary = filename.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(filename)
        if replay:
            shard = filename.with_name(filename.stem + ".rank" + str(self.global_rank) + ".replay.pt")
            torch.save({
                "replay": self.replay.state_dict(), "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(), "environment": self.vec_env.get_env_state(),
                "observations": self.observations, "actor_state": self.actor_state,
                "rollout_command": self.rollout_command, "command_age": self.command_age,
                "reset_before": self.reset_before, "episode_reward": self.episode_reward,
                "previous_block": self.previous_block, "previous_initial_state": self.previous_initial_state,
            }, shard)
        if self.world_size > 1:
            distributed.barrier()
        self.last_checkpoint = filename

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        expected = {"observation": self.observation_size, "state": self.state_size, "action": self.action_size}
        if checkpoint["schema"] != expected:
            raise ValueError("Checkpoint and environment observation/state/action schemas differ")
        self.model.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint["target"])
        self.observation_normalizer.load_state_dict(checkpoint["observation_normalizer"])
        self.state_normalizer.load_state_dict(checkpoint["state_normalizer"])
        self.task_command = checkpoint["task_command"]
        self.frame = checkpoint["frame"]
        self.epoch_num = checkpoint["epoch"]
        self.optimizer_step = checkpoint["optimizer_step"]
        self.elapsed_before_resume = checkpoint["elapsed"]
        for name in ("actor_optimizer", "representation_optimizer", "auxiliary_optimizer"):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.last_checkpoint = Path(filename)

    def train(self):
        self.run_started = time.monotonic()
        self._initialize_logging()
        self._initialize_rollout()
        if self.last_checkpoint is not None:
            shard = self.last_checkpoint.with_name(self.last_checkpoint.stem + ".rank" + str(self.global_rank) + ".replay.pt")
            if shard.exists():
                saved = torch.load(shard, map_location=self.device)
                self.replay.load_state_dict(saved["replay"])
                # Retain off-policy experience, but start fresh episodes. Old
                # physical tensors are not a GPU PhysX solver snapshot; copying
                # them into a new scene crashed its broadphase in the GPU check.
                for name in ("success_tolerance", "frame_since_restart", "last_curriculum_update"):
                    if name in saved["environment"]:
                        setattr(self.vec_env.env, name, saved["environment"][name])
                torch.set_rng_state(saved["rng"].cpu())
                torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), self.device)
                np.random.set_state(saved["numpy_rng"])
                random.setstate(saved["python_rng"])
            elif not self.options.get("allow_fresh_replay", False):
                raise FileNotFoundError("Training resume requires its rank replay shard")
        initial_frame = self.frame
        self._status("running", schema=self.env_info.__repr__())
        last_metrics = {}
        try:
            while self.epoch_num < self.config["max_epochs"]:
                iteration_started = time.monotonic()
                metrics = self.collect()
                rollout_time = time.monotonic() - iteration_started
                if self.replay.size >= self.options["warmup_blocks"]:
                    for _ in range(self.options["updates_per_iteration"]):
                        metrics.update(self.update())
                self.epoch_num += 1
                elapsed = time.monotonic() - self.run_started
                rate = (self.frame - initial_frame) / max(elapsed, 1e-6)
                remaining = max(0, self.options["target_frames"] - self.frame) / max(rate, 1)
                metrics.update({
                    "timing/rollout_seconds": rollout_time,
                    "timing/update_seconds": time.monotonic() - iteration_started - rollout_time,
                    "timing/environment_steps_per_second": rate,
                    "timing/projected_remaining_hours": remaining / 3600,
                    "timing/elapsed_hours": (elapsed + self.elapsed_before_resume) / 3600,
                    "optimizer/learning_rate": self.actor_optimizer.param_groups[0]["lr"],
                    "memory/peak_gpu_gib": torch.cuda.max_memory_allocated(self.device) / 2 ** 30,
                    "replay/blocks": self.replay.size,
                })
                if self.epoch_num % self.options["log_interval"] == 0:
                    last_metrics = self._log(metrics)
                decision = torch.zeros((), device=self.device, dtype=torch.long)
                if self.global_rank == 0:
                    stop_requested = (self.output_dir / "stop.request").exists()
                    slow = elapsed >= self.options["speed_gate_seconds"] and remaining + elapsed + self.elapsed_before_resume > self.options["max_seconds"]
                    complete = self.frame >= self.options["target_frames"]
                    expired = elapsed + self.elapsed_before_resume >= self.options["max_seconds"]
                    if complete:
                        decision.fill_(1)
                    elif stop_requested or expired:
                        decision.fill_(2)
                    elif slow and self.options["stop_on_slow"]:
                        decision.fill_(3)
                if self.world_size > 1:
                    distributed.broadcast(decision, 0)
                if self.epoch_num % self.config["save_frequency"] == 0 or decision.item():
                    self.save(self.nn_dir / "last.pt", replay=bool(decision.item()))
                if self.global_rank == 0:
                    self._status("running", elapsed_seconds=elapsed, projected_remaining_hours=remaining / 3600, **last_metrics)
                if decision.item():
                    state = {1: "trained", 2: "budget_stopped", 3: "speed_gate"}[decision.item()]
                    self._status(state, elapsed_seconds=elapsed, projected_remaining_hours=remaining / 3600, **last_metrics)
                    break
            else:
                self.save(self.nn_dir / "last.pt", replay=True)
                self._status("iteration_limit", **last_metrics)
        except Exception as error:
            self._status("failed", error=repr(error))
            raise
        finally:
            if self.writer is not None:
                self.writer.close()
                import wandb
                if wandb.run is not None:
                    wandb.finish()
        return 0.0, self.epoch_num


class FBPlayer(FBAgent):
    def __init__(self, params):
        super().__init__("fb_eval", params)

    @torch.no_grad()
    def run(self):
        self.run_started = time.monotonic()
        self._initialize_rollout()
        self.model.eval()
        returns = torch.zeros(self.num_actors, device=self.device)
        completed = []
        steps = self.options["evaluation_steps"]
        environment = self.vec_env.env
        for step in range(steps):
            memory, self.actor_state = self.model.actor.encode(
                self.observation_normalizer(self.observations["obs"]).unsqueeze(0),
                self.actor_state, self.reset_before.float().unsqueeze(0),
            )
            command = self.task_command.expand(self.num_actors, -1)
            action = self.model.actor.act(memory[0], command)
            observations, reward, done, info = self.vec_env.step(action)
            returns += reward.reshape(-1)
            for index in done.nonzero(as_tuple=False).flatten().tolist():
                completed.append({
                    "environment_id": index, "end_step": step + 1,
                    "return": float(returns[index]),
                    "successes": float(environment.successes[index]),
                    "goal_fraction": float(environment.successes[index]) / environment.max_consecutive_successes,
                    "keypoint_distance": float(environment.closest_keypoint_max_dist[index]),
                })
            returns.masked_fill_(done.bool(), 0)
            self.observations = {key: value.clone() for key, value in observations.items()}
            self.reset_before = done.bool()
        summary = {
            "episodes": len(completed), "steps": steps,
            "mean_return": float(np.mean([row["return"] for row in completed])) if completed else None,
            "mean_successes": float(np.mean([row["successes"] for row in completed])) if completed else None,
            "mean_goal_fraction": float(np.mean([row["goal_fraction"] for row in completed])) if completed else None,
            "checkpoint": str(self.last_checkpoint), "records": completed,
            "object_name": environment.cfg["env"]["objectName"],
            "fixed_goal_path": environment.cfg["env"].get("fixedGoalStatesJsonPath"),
            "training_environment_steps": self.frame,
            "evaluation_tolerance": float(environment.success_tolerance),
            "schema": {"observation": self.observation_size, "state": self.state_size, "action": self.action_size},
        }
        (self.output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        self._status("evaluated", episodes=len(completed), mean_successes=summary["mean_successes"])
        print(json.dumps({key: value for key, value in summary.items() if key != "records"}), flush=True)
        return summary
