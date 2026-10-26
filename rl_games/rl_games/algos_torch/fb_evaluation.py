"""Compact, aligned evaluation trajectories for physical and reward metrics."""
import torch


class EvaluationRecorder:
    def __init__(self, player):
        self.player = player
        self.environment = player.vec_env.env
        self.original_step = player.vec_env.step
        self.rows = []
        self.reward_names = None
        player.vec_env.step = self.step

    @torch.no_grad()
    def step(self, action):
        environment = self.environment
        before_reset = environment.reset_buf.bool().clone()
        previous_lifted = environment.lifted_object.clone()
        row = {
            'observation': self.player.observations['obs'].detach().cpu().clone(),
            'state': self.player.observations['states'].detach().cpu().clone(),
            'action': action.detach().cpu().clone(),
            'reset': before_reset.cpu(),
        }
        result = self.original_step(action)
        observations, reward, done, info = result
        components = info['episode_cumulative']
        if self.reward_names is None:
            self.reward_names = [name for name in components if not name.startswith('raw_')]
        row.update({
            'next_state': observations['states'].detach().cpu().clone(),
            'reward': reward.reshape(-1).detach().cpu(),
            'reward_components': torch.stack([components[name] for name in self.reward_names], -1).detach().cpu(),
            'done': done.bool().reshape(-1).cpu(),
            'lifted': environment.lifted_object.detach().cpu(),
            'new_lift': (~before_reset & ~previous_lifted & environment.lifted_object).cpu(),
            'successes': environment.successes.detach().cpu(),
            'goal_distance': environment.keypoints_max_dist.detach().cpu(),
            'object_state': environment.object_state.detach().cpu().clone(),
            'object_initial_height': environment.object_init_state[:, 2].detach().cpu().clone(),
            'goal_keypoints': environment.goal_keypoint_pos.reshape(self.player.num_actors, -1).detach().cpu(),
        })
        self.rows.append(row)
        return result

    def finish(self, summary):
        self.player.vec_env.step = self.original_step
        arrays = {name: torch.stack([row[name] for row in self.rows]) for name in self.rows[0]}
        if not all(torch.isfinite(value).all() for value in arrays.values()):
            raise ValueError('Nonfinite evaluation trajectory')
        destination = self.player.output_dir / 'trajectory.pt'
        torch.save({'schema_version': 1, 'reward_names': self.reward_names,
                    'control_dt': self.environment.control_dt,
                    'checkpoint': str(self.player.last_checkpoint), 'frame': self.player.frame,
                    'joint_lower_limits': self.environment.arm_hand_dof_lower_limits.detach().cpu(),
                    'joint_upper_limits': self.environment.arm_hand_dof_upper_limits.detach().cpu(),
                    'trajectory': arrays}, destination)
        summary['trajectory'] = destination.name
        summary['trajectory_schema_version'] = 1
