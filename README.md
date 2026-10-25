# SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation

[Project Page](https://simtoolreal.github.io/)

https://github.com/user-attachments/assets/e2d0db98-2e31-46aa-9480-c4c6f4a48f7d

# Overview

This repository is the official implementation of _SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation_. It provides:

* Simulation environments: Isaac Sim (recommended) and Isaac Gym (legacy) environments for training and evaluating dexterous tool manipulation policies.

* DexToolBench: A benchmark for dexterous tool manipulation.

* RL training: Reinforcement learning algorithms for training these policies.

* Deployment: Policy deployment in simulation and the real world.

# Project Structure

```
simtoolreal
  ├── assets
  │   └── // Robot URDFs, object models, and other assets
  ├── baselines
  │   └── // Kinematic retargeting and fixed-grasp baselines
  ├── deployment
  │   └── // Sim-to-real and sim-to-sim deployment of the policy
  ├── dextoolbench
  │   ├── data
  │   │   └── // DexToolBench data (needs to be downloaded)
  │   ├── // Scripts for evaluating policies on DexToolBench
  │   └── // Scripts for visualizing DexToolBench objects and trajectories
  ├── docs
  │   └── // Documentation
  ├── isaacsimenvs
  │   └── // Simulation environment (Isaac Sim / Isaac Lab — recommended)
  ├── isaacgymenvs
  │   └── // Simulation environment (Isaac Gym — legacy, used for the paper's results)
  ├── pretrained_policy
  │   └── // Checkpoint of the pretrained policy (needs to be downloaded)
  ├── recorded_data
  │   └── // Interface and tools for saving, loading, and visualizing recorded data
  └── rl_games
      └── // RL algorithms, including PPO and SAPG
```

**External repos:**
[FoundationPose](https://github.com/kushal2000/FoundationPose) — Perception system (SAM + FoundationPose pose tracking)

# Installation

The recommended setup runs SimToolReal in **Isaac Sim** (via Isaac Lab, Python 3.11, pip-installable): see the [IsaacSim Installation](docs/isaacsim_installation.md) documentation.

The legacy **Isaac Gym** environment (Python 3.8, manual binary download) is covered in the [IsaacGym Installation](docs/isaacgym_installation.md) documentation. The two environments live in separate venvs (`.venv_isaacsim` and `.venv`) and can coexist.

# Quick Start

Run all commands from the repository root. Most commands accept `--help` to list available options.

## 1. Download the Pretrained Policy

```
python download_pretrained_policy.py
```

This downloads `config.yaml` and `model.pth` into `pretrained_policy/`.

## 2. Interactive Evaluation on DexToolBench

Launch the web-based interactive demo (default `http://localhost:8080`): pick a tool category, object instance, and task from the dropdown menus, then load the environment and run episodes.

```
.venv_isaacsim/bin/python dextoolbench/eval_interactive_isaacsim.py \
--config-path pretrained_policy/config.yaml \
--checkpoint-path pretrained_policy/model.pth
```

On the Isaac Gym setup, run `dextoolbench/eval_interactive_isaacgym.py` (same arguments) from the `.venv` environment instead.

https://github.com/user-attachments/assets/58eb188b-662c-4190-8148-29710c9eb20f

## 3. Train a Policy

We recommend training in Isaac Sim. Training logs are tracked with [Weights & Biases](https://wandb.ai/); run `wandb login` before launching:

```
.venv_isaacsim/bin/python isaacsimenvs/train.py \
--task Isaacsimenvs-SimToolReal-Direct-v0 \
--agent rl_games_sapg_cfg_entry_point \
--headless \
--capture_viewer \
--wandb_activate \
--wandb_project <project> \
--wandb_entity <entity> \
env.scene.num_envs=24576 \
agent.params.config.expl_coef_block_size=4096
```

`--capture_viewer` periodically uploads an interactive 3D rollout viewer to W&B (pose-only, no cameras) so you can visually check the policy during training.

To finetune from a checkpoint, add `--checkpoint pretrained_policy/model.pth --checkpoint_load_mode weights` (`resume` also restores the optimizer state). If you run out of GPU memory, reduce `env.scene.num_envs` and `expl_coef_block_size` together (`num_envs / expl_coef_block_size` is the SAPG block count — keep it at 6).

### Isaac Gym (legacy)

The paper's results were trained with the Isaac Gym pipeline. It logs to [Weights & Biases](https://wandb.ai/); run `wandb login` and set `wandb_entity` in `isaacgymenvs/launch_training.py` before launching:

```
python isaacgymenvs/launch_training.py \
--custom_experiment_name my_experiment
```

To finetune from a checkpoint, add `--checkpoint <checkpoint_path>` (e.g. `pretrained_policy/model.pth`). If you run out of GPU memory, reduce `--num_envs` (must be divisible by `num_blocks`, default 6).

Generate a local 10-second interactive Three.js rollout from an Isaac Gym
checkpoint without camera rendering or W&B:

```bash
scripts/replay_checkpoint_gym.sh \
  --checkpoint runs/0_example/nn/checkpoint.pth \
  --output-dir train_dir/local_replays/example
```

The script defaults to the checkout's `.venv`; pass `--venv PATH` (or set
`SIMTOOLREAL_VENV`) to reuse a compatible Isaac Gym environment. Use `--task`,
`--train`, and repeatable `--override` arguments when the checkpoint was trained
with non-default Hydra configurations.

## 4. Evaluate a Policy on DexToolBench

Evaluate a policy numerically on all 24 DexToolBench combinations:

```
.venv_isaacsim/bin/python dextoolbench/run_all_evals_isaacsim.py
```

(Isaac Gym variant: `python dextoolbench/run_all_evals_isaacgym.py`.)

# DexToolBench

DexToolBench covers 6 tool categories (hammer, marker, eraser, brush, spatula, screwdriver) × 2 objects × 2 tasks each. See the [DexToolBench Reference](docs/dextoolbench.md) for dataset download, visualization tooling, creating new tasks and objects, real-world data collection, and acquiring the physical objects.

# Deployment

Sim2Real deployment runs four nodes: an RL policy node and a goal pose node (this repo), a perception node (SAM + FoundationPose, in the [FoundationPose fork](https://github.com/kushal2000/FoundationPose)), and a robot node. For safe testing, the Sim2Sim setup replaces the robot and perception nodes with a simulation node. See the [Deployment](docs/deployment.md) documentation for node diagrams and step-by-step run instructions.

# Forward-backward agent (`fb` branch)

`train=SimToolRealFB` registers a recurrent off-policy FB learner in the existing
rl_games runner. The original task, 140-dimensional policy observation,
asymmetric state, 29-dimensional action, domain randomization, goal resampling,
reward and tolerance curriculum are retained. The actor's LSTM sees policy
observations only; the task command is concatenated after memory. Backward maps
also see only policy observations. Forward and auxiliary critics see the
existing asymmetric state. No expert dataset or discriminator is assumed.

Local Python 3.8 / Isaac Gym setup (run in this checkout):

```bash
source /home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh
uv venv --python /usr/bin/python3.8 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-fb.txt
PYTORCH3D_NO_EXTENSION=1 uv pip install --python .venv/bin/python --no-build-isolation --no-deps \
  -e ../third_party/isaacgym/python -e ../third_party/pytorch3d -e ./rl_games -e .
source .venv/bin/activate
export LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
python -m pytest isaacgymenvs/tests/test_fb_agent.py -q
```

This training profile uses PyTorch3D transforms, not its renderer/CUDA operators.
The full ROS/perception/Isaac Sim dependency set is outside this profile.
Run GPU commands only inside a PBS allocation. A four-GPU run gives 24,576
global environments and 98,304 learner examples per rollout iteration:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m isaacgymenvs.train task=SimToolRealLSTMAsymmetric train=SimToolRealFB \
  num_envs=6144 multi_gpu=True headless=True force_render=False graphics_device_id=-1 \
  task.env.enableCameraSensors=False task.env.capture_video=False \
  'task.env.objectScaleNoiseMultiplierRange=[0.9,1.1]' \
  task.env.forceConsecutiveNearGoalSteps=True \
  experiment=fb_aux train.params.config.train_dir=/tmp/simtoolreal-fb \
  wandb_activate=True wandb_project=simtoolreal_fb wandb_entity=YOUR_ENTITY
```

Use `train.params.config.fb.auxiliary_coefficient=0` for the pure-FB control.
Architecture and optimizer settings remain identical. Both runs include random
tasks, replay-goal tasks and task-reward inference, and preserve original
rollout commands before learner relabeling. Training metrics go to JSONL,
TensorBoard and W&B; `status.json` reports lifecycle and projected runtime.
The speed gate saves a resumable checkpoint and stops if the configured sample
budget cannot fit the time budget after warm-up. A `stop.request` file in the
experiment directory also requests a coordinated checkpoint and stop.

Resume requires `last.pt` and the matching `last.rankN.replay.pt` shards;
pass `checkpoint=/absolute/path/last.pt`. The replay shards include recorded
sequences, recurrent context, environment audit state and RNG states. Recovery
restores learner/replay and scalar curriculum, then starts new episodes through
the original reset path. It does not write old physical tensors into a new
PhysX scene or claim bitwise restoration of the solver state.

Evaluate with the same architecture/config, a new seed and the learned reward
command. Evaluation does not initialize W&B:

```bash
python -m isaacgymenvs.train task=SimToolRealLSTMAsymmetric train=SimToolRealFB \
  test=True checkpoint=/absolute/path/last.pt seed=9000 num_envs=384 \
  headless=True force_render=False graphics_device_id=-1 \
  task.env.enableCameraSensors=False task.env.capture_video=False \
  'task.env.objectScaleNoiseMultiplierRange=[0.9,1.1]' \
  task.env.forceConsecutiveNearGoalSteps=True task.env.evalSuccessTolerance=0.01 \
  experiment=fb_eval train.params.config.train_dir=/tmp/simtoolreal-fb-eval
```

The native FB player writes episode-level `evaluation.json`. To evaluate a
specific DexToolBench object or fixed target sequence, use the original task's
`objectName`, `useFixedGoalStates`, `fixedGoalStatesJsonPath` and table settings;
these remain environment configuration, not a separate task implementation.

For the 600-step noisy tool-sequence protocol, add these overrides to the
evaluation command above (and use `num_envs=10`):

```bash
train.params.config.fb.evaluation_steps=600 \
task.env.objectName=claw_hammer task.env.startArmHigher=True \
task.env.useFixedGoalStates=True \
task.env.fixedGoalStatesJsonPath=/home/MLTL-INTERN/shaoyang/zvi/dex_fb_ws/SimToolReal-fb/dextoolbench/trajectories/hammer/claw_hammer/swing_down.json \
task.env.asset.table=urdf/dextoolbench/environments/hammer/claw_hammer/swing_down.urdf
```

This retains training-style noise and the 600-step horizon. The official
`dextoolbench/eval_isaacgym.py` protocol changes additional settings; report
the two protocols separately.

To record an MP4 through the existing camera recorder, add these evaluation
overrides inside a PBS allocation:

```bash
graphics_device_id=0 task.env.enableCameraSensors=True \
task.env.capture_video=True task.env.capture_video_freq=1 \
task.env.capture_video_len=600 train.params.config.fb.evaluation_steps=1200
```

Set `hydra.run.dir` to the desired normal run directory, such as
`train_dir/dbfm-v1/aux/video_eval/claw_hammer/swing_down`; the recorder writes
`videos/*.mp4` there. The original recorder starts after an environment reset,
so 1,200 evaluation steps allow a full recording while retaining the original
600-step episode horizon. Its headless path removes the first captured frame;
validate the actual encoded frame count and FPS. Keep these extended camera
recordings separate from the original 600-step numerical evaluation.

The design/implementation audit lives in the workspace's
`doc/D-BFM/v1/simtoolreal_fb_v1.md`.

The first formal matrix completed 2,000,289,792 transitions per variant on
2026-10-25, with recorded training times of 2.492 hours (pure) and 2.757 hours
(aux). All 50 evaluation cases were valid, covering 2,334 completed episodes;
neither variant completed a goal under the noisy stress protocol. Detailed
results, checkpoint audits and data-coverage measurements are in the workspace's
`doc/D-BFM/v1/simtoolreal_fb_v1_results.md`.

# Formatting

Python files:

```
./format_pys.sh
```

URDF files:

```
./format_urdfs.sh
```

# Acknowledgements

This implementation builds on the following codebases:

1. [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)
2. [rl_games](https://github.com/Denys88/rl_games)
3. [SAPG](https://github.com/jayeshs999/sapg)

# Citation

```
@misc{kedia2026simtoolrealobjectcentricpolicyzeroshot,
      title={SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation},
      author={Kushal Kedia and Tyler Ga Wei Lum and Jeannette Bohg and C. Karen Liu},
      year={2026},
      eprint={2602.16863},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.16863}
}
```

# Contact

If you have any questions, issues, or feedback, please contact [Tyler Lum](https://tylerlum.github.io/) or [Kushal Kedia](https://kushal2000.github.io/).


### FB-CPR v2 experiment

The `fb-v2` worktree adds BFM-Zero residual actor/critics, four-frame sensor
history, conditional imitation reward, expert prompts and reference-state resets.
The public observation/action contract remains 140/29. Design and current results:
[design](../doc/D-BFM/v2/simtoolreal_fb_v2.md),
[results](../doc/D-BFM/v2/simtoolreal_fb_v2_results.md).

Use the existing uv-managed Isaac Gym environment without copying dependencies:

```bash
export PYTHONPATH="$PWD:$PWD/rl_games"
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
/home/MLTL-INTERN/shaoyang/zvi/dex_fb_ws/SimToolReal-fb/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node=4 -m isaacgymenvs.train task=SimToolRealLSTMAsymmetric \
  train=SimToolRealFBV2 num_envs=6144 multi_gpu=True headless=True \
  task.env.clearActionQueueOnReset=True task.env.referenceResetProbability=1.0 \
  experiment=bfm_v2 train.params.config.train_dir=train_dir/dbfm-v2
```

Run GPU commands only within an allocated PBS job. Formal runs additionally set
camera/graphics options and the fixed evaluation protocol described in the design.
Teacher weights are the author's `pretrained_policy/config.yaml` and `model.pth`;
`train.params.config.fb.teacher_path` selects their absolute location. Collection
saves per-rank expert files under the normal run directory and validates contiguous
expert sequences before learner updates. Evaluation does not load reference resets.
BFM-Zero-derived network/loss components carry the upstream CC BY-NC 4.0 notice in
[docs/licenses/bfm-zero-LICENSE](docs/licenses/bfm-zero-LICENSE).
