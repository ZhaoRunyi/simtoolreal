from types import SimpleNamespace
from pathlib import Path
import tempfile
import torch
from rl_games.algos_torch.fb_evaluation import EvaluationRecorder

def test_aligned_trajectory_and_native_step_unchanged():
    with tempfile.TemporaryDirectory() as folder:
        environment=SimpleNamespace(reset_buf=torch.tensor([1,0]),lifted_object=torch.tensor([False,False]),
            successes=torch.zeros(2),keypoints_max_dist=torch.ones(2),object_state=torch.zeros(2,13),
            object_init_state=torch.zeros(2,13),goal_keypoint_pos=torch.zeros(2,4,3),control_dt=1/60,
            arm_hand_dof_lower_limits=torch.zeros(29),arm_hand_dof_upper_limits=torch.ones(29))
        def step(action):
            environment.lifted_object=torch.tensor([True,True])
            return {'states':torch.ones(2,162),'obs':torch.ones(2,140)},torch.tensor([2.,3.]),torch.tensor([0,1]),{'episode_cumulative':{'a':torch.tensor([2.,3.])}}
        vec=SimpleNamespace(env=environment,step=step)
        player=SimpleNamespace(vec_env=vec,num_actors=2,observations={'states':torch.zeros(2,162),'obs':torch.zeros(2,140)},output_dir=Path(folder),frame=12,last_checkpoint=Path('fake.pt'))
        recorder=EvaluationRecorder(player)
        result=vec.step(torch.ones(2,29))
        summary={}
        recorder.finish(summary)
        saved=torch.load(Path(folder)/'trajectory.pt')
        row=saved['trajectory']
        assert row['state'].shape==(1,2,162)
        assert not row['state'].any() and row['next_state'].all()
        assert row['new_lift'].tolist()==[[False,True]]
        assert row['done'].tolist()==[[False,True]]
        assert torch.equal(row['reward'][0],result[1])
        assert vec.step is step and saved['reward_names']==['a']
        assert summary['trajectory']=='trajectory.pt'
