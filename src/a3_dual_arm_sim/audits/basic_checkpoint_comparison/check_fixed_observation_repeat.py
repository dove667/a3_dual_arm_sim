"""Test inference RNG on one immutable observation and one loaded policy instance."""
import json
from pathlib import Path
import numpy as np
from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark, SmolVLAPolicyAdapter
r=project_root();settings=load_randomization_config(r/'configs/randomization.yaml');settings['profile']='basic';settings['source_column']='1'
bench=CookieBatchBenchmark(r/'configs/cookie_batch.yaml',profile='basic',source_column='1',randomization_settings=settings,max_steps=500)
env=bench.create_env(render_cameras=True)
try:
    obs,info=env.reset(seed=13300,options={'randomize_cookies':True,'randomize_boxes':True,'randomize_source_bin':True,'randomize_target_bin':True,'source_bin_noise_m':bench.source_bin_noise_m,'target_bin_noise_m':bench.target_bin_noise_m,'target_bin_yaw_noise_rad':bench.target_bin_yaw_noise_rad,'randomization_profile':'basic','appearance_seed':13300,'randomization_settings':settings})
    checkpoint=r/'outputs/cookie_2x5_frontfirst_basic_medium_left11_40k_resume_v1/model/checkpoints/022000/pretrained_model'
    policy=SmolVLAPolicyAdapter(checkpoint, dataset_root=r/'datasets/cookie_2x5_frontfirst_basic_medium_left11_v1',repo_id='local/cookie-2x5-frontfirst-basic-medium-left11-v1',device='cuda',n_action_steps=8,num_steps=25,inference_seed=13000)
    policy.bind_env(env)
    cycles=[]
    context=EpisodeContext(seed=13300,task='transfer 10 cookies into target box',action_mode='joint_position')
    for _ in range(2):
        policy.reset(context)
        cycles.append(np.asarray([policy.act(obs,context.task) for step in range(24)]))
    diff=np.max(np.abs(cycles[0]-cycles[1]),axis=1)
    result={'per_step_max_action_difference':diff.tolist(),'first_difference_over_1e-6':next((i for i,d in enumerate(diff) if d>1e-6),None),'max_difference':float(diff.max())}
    Path(__file__).with_name('fixed_observation_repeat.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
finally:
    if 'policy' in locals():policy.close()
    env.close()
