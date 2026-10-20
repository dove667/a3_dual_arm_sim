"""Locate the first policy-input divergence across two same-seed closed-loop prefixes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark, SmolVLAPolicyAdapter
r=project_root();settings=load_randomization_config(r/'configs/randomization.yaml');settings['profile']='basic';settings['source_column']='1'
bench=CookieBatchBenchmark(r/'configs/cookie_batch.yaml',profile='basic',source_column='1',randomization_settings=settings,max_steps=100)
ckpt=r/'outputs/cookie_2x5_frontfirst_basic_medium_left11_40k_resume_v1/model/checkpoints/022000/pretrained_model'
policy=SmolVLAPolicyAdapter(ckpt,dataset_root=r/'datasets/cookie_2x5_frontfirst_basic_medium_left11_v1',repo_id='local/cookie-2x5-frontfirst-basic-medium-left11-v1',device='cuda',n_action_steps=8,num_steps=25,inference_seed=13000)
original=policy.act
runs=[];rows=[]
def act(obs,task):
    hashes={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in obs.items() if k.startswith('observation.images.')}
    action=original(obs,task)
    rows.append({'state':np.asarray(obs['observation.state']).tolist(),'eef':np.asarray(obs['observation.eef_pose']).tolist(),'images':hashes,'raw_images':{k:np.asarray(obs[k]).copy() for k in hashes},'action':action.tolist()})
    return action
policy.act=act
try:
    for _ in range(2):
        rows=[]
        bench.run_episode(policy,13300)
        runs.append(rows)
finally:policy.close()
fields=['state','eef','action']
result={'frames':[len(x) for x in runs]}
for field in fields:
    a=np.asarray([r[field] for r in runs[0]]);b=np.asarray([r[field] for r in runs[1]])
    d=np.max(np.abs(a-b),axis=1)
    result[field]={'first_diff_over_1e-6':next((i for i,v in enumerate(d) if v>1e-6),None),'max_diff':float(d.max())}
for k in runs[0][0]['images']:
    d=[i for i,(a,b) in enumerate(zip(runs[0],runs[1])) if a['images'][k]!=b['images'][k]]
    result[k]={'first_hash_diff':d[0] if d else None,'different_frames':len(d)}
    early=[(runs[0][i]['raw_images'][k].astype(np.int16)-runs[1][i]['raw_images'][k].astype(np.int16)) for i in range(min(16,len(runs[0])))]
    result[k]['first_16_raw_max_abs_diff']=int(max(np.abs(v).max() for v in early))
    result[k]['first_16_raw_mean_abs_diff']=float(np.mean([np.abs(v).mean() for v in early]))
Path(__file__).with_name('closed_loop_repeat.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
