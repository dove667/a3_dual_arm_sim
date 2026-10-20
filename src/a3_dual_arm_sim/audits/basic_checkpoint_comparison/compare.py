"""Summarize paired basic tests using recorded physical motion, not gripper closure alone."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OTHER=ROOT.parent/'basic_ckpt_30k_diag'

def trace_metrics(path):
    rows=[json.loads(line) for line in path.open() if line.strip()]
    cookie=np.asarray([row['after']['cookie_positions'] for row in rows])
    lifted=(cookie[:,:,2]-cookie[0,:,2]).max(axis=0)
    target=np.asarray([row['after']['cookies_in_target'] for row in rows])
    geoms=[]
    for row in rows:
        names={str(x) for c in row['after']['contacts'] for x in (c['geom1'],c['geom2']) if 'L_finger' in str(x)}
        geoms.append(len(names)>=2)
    longest=run=0
    for hit in geoms:
        run=run+1 if hit else 0
        longest=max(longest,run)
    return {'max_cookie_lift_mm':round(float(lifted.max()*1000),2),
            'cookies_lifted_10mm':int(sum(lifted>=.01)),
            'max_target_cookies':int(target.max()),
            'longest_dual_finger_contact_steps':longest,
            'dual_finger_contact_steps':sum(geoms),
            'trace_steps':len(rows)}

def report(name):
    if name=='030000':
        data=[]
        for f in [OTHER/'seed_13300_cuda.json',OTHER/'seeds_13301_13304.json']:
            data.extend(json.loads(f.read_text())['episodes'])
        paths={e['seed']:OTHER/('seed_13300_cuda' if e['seed']==13300 else 'seeds_13301_13304')/f"seed_{e['seed']}"/'trace.jsonl' for e in data}
    else:
        parts=[name]
        if name in ('022000','020000_h8_n10') and (ROOT/f'{name}_tail2.json').exists():
            parts.append(name+'_tail2')
        data=[];paths={}
        for part in parts:
            f=ROOT/f'{part}.json'
            if not f.exists(): return None
            rows=json.loads(f.read_text())['episodes']
            data.extend(rows)
            paths.update({e['seed']:ROOT/part/f"seed_{e['seed']}"/'trace.jsonl' for e in rows})
        if len(data)!=len(paths): raise RuntimeError('Duplicate paired seed')
    episodes=[]
    for e in data:
        physical=trace_metrics(paths[e['seed']])
        if physical['trace_steps']!=e['steps']:raise RuntimeError('Trace mismatch '+str(paths[e['seed']]))
        episodes.append({'seed':e['seed'],'success':e['success'],'final_cookies':e['cookies_in_target'],**physical})
    return {'checkpoint_or_probe':name,'episodes':episodes,
            'n':len(episodes),'successes':sum(e['success'] for e in episodes),
            'final_cookies':sum(e['final_cookies'] for e in episodes),
            'episodes_with_five_lifted':sum(e['cookies_lifted_10mm']>=5 for e in episodes),
            'episodes_with_five_target_transient':sum(e['max_target_cookies']>=5 for e in episodes)}

if __name__=='__main__':
    names=['020000','022000','024000','028000','030000','032000','020000_h4_n25','020000_h8_n10','030000_h4_n25','022000_h8_n10','022000_h8_n25_m650']
    output={name:report(name) for name in names}
    output={k:v for k,v in output.items() if v is not None}
    (ROOT/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    for n,x in output.items():
        print(n,f"success={x['successes']}/{x['n']}",f"cookies={x['final_cookies']}/{x['n']*10}",f"grasp_lift={x['episodes_with_five_lifted']}/{x['n']}",f"target_five_anytime={x['episodes_with_five_target_transient']}/{x['n']}")
