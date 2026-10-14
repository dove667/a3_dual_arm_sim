"""Read-only dynamics audit; run with MUJOCO_GL=egl python <this file>."""
import json
import platform
import subprocess
from pathlib import Path
import mujoco
import numpy as np
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from a3_dual_arm_sim.controllers.batch_expert import A3CookieBatchExpert
from a3_dual_arm_sim.contracts import FRONT_IMAGE

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / 'configs/cookie_batch.yaml'

def env(mode='joint_position', cameras=False):
    return A3CookieTransferEnv(CONFIG, action_mode=mode, render_cameras=cameras)

result = {'revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'python':platform.python_version(),'mujoco':mujoco.__version__, 'config':str(CONFIG),
          'physics_hz':1000,'control_hz':20,'camera_rendering_replay':False}
# Compare dynamics after replaying the exact saved commands, including contact.
replays=[]
for seed in (0,1,2):
    e=env(); r=env()
    try:
        obs,_=e.reset(seed=seed); r.reset(seed=seed)
        expert=A3CookieBatchExpert(e)
        qerr=aerr=0.; stopped=False
        for step in range(160):
            obs,_,t,_,info=e.step(expert.act(obs))
            _,_,rt,_,ri=r.step(info['applied_action'])
            qerr=max(qerr,float(np.max(np.abs(e.data.qpos-r.data.qpos))))
            aerr=max(aerr,float(np.max(np.abs(info['applied_action']-ri['applied_action']))))
            if t or rt:
                stopped=True; break
        replays.append(dict(seed=seed,steps=step+1,max_qpos_difference=qerr,
                            max_action_difference=aerr,safety_stop=stopped,phase=expert.phase.name))
    finally:
        e.close(); r.close()
result['expert_command_replay']=replays
# Forward on a separate data copy avoids changing the probed trajectory.
e=env(cameras=True)
try:
    e.reset(seed=0); target=e.last_applied_action.copy(); target[0]+=0.4
    stale=tracking=pixelmax=0.; pixelmean=0.
    for _ in range(30):
        obs,_,t,_,_=e.step(target)
        work=mujoco.MjData(e.model); mujoco.mj_copyData(work,e.model,e.data)
        mujoco.mj_forward(e.model,work)
        stale=max(stale,float(np.max(np.linalg.norm(e.data.site_xpos[list(e._eef_sites)]-work.site_xpos[list(e._eef_sites)],axis=1))))
        tracking=max(tracking,float(np.max(np.abs(e.current_joint_action[:7]-e.last_applied_action[:7]))))
        e._renderer.update_scene(work,camera=e._camera_ids['front'])
        fresh=e._renderer.render().copy()
        diff=np.abs(fresh.astype(float)-obs[FRONT_IMAGE].astype(float))
        pixelmean=max(pixelmean,float(diff.mean())); pixelmax=max(pixelmax,float(diff.max()))
    result['observation_sync']={'max_eef_stale_m':stale,'max_front_pixel_difference':pixelmax,
      'max_front_mean_pixel_difference':pixelmean,'max_tracking_error_rad':tracking,
      'final_tracking_error_rad':float(np.max(np.abs(e.current_joint_action[:7]-e.last_applied_action[:7]))),
      'sim_time':e.data.time,'steps':30,'safety_stop':t}
finally: e.close()
# Same Cartesian request, different constructor mode.
e=env(); c=env('cartesian_delta')
try:
    e.reset(seed=0); c.reset(seed=0)
    a=np.zeros(14); a[0]=1.; a[2]=1.; a[6]=a[13]=1.
    _,_,_,_,ei=e.step(a); _,_,_,_,ci=c.step(a)
    result['constructor_mode_difference']={'cartesian_request':a.tolist(),
      'max_applied_action_difference_rad':float(np.max(np.abs(ei['applied_action']-ci['applied_action'])))}
    try: c.step(c.last_applied_action)
    except ValueError as exc: result['joint_replay_in_cartesian_env_error']=str(exc)
finally: e.close(); c.close()
print(json.dumps(result,indent=2))
Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
