"""Wait for audited collection, then fine-tune frozen-contract 22k for two frame epochs."""
import fcntl, json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
from a3_dual_arm_sim.data.audit import audit_training_dataset
from a3_dual_arm_sim.learning.training import train_smolvla
ROOT = Path('/home/eter/桌面/workspace/a3_dual_arm_sim')
OUT = ROOT/'src/a3_dual_arm_sim/audits/second_round_finetune_v1'
lock = (OUT/'run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')
def status(value):
    save('status.json', {'status':value,'time':time.time(),'pid':os.getpid()})
try:
    production=ROOT/'datasets/cookie_second_round_medium100_v2'
    status('waiting_for_100_audited_episodes')
    while not (production/'TRAINING_DATA_READY.json').exists():
        time.sleep(30)
    source=production/'training_left11'
    report=audit_training_dataset(source,repo_id='local/cookie-second-round-medium100-left11')
    assert (report['episodes'],report['state_dim'],report['action_dim'])==(100,11,8),report
    summary=json.loads((source/'collection_summary.json').read_text())
    assert summary['category_counts']=={'normal':40,'offset':30,'empty_recovery':20,'first_stack_variation':10},summary
    # Preserve the old policy's coordinate transform while retaining raw collection statistics.
    view=OUT/'dataset_fixed_normalization'
    view.mkdir(exist_ok=True)
    for path in source.iterdir():
        destination=view/path.name
        if destination.exists() or destination.is_symlink():
            continue
        if path.name=='meta': shutil.copytree(path,destination)
        else: destination.symlink_to(path,target_is_directory=path.is_dir())
    old=ROOT/'datasets/cookie_2x5_frontfirst_basic_medium_left11_v1'
    shutil.copy2(view/'meta/stats.json',OUT/'new_data_stats.json')
    shutil.copy2(old/'meta/stats.json',view/'meta/stats.json')
    base=ROOT/'outputs/cookie_2x5_frontfirst_basic_medium_left11_40k_resume_v1/model/checkpoints/022000/pretrained_model'
    epoch=math.ceil(report['frames']/16)
    save('protocol.json',{'base':str(base),'source_dataset':str(source),'dataset_audit':report,'batch_size':16,'steps_per_frame_epoch':epoch,'steps':2*epoch,'epochs':2,'lr':5e-6,'decay_lr':1e-6,'warmup_steps':min(100,max(1,epoch//10)),'normalization_stats_source':str(old/'meta/stats.json'),'resume_optimizer':False,'scope':'second_round_suffix_only'})
    status('training')
    result=train_smolvla(dataset_root=view,repo_id='local/cookie-second-round-medium100-left11',base_model=base,output_dir=OUT/'model',steps=2*epoch,batch_size=16,lr=5e-6,decay_lr=1e-6,warmup_steps=min(100,max(1,epoch//10)),save_freq=epoch,num_workers=2,seed=1000,device='cuda',offline=True)
    save('result.json',result)
    status('evaluating_basic')
    checkpoints=[('base22k',base),('epoch1',OUT/'model/checkpoints'/f'{epoch:06d}'/'pretrained_model'),('epoch2',Path(result['checkpoint']))]
    for label, checkpoint in checkpoints:
        command=[sys.executable,str(ROOT/'examples/benchmark_cookie_batch.py'),'--task','2x5','--profile','basic','--source-column','1','--policy',f'smolvla:{checkpoint}','--dataset-root',str(old if label=='base22k' else view),'--repo-id','local/cookie-2x5-frontfirst-basic-medium-left11-v1' if label=='base22k' else 'local/cookie-second-round-medium100-left11','--device','cuda','--prompt-style','training','--episodes','5','--seed-start','13400','--inference-seed','13000','--max-steps','500','--n-action-steps','8','--num-steps','25','--workers','1','--diagnostic-dir',str(OUT/label),'--output',str(OUT/f'{label}.json')]
        with (OUT/f'{label}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    status('completed')
except BaseException:
    status('failed')
    traceback.print_exc()
    raise
