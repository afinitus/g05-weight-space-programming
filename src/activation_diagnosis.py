import os
import sys,json,numpy as np
from pathlib import Path
from policy_rpc import RemotePolicy,unpack
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();rpc=RemotePolicy('http://127.0.0.1:18400');rpc.wait_ready();rows=[];early=[];late=[]
for seed,last in [(200000,280),(200002,240),(200003,280),(200009,240),(200010,240),(200013,240)]:
 for phase,step in [('regrasp',160),('approach',last)]:
  d=unpack((r/f'results/baseline/dev/{seed}/obs_{step:04d}.npz').read_bytes());o=rpc.call('/activations',dict(observation=d['observation'],instruction=d['instruction'],seed=123456+seed+step));f=o['features'];np.savez_compressed(r/f'results/activation_{seed}_{step}.npz',features=f,actions=o['actions']);print(seed,step,f.shape,flush=True)
  # [flow step,batch,action time,hidden]
  mean=f[:,:,:8,:].mean(axis=(0,1,2));(early if phase=='regrasp' else late).append(mean);rows.append(dict(seed=seed,step=step,phase=phase,shape=list(f.shape)))
early=np.array(early);late=np.array(late);ratio=np.abs(late.mean(0))/(np.abs(early).mean(0)+.1);consistent=(np.sign(late)==np.sign(late.mean(0))).mean(0);rank=np.argsort(-(ratio*consistent));top=[dict(column=int(j),ratio=float(ratio[j]),sign_consistency=float(consistent[j]),early=early[:,j].tolist(),late=late[:,j].tolist()) for j in rank[:20]];(r/'results/activation_diagnosis.json').write_text(json.dumps(dict(observations=rows,top_phase_contrasts=top,interpretation='Activation contrast only; not evidence of causal control without parameter interventions.'),indent=2));print('SAVED',flush=True)
