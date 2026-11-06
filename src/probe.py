import os
import sys,json,argparse,numpy as np
from pathlib import Path
from policy_rpc import RemotePolicy,unpack
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();ap=argparse.ArgumentParser();ap.add_argument('spec');ap.add_argument('--port',type=int,default=18308);a=ap.parse_args();spec=json.loads(Path(a.spec).read_text());rpc=RemotePolicy('http://127.0.0.1:'+str(a.port));rows=[]
for seed,step in spec['observations']:
 d=unpack((r/f'results/baseline/dev/{seed}/obs_{step:04d}.npz').read_bytes())
 def infer(patch):
  rpc.call('/patch',dict(patch=patch));rpc.reset(123456+seed+step);return rpc.predict(d['observation'],d['instruction'])['actions']
 base=infer(None);repeat=infer(None)
 for index,delta in spec['probes']:
  patch=dict(id=f'probe_{index}_{delta}',edits=[dict(name='model.action_expert.output_proj.bias',index=index,delta=delta)]);edited=infer(patch);diff=edited-base
  rows.append(dict(seed=seed,step=step,index=index,delta=delta,baseline_repeat_max=float(np.max(abs(repeat-base))),first8_mean_delta=diff[:8].mean(0).tolist(),first8_max_abs_delta=np.max(abs(diff[:8]),axis=0).tolist(),finite=bool(np.isfinite(edited).all())))
  np.savez_compressed(r/f'results/probe_{seed}_{step}_{index}_{delta}.npz',base=base,edited=edited)
rpc.call('/patch',dict(patch=None));(r/'results'/spec['output']).write_text(json.dumps(rows,indent=2));print('SAVED',len(rows),'forward comparisons',flush=True)
