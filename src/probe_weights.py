import os
import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0)
from policy_rpc import RemotePolicy,unpack
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();rpc=RemotePolicy('http://127.0.0.1:18400');name='model.action_expert.output_proj.weight';matrix=rpc.call('/inspect',{'names':[name]})[name];specs=[('right_base_feature200_p0p08',200,.08),('right_base_feature168_p0p20',168,.20)];patches=[]
for cid,col,delta in specs:
 patch=dict(id=cid,base_checkpoint_sha256='300d24c691f8d76846a218b9bd670312cf577b701ea5863f221af1b8d1c0d637',edits=[dict(name=name,index=10*1024+col,indices=[10,col],delta=delta,original_value=float(matrix[10,col]))],changed_scalars=1,hypothesis='One learned output-projection coefficient on a feature stronger during rack approach than during regrasp. Activation contrast motivated location; forward interventions test causal action effects. Not an interpreted semantic neuron.');patches.append(patch);(r/f'patches/{cid}.json').write_text(json.dumps(patch,indent=2))
rows=[]
for seed,last in [(200000,280),(200002,240),(200003,280),(200009,240),(200010,240),(200013,240)]:
 for phase,step in [('regrasp',160),('approach',last)]:
  d=unpack((r/f'results/baseline/dev/{seed}/obs_{step:04d}.npz').read_bytes())
  def infer(patch):
   rpc.call('/patch',dict(patch=patch));rpc.reset(123456+seed+step);return rpc.predict(d['observation'],d['instruction'])['actions']
  b=infer(None);repeat=infer(None)
  for patch in patches:
   a=infer(patch);diff=a-b;rows.append(dict(candidate=patch['id'],seed=seed,step=step,phase=phase,repeat_max=float(abs(repeat-b).max()),finite=bool(np.isfinite(a).all()),mean_delta=diff[:8].mean(0).tolist(),max_abs_delta=np.abs(diff[:8]).max(0).tolist()));np.savez_compressed(r/f'results/weight_probe_{patch["id"]}_{seed}_{step}.npz',base=b,edited=a)
rpc.call('/patch',dict(patch=None));(r/'results/weight_forward_sensitivity.json').write_text(json.dumps(dict(row10_weight_std=float(matrix[10].std()),probes=rows),indent=2));print('SAVED',len(rows),'forward comparisons',flush=True)
