import os
import json,numpy as np
from pathlib import Path
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();cid=json.loads((r/'configs/final_edit.json').read_text())['id'];rows=[]
for c in ['baseline',cid]:
 for p in sorted((r/f'results/{c}/test').glob('*/result.json')):
  d=json.loads(p.read_text());z=np.load(p.parent/'trajectory.npz');rise=float(z['mug'][:,2].max()-d['initial']['mug_pose'][2]);q=z['joints'];ix=np.where((q[:-1,13]<.2)&(q[1:,13]>=.2))[0]+1;ix=ix[ix>150];i=int(ix[0]) if len(ix) else None;err=z['mug_fp'][i,:3]-(z['rack'][i,:3]+z['rack_fp'][i,:3])/2 if i else None
  rows.append(dict(candidate=c,seed=d['seed'],success=d['success'],max_mug_rise_m=rise,first_right_release=i,release_handle_error_xyz_m=err.tolist() if err is not None else None,final_mug_xyz=d['final']['mug_pose'][:3],observed_stage='supported hang' if d['success'] else 'no5cm lift achieved' if rise<.05 else 'lift achieved but supported hang not achieved',mechanism='Stage classification from measured poses; not a complete causal explanation'))
assert len(rows)==20;(r/'results/failure_modes.json').write_text(json.dumps(rows,indent=2));print(json.dumps([x for x in rows if x['candidate']==cid and not x['success']],indent=2))
