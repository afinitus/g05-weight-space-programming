import os
import json,numpy as np
from pathlib import Path
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();cid=json.loads((r/'configs/final_edit.json').read_text())['id'];rows=[]
for c in ['baseline',cid]:
 for p in sorted((r/f'results/{c}/test').glob('*/result.json')):
  d=json.loads(p.read_text());a=np.load(p.parent/'physics.npz')['samples'];assert np.isfinite(a).all(),p;v=np.diff(a[:,2:5],axis=0)/.004;dv=np.linalg.norm(np.diff(v,axis=0),axis=1);rack=(a[:,8]>0)&(a[:,4]>.8);rj=rack[2:]|rack[1:-1]
  speed=float(a[rack,5].max()) if rack.any() else 0;jump=float(dv[rj].max()) if rj.any() else 0
  rows.append(dict(candidate=c,seed=d['seed'],success=d['success'],max_rack_speed_m_s=speed,max_rack_velocity_jump_m_s=jump,suspicious_rack_event=bool(speed>3 or jump>3),physics=d['physics'],hold=d['hold']))
assert len(rows)==20
(r/'results/final_physics_audit.json').write_text(json.dumps(rows,indent=2))
for c in ['baseline',cid]:
 a=[x for x in rows if x['candidate']==c];print(c,'rack flags',sum(x['suspicious_rack_event'] for x in a),'maxrack',max(x['max_rack_speed_m_s'] for x in a),'maxjump',max(x['max_rack_velocity_jump_m_s'] for x in a),'maxpenetration',max(x['physics']['max_penetration_m'] for x in a))
