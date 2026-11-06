import os
from pathlib import Path
import json,numpy as np
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();rows=[]
for p in sorted((r/'results').glob('*/dev/*/result.json')):
 if p.parts[-4].startswith('integration_'):continue
 d=json.loads(p.read_text());a=np.load(p.parent/'physics.npz')['samples'];v=np.diff(a[:,2:5],axis=0)/.004;dv=np.linalg.norm(np.diff(v,axis=0),axis=1);rack=(a[:,8]>0)&(a[:,4]>.8);rj=rack[2:]|rack[1:-1];speed=float(a[rack,5].max()) if rack.any() else 0;jump=float(dv[rj].max()) if rj.any() else 0
 rows.append(dict(candidate=d['candidate'],seed=d['seed'],success=d['success'],max_rack_speed_above_0p8m=speed,max_rack_velocity_jump=jump,suspicious_rack_event=bool(speed>3 or jump>3),max_penetration=d['physics']['max_penetration_m'],hold=d['hold']))
(r/'results/development_physics_audit.json').write_text(json.dumps(rows,indent=2))
for cid in sorted(set(x['candidate'] for x in rows)):
 a=[x for x in rows if x['candidate']==cid];print(cid,len(a),'complete',sum(x['success'] for x in a),'success',sum(x['suspicious_rack_event'] for x in a),'suspicious rack events')
