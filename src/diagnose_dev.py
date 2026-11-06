import os
from pathlib import Path
import json,numpy as np
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();rows=[]
for p in sorted((r/'results/baseline/dev').glob('*/result.json')):
 d=json.loads(p.read_text());z=np.load(p.parent/'trajectory.npz');q=z['joints'];a=np.load(p.parent/'physics.npz')['samples'];m=z['mug'][:,:3];err=z['mug_fp'][:,:3]-(z['rack'][:,:3]+z['rack_fp'][:,:3])/2;opens=np.where((q[:-1,13]<.2)&(q[1:,13]>=.2))[0]+1;opens=opens[opens>150];i=int(opens[0]) if len(opens) else None;peak=int(a[:,5].argmax());step=a[peak];velocity=(a[peak,2:5]-a[max(0,peak-1),2:5])/.004
 row=dict(seed=d['seed'],success=d['success'],native=d['native_success'],release_action=i,release_handle_error=err[i].tolist() if i is not None else None,peak_speed=d['physics']['max_speed_m_s'],peak_speed_time=float(step[0]),peak_speed_action=int(step[1]),peak_speed_position=step[2:5].tolist(),peak_velocity_xyz=velocity.tolist(),peak_contacts=dict(rack=bool(step[8]),robot=bool(step[9]),table=bool(step[10]),ground=bool(step[11])),penetration=d['physics']['max_penetration_m'],final_mug_xyz=d['final']['mug_pose'][:3]);rows.append(row)
(r/'results/development_diagnosis.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
