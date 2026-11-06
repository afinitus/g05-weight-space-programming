import os
import json,argparse,numpy as np
from pathlib import Path
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();ap=argparse.ArgumentParser();ap.add_argument('candidate',nargs='?',default='baseline');ap.add_argument('--split',default='dev');a=ap.parse_args();rows=[]
for p in sorted((r/'results'/a.candidate/a.split).glob('*/result.json')):
 d=json.loads(p.read_text());z=np.load(p.parent/'trajectory.npz');m=z['mug'][:,:3];q=z['joints'];ph=np.load(p.parent/'physics.npz')['samples'];rack=ph[:,8]>0;robot=ph[:,9]>0
 row=dict(seed=d['seed'],mug_id=d['initial']['mug_id'],success=d['success'],native=d['native_success'],steps=d['steps'],max_lift=float(m[:,2].max()-m[0,2]),max_x=float(m[:,0].max()),final_pose=d['final']['mug_pose'][:3],last_left_gripper=float(q[-1,6]),last_right_gripper=float(q[-1,13]),rack_contact_steps=int(rack.sum()),robot_contact_steps=int(robot.sum()),**d['physics']);rows.append(row)
print(json.dumps(rows,indent=2));print('TOTAL',sum(x['success'] for x in rows),'/',len(rows))
