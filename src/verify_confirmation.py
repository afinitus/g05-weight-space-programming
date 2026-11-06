import os
import json,numpy as np
from pathlib import Path
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve(); cid='right_base_feature168_p0p20';rows=[]
for cfg in json.loads((r/'configs/splits.json').read_text())['dev']:
 s=str(cfg['seed']);a=r/f'results/{cid}/dev_screen/{s}';b=r/f'results/{cid}/dev/{s}'
 if not (b/'result.json').exists():continue
 ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text());base=json.loads((r/f'results/baseline/dev/{s}/result.json').read_text())
 row=dict(seed=int(s),success=rb['success'],initial_images_equal=base['initial_image_hashes']==rb['initial_image_hashes'],initial_state_equal=base['initial']==rb['initial'],repeat_arrays={})
 for f in ['trajectory.npz','physics.npz']:
  x=np.load(a/f);y=np.load(b/f);row['repeat_arrays'][f]=all(np.array_equal(x[k],y[k]) for k in x.files)
 rows.append(row)
print(json.dumps(rows,indent=2));(r/'results/confirmation_audit.json').write_text(json.dumps(rows,indent=2))
assert all(x['success'] and x['initial_images_equal'] and x['initial_state_equal'] and all(x['repeat_arrays'].values()) for x in rows)
print('complete',len(rows)==6)
