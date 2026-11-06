import os
import json,csv,math,numpy as np
from pathlib import Path
from scipy.stats import binomtest
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();freeze=json.loads((r/'configs/final_edit.json').read_text());cid=freeze['id'];seeds=[x['seed'] for x in json.loads((r/'configs/splits.json').read_text())['test']];rows=[]
def release_error(p):
 z=np.load(p/'trajectory.npz');q=z['joints'];idx=np.where((q[:-1,13]<.2)&(q[1:,13]>=.2))[0]+1;idx=idx[idx>150]
 if not len(idx):return None
 i=int(idx[0]);return (z['mug_fp'][i,:3]-(z['rack'][i,:3]+z['rack_fp'][i,:3])/2).tolist()
for seed in seeds:
 bp=r/f'results/baseline/test/{seed}';ep=r/f'results/{cid}/test/{seed}';b=json.loads((bp/'result.json').read_text());e=json.loads((ep/'result.json').read_text());assert b['candidate']=='baseline' and e['candidate']==cid;row=dict(seed=seed,mug_id=b['initial']['mug_id'],baseline=b['success'],edited=e['success'],baseline_native=b['native_success'],edited_native=e['native_success'],baseline_steps=b['steps'],edited_steps=e['steps'],initial_images_equal=b['initial_image_hashes']==e['initial_image_hashes'],initial_physics_equal=all(np.array_equal(b['initial'][k],e['initial'][k]) for k in ['mug_pose','rack_pose','joints']),baseline_speed=b['physics']['max_speed_m_s'],edited_speed=e['physics']['max_speed_m_s'],baseline_penetration=b['physics']['max_penetration_m'],edited_penetration=e['physics']['max_penetration_m'],baseline_release_error=release_error(bp),edited_release_error=release_error(ep));rows.append(row)
assert all(x['initial_images_equal'] and x['initial_physics_equal'] for x in rows)
n=len(rows);sb=sum(x['baseline'] for x in rows);se=sum(x['edited'] for x in rows);g=sum(not x['baseline'] and x['edited'] for x in rows);l=sum(x['baseline'] and not x['edited'] for x in rows)
def wilson(k):
 z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [mid-half,mid+half]
delta=np.array([int(x['edited'])-int(x['baseline']) for x in rows]);rng=np.random.default_rng(271828);boot=rng.choice(delta,(100000,n)).mean(1)
a=dict(candidate=cid,n=n,baseline_successes=sb,edited_successes=se,delta_pp=100*(se-sb)/n,gains=g,losses=l,both_success=sum(x['baseline'] and x['edited'] for x in rows),both_fail=sum(not x['baseline'] and not x['edited'] for x in rows),baseline_wilson95=wilson(sb),edited_wilson95=wilson(se),paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),mcnemar_exact_p=binomtest(g,g+l,.5).pvalue if g+l else 1.0,initial_images_all_identical=True,initial_physics_all_identical=True,baseline_native=sum(x['baseline_native'] for x in rows),edited_native=sum(x['edited_native'] for x in rows))
(r/'results/final_analysis.json').write_text(json.dumps(a,indent=2));(r/'results/paired_episodes.json').write_text(json.dumps(rows,indent=2))
with (r/'results/paired_episodes.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,4));ax.imshow(np.array([[x['baseline'] for x in rows],[x['edited'] for x in rows]]),cmap='RdYlGn',vmin=0,vmax=1,aspect='auto');ax.set_yticks([0,1],['Original','One-scalar edit']);ax.set_xticks(range(n),[str(x['seed']) for x in rows],rotation=45);ax.set_title('Supported hang for1second: every held-out pair');fig.tight_layout();fig.savefig(r/'plots/paired_outcomes.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5));splits={x['seed']:x for x in json.loads((r/'configs/splits.json').read_text())['test']}
for x in rows:
 p=splits[x['seed']]['initial']['mug_pose'];c='#3baf79' if x['edited'] and not x['baseline'] else '#df6767' if x['baseline'] and not x['edited'] else '#406b9c' if x['baseline'] else '#777777';ax.scatter(p[0],p[1],c=c,s=100);ax.annotate(str(x['seed']),p[:2],xytext=(4,4),textcoords='offset points',fontsize=8)
ax.set(xlabel='Initial mug x(m)',ylabel='Initial mug y(m)',title='Green: gain · red: regression · blue: both pass · gray: both fail');fig.tight_layout();fig.savefig(r/'plots/spatial_outcomes.png',dpi=180);plt.close(fig);print(json.dumps(a,indent=2))
