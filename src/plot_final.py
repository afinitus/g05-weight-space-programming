import os
import json,numpy as np
from pathlib import Path
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
r=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();rows=json.loads((r/'results/paired_episodes.json').read_text());cid=json.loads((r/'configs/final_edit.json').read_text())['id'];fig,axes=plt.subplots(2,5,figsize=(16,7))
for ax,row in zip(axes.flat,rows):
 for label,c,color in [('Original','baseline','#df6767'),('Edited',cid,'#319e75')]:
  p=r/f'results/{c}/test/{row["seed"]}';z=np.load(p/'trajectory.npz');q=z['joints'];ix=np.where((q[:-1,13]<.2)&(q[1:,13]>=.2))[0]+1;ix=ix[ix>150];end=int(ix[0])+5 if len(ix) else min(len(q),350);start=max(150,end-80)
  target=(z['rack'][:,:3]+z['rack_fp'][:,:3])/2;rel=(z['mug_fp'][:,:3]-target)*100;ax.plot(rel[start:end,0],rel[start:end,1],color=color,label=label,alpha=.85)
  if len(ix):ax.scatter(rel[ix[0],0],rel[ix[0],1],color=color,s=25)
 ax.add_patch(plt.Rectangle((-2,-2),4,4,fill=False,color='#666',linestyle='--'));ax.set_title(f'{row["seed"]}: {int(row["baseline"])} → {int(row["edited"])}');ax.set(xlabel='Handle x error(cm)',ylabel='Handle y error(cm)',xlim=(-12,12),ylim=(-12,12));ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8);fig.suptitle('Approach before first right-gripper release; dots mark release, box marks native xy tolerance');fig.tight_layout();fig.savefig(r/'plots/approach_trajectories.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(10)
for key,c,label,offset in [('baseline','#df6767','Original',-.17),('edited','#319e75','Edited',.17)]:
 vals=[np.linalg.norm(row[key+'_release_error'][:2])*100 if row[key+'_release_error'] is not None else np.nan for row in rows];ax.bar(x+offset,vals,width=.34,color=c,label=label)
ax.set_xticks(x,[str(row['seed']) for row in rows],rotation=45);ax.set_ylabel('Planar handle error at first release(cm)');ax.legend();fig.tight_layout();fig.savefig(r/'plots/release_errors.png',dpi=180)
f=json.loads((r/'configs/final_edit.json').read_text());fig,ax=plt.subplots(figsize=(7,3));ax.bar(['Original','Edited'],[f['values'][0]['old'],f['values'][0]['new']],color=['#df6767','#319e75']);ax.set_title('Only changed coefficient: output_proj.weight[10,168]');ax.set_ylabel('FP32 weight value');fig.tight_layout();fig.savefig(r/'plots/weight_delta.png',dpi=180)
