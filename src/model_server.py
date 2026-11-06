import os
import os, sys, json, traceback, argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
ROOT=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve()
G=Path(os.environ['GALAXEA_VLA_ROOT']).resolve()
sys.path[:0]=[str(G),str(G/'src')]
from policy_rpc import pack,unpack
import torch,numpy as np
torch.set_grad_enabled(False)
torch.set_num_threads(8)
from experiments.robotwin.galaxeafm_policy.deploy_policy import get_model
ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=18300);args=ap.parse_args()
os.chdir(G)
model=get_model(dict(ckpt_setting=str(ROOT/'checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt'),dataset_stats_path=str(ROOT/'checkpoints/g05-robotwin20/dataset_stats.json'),sim_task='robotwin',device='cuda:0',mixed_precision='bf16',replan_steps=8,num_inference_steps=10,seed=20261106))
params=dict(model.policy.named_parameters())
inventory=[dict(name=n,shape=list(p.shape),numel=p.numel(),dtype=str(p.dtype)) for n,p in params.items()]
(ROOT/'results/parameter_inventory.json').write_text(json.dumps(inventory,indent=2))
current='baseline';saved={}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(pack(dict(ready=True,candidate=current,pid=os.getpid())))
 def do_POST(self):
  global current,saved
  try:
   req=unpack(self.rfile.read(int(self.headers['Content-Length'])))
   if self.path=='/reset':
    model.reset();torch.manual_seed(req['seed']);np.random.seed(req['seed']);out=dict(candidate=current)
   elif self.path=='/predict':
    out=dict(actions=model._infer_action_chunk(req['observation'],req['instruction']),candidate=current)
   elif self.path=='/patch':
    for (name,idx),v in saved.items():params[name].view(-1)[idx]=v
    saved={};current='baseline'
    patch=req.get('patch')
    if patch:
     for edit in patch['edits']:
      n,i=edit['name'],int(edit['index']);p=params[n].view(-1);saved[n,i]=p[i].clone();p[i]+=float(edit['delta'])
     current=patch['id']
    out=dict(candidate=current,values=[dict(name=n,index=i,old=float(v),new=float(params[n].view(-1)[i])) for (n,i),v in saved.items()])
   elif self.path=='/inspect':
    names=req.get('names',[]);out={n:params[n].float().cpu().numpy() for n in names}
   else:raise ValueError(self.path)
   payload=pack(out);self.send_response(200)
  except Exception:
   traceback.print_exc();payload=pack(dict(error=traceback.format_exc()));self.send_response(500)
  self.end_headers();self.wfile.write(payload)
print('READY',args.port,flush=True)
HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
