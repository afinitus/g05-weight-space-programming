"""Localhost-only HTTP transport for isolated model and simulation environments.

Arrays use NumPy's non-pickle npz container; structured metadata is JSON.
"""
from __future__ import annotations
import argparse, io,json,sys,time,traceback,urllib.request,urllib.error,os,hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np


def pack(value):
    arrays={}
    def visit(v):
        if isinstance(v,np.ndarray):
            key=f'array_{len(arrays)}'; arrays[key]=v; return {'__ndarray__':key}
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,dict): return {k:visit(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [visit(x) for x in v]
        return v
    meta=json.dumps(visit(value),allow_nan=False)
    arrays['__metadata__']=np.array(meta)
    stream=io.BytesIO(); np.savez(stream,**arrays); return stream.getvalue()


def unpack(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        def visit(v):
            if isinstance(v,dict) and set(v)=={'__ndarray__'}: return z[v['__ndarray__']].copy()
            if isinstance(v,dict): return {k:visit(x) for k,x in v.items()}
            if isinstance(v,list): return [visit(x) for x in v]
            return v
        return visit(json.loads(z['__metadata__'].item()))


class RemotePolicy:
    def __init__(self,url='http://127.0.0.1:18000',timeout=180):
        self.url=url.rstrip('/'); self.timeout=timeout; self.expected_candidate_id=None; self.expected_patch_sha256=None; self.expected_rpc_pid=None
    def call(self,path,payload):
        req=urllib.request.Request(self.url+path,data=pack(payload),headers={'Content-Type':'application/octet-stream'})
        try:
            with urllib.request.urlopen(req,timeout=self.timeout) as r:result=unpack(r.read())
        except urllib.error.HTTPError as e:
            try:result=unpack(e.read())
            except Exception:raise e
            if result.get('error_class')!='invalid_action':raise RuntimeError(result.get('error',str(e))) from e
        if result.get('error_class')=='invalid_action':
            from evaluation import PolicyActionError
            raise PolicyActionError(result['error'])
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def health(self):
        with urllib.request.urlopen(self.url+'/health',timeout=5) as r:return unpack(r.read())
    def wait_ready(self,timeout=600):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:
                with urllib.request.urlopen(self.url+'/health',timeout=2) as r:
                    if unpack(r.read()).get('ready'):return
            except Exception:pass
            time.sleep(2)
        raise TimeoutError('Policy service did not become ready')
    def reset(self,seed=0): return self.call('/reset',{'seed':seed})
    def predict(self,observation,instruction='Pick up the red soda can.'):
        payload={'observation':observation,'instruction':instruction}
        if self.expected_rpc_pid is not None:
            if self.health().get('pid')!=self.expected_rpc_pid:raise RuntimeError('RPC process changed before inference')
            payload['expected_rpc_pid']=self.expected_rpc_pid
        if self.expected_candidate_id is not None:
            payload.update(expected_candidate_id=self.expected_candidate_id,expected_patch_sha256=self.expected_patch_sha256)
        result=self.call('/predict',payload)
        if self.expected_rpc_pid is not None and self.health().get('pid')!=self.expected_rpc_pid:
            raise RuntimeError('RPC process changed during inference')
        identity=result.get('policy_identity')
        if identity is not None and self.expected_candidate_id is not None:
            if identity.get('candidate_id')!=self.expected_candidate_id or identity.get('patch_sha256')!=self.expected_patch_sha256:
                raise RuntimeError('Policy identity changed during episode')
        return result
    def patch(self,path,reverse=False): return self.call('/patch',{'path':str(path),'reverse':reverse})


def serve(args):
    import torch
    torch.set_grad_enabled(False)
    from policy_g05 import G05Policy
    from sparse_patch import apply_patch,atomic_json
    from evaluation import action_steps, PolicyActionError
    policy=G05Policy(checkpoint=args.checkpoint,device=args.device,patch_path=args.patch)
    kernel_manifest_identity=None
    if args.kernel_choices:
        from pathlib import Path
        from triton_kernel_state import restore
        blob=Path(args.kernel_choices).read_bytes()
        count=restore(json.loads(blob),freeze=True)
        kernel_manifest_identity={'path':str(Path(args.kernel_choices).resolve()),
                                  'sha256':hashlib.sha256(blob).hexdigest(),'fixed_kernel_count':count}
        print('FIXED_KERNEL_CHOICES',kernel_manifest_identity,flush=True)
    if args.config_out: atomic_json(args.config_out,policy.config)
    current_patch = None
    if args.patch:
        from pathlib import Path
        current_patch = json.loads((Path(args.patch)/'manifest.json').read_text())
    from pathlib import Path
    source_identity={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in ['policy_g05.py','policy_rpc.py','sparse_patch.py','triton_kernel_state.py']}
    def runtime_identity():
        from triton_kernel_state import snapshot
        kernels=snapshot()
        stable={name:{k:v for k,v in record.items() if k!='best_config'} for name,record in kernels.items()}
        digest=hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return {'pid':os.getpid(),'kernel_choices_sha256':digest,'kernel_count':len(kernels),
                'source_sha256':source_identity,'kernel_manifest':kernel_manifest_identity}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            data=pack({'ready':True,'candidate_id':current_patch['candidate_id'] if current_patch else 'baseline','patch_sha256':current_patch['patch_sha256'] if current_patch else None,'base_checkpoint_sha256':policy.checkpoint_identity['sha256'],'base_checkpoint_path':policy.checkpoint_identity['path'],**runtime_identity()})
            self.send_response(200);self.end_headers();self.wfile.write(data)
        def do_POST(self):
            nonlocal current_patch
            try:
                length=int(self.headers['Content-Length'])
                if length>128*1024**2: raise ValueError('Request too large')
                payload=unpack(self.rfile.read(length))
                if self.path=='/reset': policy.reset(payload['seed']); out={'ok':True}
                elif self.path=='/predict':
                    identity={'candidate_id':current_patch['candidate_id'] if current_patch else 'baseline',
                              'patch_sha256':current_patch['patch_sha256'] if current_patch else None,'pid':os.getpid()}
                    if payload.get('expected_rpc_pid') is not None and payload['expected_rpc_pid']!=os.getpid():
                        raise RuntimeError('Policy process restarted during episode')
                    if 'expected_candidate_id' in payload:
                        if payload['expected_candidate_id']!=identity['candidate_id'] or payload.get('expected_patch_sha256')!=identity['patch_sha256']:
                            raise RuntimeError('Policy identity mismatch; model may have restarted during episode')
                    out=policy.predict(payload['observation'],instruction=payload['instruction'])
                    if not isinstance(out,dict) or 'actions' not in out:
                        raise PolicyActionError('Policy returned no action chunk')
                    action_steps(out['actions'],16)  # Validate only; preserve exact generated values.
                    out['policy_identity']=identity
                elif self.path=='/parameter-info': out=policy.parameter_info(**payload)
                elif self.path=='/causal-probe': out=policy.causal_probe(**payload)
                elif self.path=='/save-patch': out=policy.save_patch(**payload)
                elif self.path=='/runtime-digest':
                    from audit_patch import all_tensors,digest_tensors
                    out={'tensors':digest_tensors(all_tensors(policy)),
                         'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,
                            'grad_enabled':torch.is_grad_enabled(),'threads':torch.get_num_threads(),
                            'matmul_precision':torch.get_float32_matmul_precision(),
                            'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                            'deterministic':torch.are_deterministic_algorithms_enabled(),
                            'env':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']}}}
                elif self.path=='/audit-patch':
                    if current_patch is not None:raise ValueError('Full reversible audit requires baseline model')
                    from audit_patch import audit_live_patch
                    out=audit_live_patch(policy,payload['path'],payload['output'])
                    out['serving_runtime_identity']=runtime_identity()
                    atomic_json(payload['output'],out)
                elif self.path=='/kernel-cache':
                    from triton_kernel_state import snapshot
                    out={'kernels':snapshot()}
                elif self.path=='/patch':
                    from pathlib import Path
                    requested=json.loads((Path(payload['path'])/'manifest.json').read_text())
                    reverse=payload.get('reverse',False)
                    if reverse:
                        if current_patch is None or requested['patch_sha256']!=current_patch['patch_sha256']:raise ValueError('Cannot reverse inactive patch')
                    elif current_patch is not None:raise ValueError('Patch already active; reverse before applying another')
                    out=apply_patch(policy.model,payload['path'],reverse=reverse)
                    current_patch=None if reverse else out
                else: raise ValueError('Unknown operation')
                data=pack(out);self.send_response(200)
            except Exception as e:
                traceback.print_exc();data=pack({'error':repr(e),'error_class':'invalid_action' if isinstance(e,PolicyActionError) else 'execution_error'});self.send_response(500)
            self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
    server=HTTPServer(('127.0.0.1',args.port),Handler)
    print(f'POLICY_READY 127.0.0.1:{args.port}',flush=True);server.serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18000);p.add_argument('--device',default='cuda')
    p.add_argument('--checkpoint');p.add_argument('--patch');p.add_argument('--config-out');p.add_argument('--kernel-choices');serve(p.parse_args())
