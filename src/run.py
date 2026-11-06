import os
import os,sys,json,time,hashlib,argparse,random,traceback
from pathlib import Path
ROOT=Path(os.environ.get('G05_EXPERIMENT_ROOT', Path(__file__).resolve().parents[1])).resolve();SIM=Path(os.environ['ROBOTWIN_ROOT']).resolve()
os.chdir(SIM);sys.path[:0]=[str(SIM),str(SIM/'description/utils')]
import numpy as np,yaml,torch,imageio.v2 as imageio,transforms3d as t3d
import sapien.core as sapien
from functools import partial
sapien.SapienRenderer=partial(sapien.SapienRenderer,device=os.environ.get('ROBOTWIN_RENDER_DEVICE','cuda:1'))
from envs.hanging_mug import hanging_mug
from policy_rpc import RemotePolicy,pack
INSTRUCTION='Use left arm to pick the mug on the table, rotate the mug and put the mug down in the middle of the table, use the right arm to pick the mug and hang it onto the rack.'
def config():
 c=yaml.safe_load((SIM/'task_config/demo_clean.yml').read_text());robot=str(SIM/'assets/embodiments/aloha-agilex');rc=yaml.safe_load(Path(robot,'config.yml').read_text());c.update(task_name='hanging_mug',task_config='demo_clean',left_robot_file=robot,right_robot_file=robot,left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,eval_mode=True,render_freq=0,save_data=False);return c
def snapshot(e):
 return dict(mug_pose=np.r_[e.mug.get_pose().p,e.mug.get_pose().q].tolist(),rack_pose=np.r_[e.rack.get_pose().p,e.rack.get_pose().q].tolist(),joints=e.robot.get_left_arm_jointState()+e.robot.get_right_arm_jointState(),mug_id=int(e.mug_id),mug_fp=np.asarray(e.mug.get_functional_point(0)).tolist(),rack_fp=np.asarray(e.rack.get_functional_point(0)).tolist(),left_tcp=e.robot.get_left_ee_pose(),right_tcp=e.robot.get_right_ee_pose())
def append(path,obj):
 with open(path,'a') as f:f.write(json.dumps(obj)+'\n');f.flush();os.fsync(f.fileno())
class Trace:
 def __init__(self,scene,env,ep=None):
  self.scene=scene;self.env=env;self.action=-1;self.t=0.;self.dt=scene.get_timestep();self.rows=[];self.prev=env.mug.get_pose().p.copy();self.writer=None;self.next_frame=0.
  if ep:
   self.writer=imageio.get_writer(str(ep/'rollout.mp4'),fps=15,codec='libx264',quality=7)
   self.camera=scene.add_camera('audit_overview',640,480,np.deg2rad(55),.03,12)
   eye=np.array([.85,-.95,1.65]);target=np.array([0,0,.9]);f=target-eye;f/=np.linalg.norm(f);left=np.cross([0,0,1],f);left/=np.linalg.norm(left);up=np.cross(f,left);self.camera.set_pose(sapien.Pose(eye,t3d.quaternions.mat2quat(np.stack([f,left,up],axis=1))))
 def __getattr__(self,n):return getattr(self.scene,n)
 def step(self):
  self.scene.step();self.t+=self.dt;p=self.env.mug.get_pose().p.copy();speed=float(np.linalg.norm(p-self.prev)/self.dt);self.prev=p;contacts=set();penetration=0.;impulse=0.
  for c in self.scene.get_contacts():
   names=[b.entity.name for b in c.bodies]
   if '039_mug' not in names:continue
   contacts.update(n for n in names if n!='039_mug')
   for pt in c.points:
    penetration=max(penetration,-float(pt.separation));impulse=max(impulse,float(np.linalg.norm(pt.impulse)))
  robot=any(n.startswith(('fl_','fr_')) for n in contacts)
  self.rows.append([self.t,self.action,*p,speed,penetration,impulse,int('040_rack' in contacts),int(robot),int('table' in contacts),int('ground' in contacts)])
  if self.writer and self.t+1e-8>=self.next_frame:
   from PIL import Image,ImageDraw
   self.scene.update_render();self.camera.take_picture();rgb=np.clip(self.camera.get_picture('Color')[...,:3]*255,0,255).astype('uint8');im=Image.fromarray(rgb);d=ImageDraw.Draw(im);d.rectangle([0,0,640,25],fill='black');d.text((9,7),f'Simulation time {self.t:.2f}s | action {self.action} | 15 fps fixed simulation time',fill='white');self.writer.append_data(np.asarray(im));self.next_frame+=1/15
 def summary(self):
  a=np.asarray(self.rows);return dict(sim_seconds=self.t,max_speed_m_s=float(a[:,5].max()),max_penetration_m=float(a[:,6].max()),max_contact_impulse=float(a[:,7].max()),speed_over_3m_s_steps=int((a[:,5]>3).sum()),penetration_over_1cm_steps=int((a[:,6]>.01).sum()),ground_contact_steps=int(a[:,11].sum()))
 def close(self,ep=None):
  if self.writer:self.writer.close()
  if ep:np.savez_compressed(ep/'physics.npz',samples=np.asarray(self.rows),columns=np.array(['time','action','x','y','z','speed','penetration','impulse','rack_contact','robot_contact','table_contact','ground_contact']))
def hold(e,trace):
 start=len(trace.rows);native=[]
 for i in range(250):trace.step();native.append(bool(e.check_success()))
 a=np.asarray(trace.rows[start:]);ratio=float(a[:,8].mean());robot=bool(a[:,9].any());ok=bool(all(native) and ratio>=.95 and not robot)
 return dict(validated_success=ok,native_pose_all_steps=all(native),rack_contact_fraction=ratio,robot_contact_any=robot,duration=250*trace.dt)
def setup(seed,ep=None):
 random.seed(seed);e=hanging_mug();e.setup_demo(now_ep_num=0,seed=seed,is_test=True,**config());real=e.scene;tr=Trace(real,e,ep);e.scene=tr;return e,real,tr
def freeze():
 path=ROOT/'configs/splits.json'
 if path.exists():print('ALREADY FROZEN');return
 accepted=[]
 for seed in range(200000,200200):
  e,real,tr=setup(seed)
  try:
   initial=snapshot(e);e.play_once();native=bool(e.plan_success and e.check_success());h=hold(e,tr) if native else dict(validated_success=False);record=dict(seed=seed,native_success=native,hold=h,physics=tr.summary(),initial=initial);append(ROOT/'results/expert_screen.jsonl',record)
   if h['validated_success']:accepted.append(dict(seed=seed,initial=initial))
   print('SCREEN',seed,native,h['validated_success'],len(accepted),flush=True)
  finally:e.scene=real;tr.close();e.close_env()
  if len(accepted)==16:break
 if len(accepted)!=16:raise RuntimeError('Not enough valid expert configurations')
 path.write_text(json.dumps(dict(dev=accepted[:6],test=accepted[6:]),indent=2));print('FROZEN',hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
def evaluate(args):
 import fcntl
 server_lock=open(ROOT/f'logs/server_{args.port}.lock','a')
 fcntl.flock(server_lock,fcntl.LOCK_EX)
 data=json.loads((ROOT/'configs/splits.json').read_text())[args.split];data=[data[args.index]] if args.index is not None else data[args.worker::args.workers]
 rpc=RemotePolicy('http://127.0.0.1:'+str(args.port));rpc.wait_ready();patch=json.loads(Path(args.patch).read_text()) if args.patch else None;identity=rpc.call('/patch',dict(patch=patch));policy_cid=identity['candidate'];control=json.loads(Path(args.control).read_text()) if args.control else None;cid=control['id'] if control else policy_cid;out=ROOT/'results'/cid/args.split;out.mkdir(parents=True,exist_ok=True);(out/f'patch_values_w{args.worker}.json').write_text(json.dumps(identity,indent=2))
 for item in data:
  seed=item['seed'];ep=out/str(seed)
  if (ep/'result.json').exists():continue
  ep.mkdir(parents=True,exist_ok=True);e,real,tr=setup(seed,ep);initial=snapshot(e)
  try:
   for k in ['mug_pose','rack_pose','joints']:assert np.allclose(initial[k],item['initial'][k],atol=1e-6),k
   e.set_instruction(INSTRUCTION);rpc.reset(seed);queue=[];records=[];actions=[];times=[];t=time.monotonic();hashes={}
   native_check=e.check_success;native_seen=[False];native_first_action=[None]
   def observe_native():
    if native_check():
     if not native_seen[0]:native_first_action[0]=e.take_action_cnt
     native_seen[0]=True
    return False
   e.check_success=observe_native
   h=dict(validated_success=False);hold_attempts=[]
   while e.take_action_cnt<e.step_lim and not h['validated_success']:
    obs=e.get_obs()
    if not actions:hashes={k:hashlib.sha256(obs['observation'][k]['rgb'].tobytes()).hexdigest() for k in ['head_camera','left_camera','right_camera']}
    if not queue:
     payload=dict(observation={k:dict(rgb=obs['observation'][k]['rgb']) for k in ['head_camera','left_camera','right_camera']},joint_action=dict(vector=np.asarray(obs['joint_action']['vector'])))
     if args.split=='dev' and len(actions)%40==0:(ep/f'obs_{len(actions):04d}.npz').write_bytes(pack(dict(observation=payload,instruction=INSTRUCTION,seed=seed)))
     result=rpc.predict(payload,INSTRUCTION);assert result['candidate']==policy_cid;chunk=result['actions'].copy()
     if control:chunk[:,control['action_index']]+=control['delta']
     assert np.isfinite(chunk).all();queue=list(chunk[:8])
    action=queue.pop(0);tr.action=len(actions);actions.append(action);records.append(snapshot(e));times.append(tr.t);e.take_action(action,action_type='qpos')
    # Finish the native action instead of freezing midway through gripper opening.
    # Success checking never enters policy observations or replaces policy actions.
    if native_check() and tr.rows[-1][8] and not tr.rows[-1][9]:
     e.check_success=native_check
     h=hold(e,tr);hold_attempts.append(dict(action=e.take_action_cnt,**h))
     e.check_success=observe_native
   e.check_success=native_check
   native=bool(native_seen[0]);result=dict(seed=seed,candidate=cid,native_success=native,success=h['validated_success'],hold=h,hold_attempts=hold_attempts,native_first_action=native_first_action[0],steps=e.take_action_cnt,seconds=time.monotonic()-t,initial=initial,final=snapshot(e),initial_image_hashes=hashes,physics=tr.summary(),instruction=INSTRUCTION,action_control=control)
   np.savez_compressed(ep/'trajectory.npz',actions=np.asarray(actions),time=times,mug=np.asarray([x['mug_pose'] for x in records]),rack=np.asarray([x['rack_pose'] for x in records]),joints=np.asarray([x['joints'] for x in records]),left_tcp=np.asarray([x['left_tcp'] for x in records]),right_tcp=np.asarray([x['right_tcp'] for x in records]),mug_fp=np.asarray([x['mug_fp'] for x in records]),rack_fp=np.asarray([x['rack_fp'] for x in records]))
   tr.close(ep);(ep/'result.json').write_text(json.dumps(result,indent=2));append(out/'episodes.jsonl',result);print('EPISODE',seed,cid,'native',native,'validated',h['validated_success'],result['physics'],flush=True)
  finally:e.scene=real;tr.close();e.close_env()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','eval']);ap.add_argument('--split',default='dev');ap.add_argument('--index',type=int);ap.add_argument('--port',type=int,default=18300);ap.add_argument('--patch');ap.add_argument('--control');ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1);args=ap.parse_args();freeze() if args.mode=='freeze' else evaluate(args)
