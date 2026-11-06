"""Render a fixed-simulation-time close view; verify all original action/physics states."""
import argparse,json,numpy as np,transforms3d as t3d,imageio.v2 as imageio
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from run import setup,snapshot,hold,ROOT,sapien
ap=argparse.ArgumentParser();ap.add_argument('episode');ap.add_argument('--start',type=float,default=55);ap.add_argument('--end',type=float,default=125);a=ap.parse_args();ep=Path(a.episode);result=json.loads((ep/'result.json').read_text());z=np.load(ep/'trajectory.npz');sourcephysics=np.load(ep/'physics.npz')['samples'];e,real,tr=setup(result['seed']);native=e.check_success;e.check_success=lambda:False
camera=real.add_camera('presentation_close',1280,720,np.deg2rad(46),.03,12);eye=np.array([.75,.85,1.35]);target=np.array([.03,.02,.9]);f=target-eye;f/=np.linalg.norm(f);left=np.cross([0,0,1],f);left/=np.linalg.norm(left);up=np.cross(f,left);camera.set_pose(sapien.Pose(eye,t3d.quaternions.mat2quat(np.stack([f,left,up],axis=1))))
out=ROOT/'videos'/f'{result["candidate"]}_{result["seed"]}_presentation.mp4';writer=imageio.get_writer(str(out),fps=15,codec='libx264',quality=8);next_frame=a.start;frames=0;maxerr=0.;origstep=tr.step;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
def step():
 global next_frame,frames
 origstep()
 if tr.t+1e-8>=next_frame and tr.t<=a.end:
  real.update_render();camera.take_picture();im=Image.fromarray(np.clip(camera.get_picture('Color')[...,:3]*255,0,255).astype('uint8'));d=ImageDraw.Draw(im);d.rectangle([0,0,1280,35],fill='#101827');d.text((16,6),f'Recorded-action replay | seed {result["seed"]} | simulation {tr.t:.2f}s | action {tr.action}',font=font,fill='white');writer.append_data(np.asarray(im));frames+=1;next_frame+=1/15
tr.step=step;attempts={h['action']:h for h in result['hold_attempts']}
try:
 for i,action in enumerate(z['actions']):
  state=snapshot(e)
  for key,field in [('mug','mug_pose'),('rack','rack_pose'),('joints','joints')]:maxerr=max(maxerr,float(np.max(abs(np.asarray(state[field])-z[key][i]))))
  if maxerr>1e-5:raise RuntimeError(f'Replay divergence action{i}: {maxerr}')
  tr.action=i;e.get_obs();e.take_action(action,action_type='qpos')
  if i+1 in attempts:
   e.check_success=native;h=hold(e,tr);e.check_success=lambda:False
   assert h['validated_success']==attempts[i+1]['validated_success']
  if i%100==0:print('REPLAY',i,flush=True)
 state=snapshot(e)
 for key in ['mug_pose','rack_pose','joints']:maxerr=max(maxerr,float(np.max(abs(np.asarray(state[key])-np.asarray(result['final'][key])))))
 replay=np.asarray(tr.rows);assert replay.shape==sourcephysics.shape
 physics_error=float(np.max(abs(replay-sourcephysics)));assert maxerr<=1e-5 and physics_error<=1e-5,(maxerr,physics_error)
 audit=dict(source=str(ep),video=str(out),verified=True,max_state_error=maxerr,max_physics_sample_error=physics_error,frames=frames,clip_start_simtime=a.start,clip_end_simtime=min(a.end,tr.t),source_success=result['success'],policy_episode=False,playback='15frames per simulated second; actual recorded policy actions replayed; no new policy inference')
 out.with_suffix('.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
finally:writer.close();e.scene=real;e.close_env()
