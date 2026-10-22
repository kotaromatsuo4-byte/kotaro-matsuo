"""Source-only supervised actor initialization, then target-free first discovery.

The mechanical environment and Monte-Carlo ActorCriticAgent are unchanged.
Target runs read only node_initial/member/fix, D, and a source actor checkpoint.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import contextlib
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import mechanics
from main import task_config
from env import OrigamiEnv
from policy import VariableModeActorCritic, bounded_variance
from agent import ActorCriticAgent, Transition

BASE = Path(__file__).resolve().parent
SOURCE = BASE/'result/experiments/generalization_5h_20260914'
ROOT = BASE/'result/experiments/discovery_transfer_5h_20260914'
DEADLINE = 1789391107.0  # 2026-09-14 13:05:07 UTC, five hours after start.

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    temp.replace(path)

def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

class GoalEnv(OrigamiEnv):
    def get_state(self):
        return np.append(super().get_state(),self.target_spread).astype(np.float32)

class GraphActorCritic(nn.Module):
    """Variable-node actor; fresh, separate critic at each target structure.

    Gaussian latent actions, hemisphere convention and signed tanh width are
    handled by the unmodified ActorCriticAgent, exactly as for the dense actor.
    """
    def __init__(self, coordinate_size, hidden=32, variance_bias=-1.0986122886681098):
        super().__init__();self.hidden_size=hidden;self.coordinate_size=coordinate_size
        self.node_encoder=nn.Sequential(nn.Linear(13,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh()) for _ in range(2)])
        self.mixture_mean=nn.Linear(hidden,1);self.mixture_variance=nn.Linear(hidden,1)
        self.width_body=nn.Sequential(nn.Linear(hidden+8,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.width_mean=nn.Linear(hidden,1);self.width_variance=nn.Linear(hidden,1)
        self.critic_body=nn.Sequential(nn.Linear(coordinate_size+1,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,1))
        for module in (self.mixture_variance,self.width_variance):
            nn.init.zeros_(module.weight);nn.init.constant_(module.bias,variance_bias)
        self.register_buffer('adjacency',torch.zeros(1,1),persistent=False)
        self.register_buffer('constants',torch.zeros(1,5),persistent=False)
        self.dimension=1

    def set_geometry(self, directory, dimension=1):
        directory=Path(directory)
        node=np.loadtxt(directory/'node_initial.dat',delimiter=',')
        member=np.loadtxt(directory/'member.dat',delimiter=',',dtype=int).reshape(-1,2)
        fix=np.loadtxt(directory/'fix.dat',delimiter=',',dtype=int).reshape(-1,2)
        lengths=np.linalg.norm(node[member[:,0]]-node[member[:,1]],axis=1)
        adjacency=np.zeros((len(node),len(node)),np.float32);total=np.zeros(len(node))
        for (a,b),length in zip(member,lengths):
            adjacency[a,b]=adjacency[b,a]=1;total[a]+=length/lengths.max();total[b]+=length/lengths.max()
        degree=adjacency.sum(1);fixed=np.zeros((len(node),3));fixed[fix[:,0],fix[:,1]]=1
        self.adjacency=torch.from_numpy(adjacency/np.maximum(degree[:,None],1))
        self.constants=torch.tensor(np.column_stack([fixed,degree/6,total/np.maximum(degree,1)]),dtype=torch.float32)
        self.dimension=dimension

    def batch_modal(self,states,modes):
        batch,k,c=modes.shape;n=c//3
        mask=modes.square().sum(-1)>1e-10
        positions=states[:,:-1].reshape(batch,n,3)
        positions=positions-positions.mean(1,keepdim=True)
        goal=states[:,-1:]
        feat=torch.cat([positions[:,None].expand(-1,k,-1,-1),modes.reshape(batch,k,n,3),
            self.constants[None,None].expand(batch,k,-1,-1),goal[:,None,None].expand(-1,k,n,-1),
            states.new_full((batch,k,n,1),self.dimension)],-1)
        h=self.node_encoder(feat)
        for layer in self.messages:
            neighbor=torch.einsum('ij,bkjh->bkih',self.adjacency,h)
            h=h+layer(torch.cat([h,neighbor],-1))
        features=h.mean(2)
        mean=torch.tanh(self.mixture_mean(features).squeeze(-1))*mask
        return mean,features,mask

    def batch_width(self,states,direction,features,mask):
        n=direction.shape[-1]//3
        positions=states[:,:-1].reshape(-1,n,3)
        positions=positions-positions.mean(1,keepdim=True)
        directions=direction.reshape(-1,n,3)
        # Signed orientation information is needed for the canonical hemisphere.
        descriptors=torch.cat([directions.mean(1),(directions*positions).mean(1)],-1)
        pooled=(features*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
        goal=states[:,-1:]
        h=self.width_body(torch.cat([pooled,descriptors,goal,goal.new_full(goal.shape,self.dimension)],-1))
        return 3*torch.tanh(self.width_mean(h).squeeze(-1)/3),h

    def modal_distribution_parameters(self,state,modes):
        mean,features,_=self.batch_modal(state,modes[None])
        return mean[0],bounded_variance(self.mixture_variance(features[0]).squeeze(-1)),features[0]

    def width_distribution_parameters(self,state,direction,features):
        mask=torch.ones((1,len(features)),dtype=torch.bool)
        mean,h=self.batch_width(state,direction[None],features[None],mask)
        return mean[0],bounded_variance(self.width_variance(h).squeeze())

    def value(self,state,modes): return self.critic_body(state).squeeze()
    def actor_parameters(self): return [p for n,p in self.named_parameters() if not n.startswith('critic_')]
    def critic_parameters(self): return list(self.critic_body.parameters())

def make_model(architecture,coordinate_size,seed):
    torch.manual_seed(seed)
    if architecture=='graph': return GraphActorCritic(coordinate_size)
    return VariableModeActorCritic(coordinate_size,hidden_size=96,goal_size=1,variance_bias=-1.0986122886681098)

def batch_modal(model,states,modes):
    if isinstance(model,GraphActorCritic): return model.batch_modal(states,modes)
    mask=modes.square().sum(-1)>1e-10
    features=model.mode_encoder(torch.cat([states[:,None].expand(-1,modes.shape[1],-1),modes],-1))
    return torch.tanh(model.mixture_mean(features).squeeze(-1))*mask,features,mask

def batch_width(model,states,direction,features,mask):
    if isinstance(model,GraphActorCritic): return model.batch_width(states,direction,features,mask)[0]
    pooled=(features*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
    h=model.width_body(torch.cat([states,direction,pooled],-1))
    return model.width_mean_limit*torch.tanh(model.width_mean(h).squeeze(-1)/model.width_mean_limit)

def canonical_batch(direction):
    pivot=direction.abs().argmax(-1,keepdim=True)
    sign=torch.where(direction.gather(-1,pivot)<0,-1.,1.)
    return direction*sign,sign.squeeze(-1)

def source_groups(architecture,data_root=None):
    # Explicit allowlist: neither D4 support nor torus teacher data is read.
    names=[('seven_bar_linkage','source_data.npz')]
    if architecture=='graph': names.insert(0,('five_bar_linkage','five_source_data.npz'))
    groups=[]
    for structure,name in names:
        file=Path(data_root or SOURCE)/name;data=dict(np.load(file))
        if set(np.unique(data['goal']))!={2.,3.}: raise ValueError('Source goals must be D2/D3 only')
        nodes=data['node'];directory=BASE/'model'/structure
        initial=np.loadtxt(directory/'node_initial.dat',delimiter=',')
        members=np.loadtxt(directory/'member.dat',delimiter=',',dtype=int)
        scale=np.linalg.norm(initial[members[:,0]]-initial[members[:,1]],axis=1).max()
        fix=np.loadtxt(directory/'fix.dat',delimiter=',',dtype=int).reshape(-1,2)
        free=mechanics.index_free(len(initial),fix)
        # Existing teacher records used sign-canonicalized SVD bases; legacy RL
        # uses raw SVD bases. Recompute observations without touching targets.
        raw_modes=np.zeros_like(data['modes'])
        for i,node in enumerate(nodes):
            modes=mechanics.mode(free,mechanics.dedx(node,members,free),len(node))
            raw_modes[i,:modes.shape[1]]=modes.T
        data['modes']=raw_modes
        # Hold entire initial-state/path groups out, never adjacent states.
        cutoff=18 if structure.startswith('five') else 24
        t={k:torch.tensor(data[k],dtype=torch.float32) for k in ('modes','direction','width','width_valid')}
        t['states']=torch.tensor(np.column_stack([nodes.reshape(len(nodes),-1)/scale,data['goal']]),dtype=torch.float32)
        t['train']=torch.tensor(np.flatnonzero(data['path_id']<cutoff));t['val']=torch.tensor(np.flatnonzero(data['path_id']>=cutoff))
        groups.append((directory,t,dict(path=str(file),sha256=digest(file),labels=len(nodes),train=len(t['train']),validation=len(t['val']))))
    return groups

def supervised_loss(model,t,idx):
    states=t['states'][idx];modes=t['modes'][idx]
    counts=(modes.square().sum(-1)>1e-10).sum(-1);modes=modes[:,:max(1,int(counts.max()))]
    mean,features,mask=batch_modal(model,states,modes)
    direction=F.normalize(torch.bmm(mean[:,None],modes).squeeze(1),dim=-1,eps=1e-8)
    target=t['direction'][idx]
    canonical,sign=canonical_batch(target)
    raw_width=batch_width(model,states,canonical,features,mask)
    valid=t['width_valid'][idx]
    dl=(direction-target).square().sum(-1).mean()
    wl=((torch.tanh(raw_width)-sign*t['width'][idx]).square()*valid).sum()/valid.sum().clamp(min=1)
    # Fix modal magnitude, otherwise supervised unit-direction loss leaves
    # Gaussian signal/noise ratio unconstrained. This is source-only calibration.
    normloss=(mean.norm(dim=-1)-1).square().mean()
    return dl+.5*wl+.05*normloss,dl,wl

def pretrain(architecture,seed,updates,label=None,data_root=None):
    output=ROOT/'source'/f'{label or architecture}_seed{seed}';output.mkdir(parents=True,exist_ok=False)
    model=make_model(architecture,21,seed);groups=source_groups(architecture,data_root)
    # Variance and critic are not teacher-supervised and remain at initialization.
    parameters=[p for n,p in model.named_parameters() if not n.startswith('critic') and 'variance' not in n]
    optimizer=torch.optim.Adam(parameters,lr=.001)
    rng=torch.Generator().manual_seed(seed+100);history=[];best=float('inf');start=time.monotonic()
    for update in range(1,updates+1):
        if time.time()>DEADLINE-120: break
        directory,t,_=groups[int(torch.randint(len(groups),(1,),generator=rng))]
        if architecture=='graph':model.set_geometry(directory)
        idx=t['train'][torch.randint(len(t['train']),(32,),generator=rng)]
        loss,dl,wl=supervised_loss(model,t,idx)
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(parameters,1);optimizer.step()
        if update%250==0 or update==updates:
            values=[]
            with torch.no_grad():
                for directory,vt,_ in groups:
                    if architecture=='graph':model.set_geometry(directory)
                    for idx in vt['val'].split(64):values.append(float(supervised_loss(model,vt,idx)[0]))
            val=float(np.mean(values));row=dict(update=update,train_loss=float(loss.detach()),direction_loss=float(dl.detach()),width_loss=float(wl.detach()),validation_loss=val,seconds=time.monotonic()-start)
            history.append(row);write_csv(output/'learning.csv',history)
            checkpoint=dict(architecture=architecture,seed=seed,update=update,actor={n:p for n,p in model.state_dict().items() if not n.startswith('critic')},source=[g[2] for g in groups],validation_loss=val)
            torch.save(checkpoint,output/'last.pt')
            if val<best:best=val;torch.save(checkpoint,output/'best.pt')
            print(architecture,seed,row,flush=True)
    write_json(output/'summary.json',dict(architecture=architecture,seed=seed,updates=update,best_validation=best,seconds=time.monotonic()-start,source=[g[2] for g in groups]))

def config_for(task):
    if task.startswith('seven'):
        config=task_config('five_d3');config.update(model_name='seven_bar_linkage',model_directory=BASE/'model/seven_bar_linkage',target_spread=float(task[-1]),task_name=task)
    elif task=='torus_d2':
        config=task_config(task)
        config['spectral_weight']=.05 # Existing stable first-discovery preset.
    else:config=task_config(task)
    config['actor_reference_horizon']=config['max_steps']
    return config

def make_agent(model,config):
    return ActorCriticAgent(model,max_step_width=config['max_step_width'],gamma=config['gamma'],
        actor_learning_rate=config['actor_learning_rate'],critic_learning_rate=config['critic_learning_rate'],
        entropy_coefficient=config['entropy_coefficient'],value_coefficient=config['value_coefficient'],
        actor_gradient_clip=config['actor_gradient_clip'],critic_gradient_clip=config['critic_gradient_clip'],
        actor_reference_horizon=config['actor_reference_horizon'])

def run(task,architecture,arm,seed,source_seed,budget,checkpoint=None,tag='main',variance=None):
    directory=ROOT/tag/f'{task}_{architecture}_{arm}_s{seed}';directory.mkdir(parents=True,exist_ok=False)
    config=config_for(task);env=GoalEnv(config)
    model=make_model(architecture,env.coordinate_size,seed)
    if architecture=='graph':model.set_geometry(config['model_directory'],config['flatness_subspace_dimension'])
    sourcefile=None
    if arm!='scratch':
        sourcefile=Path(checkpoint) if checkpoint else ROOT/'source'/f'{architecture}_seed{source_seed}'/'best.pt'
        saved=torch.load(sourcefile,weights_only=False)
        missing,unexpected=model.load_state_dict(saved['actor'],strict=False)
        if unexpected or any(not n.startswith('critic') for n in missing):raise ValueError((missing,unexpected))
    if variance is not None:
        logit=np.log((variance-1e-4)/(2-variance))
        for m in (model.mixture_variance,model.width_variance):nn.init.zeros_(m.weight);nn.init.constant_(m.bias,float(logit))
    agent=make_agent(model,config)
    # Sampling streams and critic initialization matched independently of source.
    torch.manual_seed(seed+100000);np.random.seed(seed+100000)
    actor_before={n:p.detach().clone() for n,p in model.state_dict().items() if not n.startswith('critic')}
    write_json(directory/'design.json',dict(task=task,architecture=architecture,arm=arm,seed=seed,source_seed=source_seed,
        config=config,budget=budget,source_checkpoint=sourcefile,source_sha256=digest(sourcefile) if sourcefile else None,
        variance_override=variance,target_coordinates_used=False,deadline_utc='2026-09-14T13:05:07Z'))
    training=arm in ('scratch','finetune');deterministic=arm=='deterministic'
    rows=[];updates=[];total=0;projection_calls=0;episode=0;hit=None;start=time.monotonic();interrupted=False
    while total<budget and time.time()<DEADLINE-120:
        episode+=1;env.configure_episode_tolerance(training);state=env.reset();done=False;reward_sum=0;nodes=[env.node.copy()]
        info={'strict_success':False,'termination_reason':'no_modes'}
        while not done and total<budget:
            if time.time()>=DEADLINE-120:interrupted=True;break
            modes=env.get_modes()
            if modes.shape[1]==0:break
            with (contextlib.nullcontext() if training else torch.no_grad()):
                action=agent.get_action(state,modes,deterministic=deterministic)
            state,reward,done,info=env.step(action.physical_direction,action.requested_width,action.mode_coefficients)
            projection_calls+=int(info.get('projection_attempts',0))
            if training:agent.add_transition(Transition(reward,action.log_probability,action.value,action.entropy))
            total+=1;reward_sum+=reward;nodes.append(env.node.copy())
            if info.get('strict_success'):break
        metrics=env.compactness_metrics();success=bool(info.get('strict_success',False))
        row=dict(episode=episode,steps=env.step_count,cumulative_steps=total,projection_calls=projection_calls,reward=reward_sum,flatness=metrics['flatness_ratio'],spread=metrics['spread'],spread_error=metrics['spread_error'],success=int(success),updates_before=len(updates),seconds=time.monotonic()-start,reason=info.get('termination_reason'))
        rows.append(row)
        if training:agent.finish_episode()
        if success:
            hit=dict(steps=total,projection_calls=projection_calls,episode=episode,seconds=time.monotonic()-start,updates=len(updates))
            np.savez_compressed(directory/'first_hit_path.npz',nodes=np.array(nodes))
            np.savetxt(directory/'first_hit.dat',env.node,delimiter=',');break
        if training and len(agent.finished_episodes)>=config['batch_size']:
            result=agent.update();updates.append(dict(update=len(updates)+1,episode=episode,steps=total,**result))
        if episode%8==0 or episode==1:
            write_csv(directory/'episodes.csv',rows);write_csv(directory/'updates.csv',updates)
            write_json(directory/'progress.json',row)
            print(directory.name,row,flush=True)
        if deterministic or interrupted or env.step_count==0:break
    write_csv(directory/'episodes.csv',rows);write_csv(directory/'updates.csv',updates)
    torch.save(dict(model=model.state_dict(),actor_optimizer=agent.actor_optimizer.state_dict(),critic_optimizer=agent.critic_optimizer.state_dict(),torch_rng=torch.get_rng_state(),total_steps=total),directory/'final.pt')
    frozen_equal=all(torch.equal(p,model.state_dict()[n]) for n,p in actor_before.items())
    result=dict(task=task,architecture=architecture,arm=arm,seed=seed,source_seed=source_seed,budget=budget,steps=total,episodes=episode,
        success=hit is not None,first_hit=hit,projection_calls=projection_calls,seconds=time.monotonic()-start,updates=len(updates),
        stop_reason='success' if hit else ('deadline' if interrupted or time.time()>=DEADLINE-120 else ('deterministic_end' if deterministic else 'budget')),
        frozen_actor_unchanged=frozen_equal,source_sha256=digest(sourcefile) if sourcefile else None)
    write_json(directory/'result.json',result);print(json.dumps(result),flush=True)
    return result

def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('pretrain');s.add_argument('--architecture',choices=['dense','graph'],required=True);s.add_argument('--seed',type=int,required=True);s.add_argument('--updates',type=int,default=5000);s.add_argument('--label');s.add_argument('--data-root')
    s=sub.add_parser('run');s.add_argument('--task',required=True);s.add_argument('--architecture',choices=['dense','graph'],required=True);s.add_argument('--arm',choices=['scratch','finetune','frozen','deterministic'],required=True);s.add_argument('--seed',type=int,required=True);s.add_argument('--source-seed',type=int,default=5601);s.add_argument('--budget',type=int,default=51200);s.add_argument('--checkpoint');s.add_argument('--tag',default='main');s.add_argument('--variance',type=float)
    args=vars(p.parse_args());command=args.pop('command')
    (pretrain if command=='pretrain' else run)(**args)

if __name__=='__main__':main()
