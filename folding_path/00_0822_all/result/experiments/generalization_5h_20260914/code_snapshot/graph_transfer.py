"""Variable-node graph/modal student and seven-bar -> torus transfer study.

Connectivity, coordinates, fix masks, modes, normalized goal and target subspace
dimension are observations. Target coordinates are exclusive to the oracle.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import mechanics
from supervised_folding import BASE, Geometry, generate_starts, write_csv
from transfer_study import propose, path_rollout, build_records, pack, train_fixed, evaluate_model


class SpatialGeometry(Geometry):
    def __init__(self,directory,step_width=.2):
        self.directory=Path(directory)
        self.initial=np.loadtxt(self.directory/'node_initial.dat',delimiter=',')
        self.member=np.loadtxt(self.directory/'member.dat',delimiter=',',dtype=int).reshape(-1,2)
        self.fix=np.loadtxt(self.directory/'fix.dat',delimiter=',',dtype=int).reshape(-1,2)
        self.coordinate_size=self.initial.size
        self.free=mechanics.index_free(len(self.initial),self.fix)
        self.lengths=np.linalg.norm(self.initial[self.member[:,0]]-self.initial[self.member[:,1]],axis=1)
        self.scale=float(self.lengths.max())
        self.dimension=2
        self.initial_potential=mechanics.compute_flatness_potential(self.initial,2)
        self.step_width=step_width
        self.pairs=mechanics.nonadjacent_bar_pairs(self.member)
        self.collision_checker=mechanics.ContinuousBarCollisionChecker(self.member)

    def advance(self,node,direction,width=None):
        candidate,accepted,_,reason=propose(self,node,direction,self.step_width if width is None else width,True)
        return candidate,reason

    def metrics(self,node,goal):
        flatness=mechanics.compute_flatness_potential(node,2)/self.initial_potential
        spread=mechanics.projected_spread(node,2,self.scale)
        strain=mechanics.current_strain(node,self.member,self.lengths)
        return dict(flatness=float(flatness),spread=float(spread),spread_error=float(abs(spread-goal)),
                    strain=float(strain),success=int(flatness<=1e-3 and abs(spread-goal)<=.05 and strain<=1e-8))


class GraphModalStudent(nn.Module):
    """Same parameter tensors for all node/mode counts; graph buffers are not weights."""
    def __init__(self,hidden=32):
        super().__init__()
        self.hidden=hidden
        self.predicts_width=True
        self.node_encoder=nn.Sequential(nn.Linear(13,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh()) for _ in range(2)])
        self.coefficient=nn.Linear(hidden,1)
        self.width=nn.Sequential(nn.Linear(hidden+2,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.register_buffer('adjacency',torch.zeros(1,1),persistent=False)
        self.register_buffer('node_constants',torch.zeros(1,5),persistent=False)
        self.dimension=1

    def set_geometry(self,g):
        n=len(g.initial)
        adjacency=np.zeros((n,n),np.float32)
        length_sum=np.zeros(n,np.float32)
        for (a,b),length in zip(g.member,g.lengths):
            adjacency[a,b]=adjacency[b,a]=1
            length_sum[a]+=length/g.scale
            length_sum[b]+=length/g.scale
        degree=adjacency.sum(1)
        self.adjacency=torch.from_numpy(adjacency/np.maximum(degree[:,None],1))
        fixed=np.zeros((n,3),np.float32)
        fixed[g.fix[:,0],g.fix[:,1]]=1
        self.node_constants=torch.from_numpy(np.column_stack([fixed,degree/6,length_sum/np.maximum(degree,1)]).astype(np.float32))
        self.dimension=getattr(g,'dimension',1)

    def forward(self,inputs,modes,teacher_direction=None):
        batch,k,coordinates=modes.shape
        n=coordinates//3
        if n!=self.adjacency.shape[0]:
            raise ValueError('Call set_geometry before changing structures')
        # Drop all-zero padding beyond the largest actual mode count in this batch.
        counts=(modes.square().sum(-1)>1e-10).sum(1)
        used=max(1,int(counts.max()))
        modes=modes[:,:used]
        mask=modes.square().sum(-1)>1e-10
        positions=inputs[:,:-1].reshape(batch,n,3)
        positions=positions-positions.mean(1,keepdim=True)
        goal=inputs[:,-1:]
        constants=self.node_constants[None,None].expand(batch,used,-1,-1)
        features=torch.cat([positions[:,None].expand(-1,used,-1,-1),
                            modes.reshape(batch,used,n,3),constants,
                            goal[:,None,None,:].expand(-1,used,n,-1),
                            inputs.new_full((batch,used,n,1),self.dimension)],-1)
        hidden=self.node_encoder(features)
        for layer in self.messages:
            neighbor=torch.einsum('ij,bkjh->bkih',self.adjacency,hidden)
            hidden=hidden+layer(torch.cat([hidden,neighbor],-1))
        modal=hidden.mean(2)
        coefficients=torch.tanh(self.coefficient(modal).squeeze(-1))*mask
        direction=F.normalize(torch.bmm(coefficients[:,None],modes).squeeze(1),dim=-1,eps=1e-8)
        pooled=(modal*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
        # Goal and subspace condition width; no target coordinates.
        width=torch.sigmoid(self.width(torch.cat([pooled,goal,goal.new_full(goal.shape,self.dimension)],-1)).squeeze(-1))
        return direction,width


def prepare_torus(root,support=16,validation=8,test=16):
    output=root/'torus'
    output.mkdir(exist_ok=False)
    g=SpatialGeometry(BASE/'model/4unit_torus')
    reference=BASE/'result/first_hit/torus/rl/alpha_0p5/seed8101'
    target=np.loadtxt(reference/'step185.dat')
    np.testing.assert_allclose(target[g.fix[:,0],g.fix[:,1]],g.initial[g.fix[:,0],g.fix[:,1]],atol=1e-10)
    if not g.metrics(target,2.)['success']:
        raise ValueError('Saved target is not a feasible successful torus state')
    np.savetxt(output/'target.dat',target,delimiter=',')
    (output/'design.json').write_text(json.dumps(dict(target_source=str(reference/'step185.dat'),
        target_sha256=hashlib.sha256((reference/'step185.dat').read_bytes()).hexdigest(),
        target_use='only torus teacher and adaptation; excluded from bar pretraining',
        initial_sampling='independent feasible walks from original torus; not samples from target trajectory',
        support=support,validation=validation,test=test,goal=2.,dimension=2),indent=2),encoding='utf-8')
    starts=generate_starts(g,support,np.random.default_rng(66001),5,25)
    val=generate_starts(g,validation,np.random.default_rng(66002),5,25,starts)
    tests=generate_starts(g,test,np.random.default_rng(66003),5,25,[*starts,*val])
    np.savez_compressed(output/'splits.npz',support=starts,validation=val,test=tests)
    records,rows=[],[]
    for i,initial in enumerate(starts):
        path,metrics=path_rollout(g,initial,2.,target=target,max_steps=400)
        np.savez_compressed(output/f'teacher_{i:03d}.npz',nodes=path)
        records.extend(build_records(g,path,target,2.,i))
        rows.append(dict(path_id=i,**metrics))
        print('torus prepare',i+1,'/',support,'success',metrics['success'],'records',len(records),flush=True)
        np.savez_compressed(output/'support_data.npz',**pack(records))
        write_csv(output/'teacher_rollouts.csv',rows)


def study(root,seeds,budgets,source_updates,updates,target_structure,split):
    source_g=Geometry(BASE/'model/seven_bar_linkage')
    source=dict(np.load(root/'source_data.npz'))
    target_root=root/'torus' if target_structure=='torus' else root
    g=SpatialGeometry(BASE/'model/4unit_torus') if target_structure=='torus' else source_g
    support=dict(np.load(target_root/'support_data.npz'))
    starts=np.load(target_root/'splits.npz')[split]
    goal=2. if target_structure=='torus' else 4.
    output=target_root/f'graph_src{source_updates}_u{updates}_{split}'
    output.mkdir(exist_ok=True)
    for seed in seeds:
        torch.manual_seed(seed)
        model=GraphModalStudent()
        initial={k:v.clone() for k,v in model.state_dict().items()}
        model.set_geometry(source_g)
        # Reuse exact source weights for both seven-bar and torus adaptation.
        source_path=root/f'graph_source_seed{seed}_u{source_updates}.pt'
        if source_path.exists():
            model.load_state_dict(torch.load(source_path,weights_only=True)['model'])
        else:
            logs=train_fixed(model,source,source_g.scale,source_updates,seed+100,batch=32)
            torch.save(dict(model=model.state_dict(),seed=seed),source_path)
            write_csv(source_path.with_suffix('.csv'),logs)
        pretrained={k:v.clone() for k,v in model.state_dict().items()}
        model.set_geometry(g)
        for budget in budgets:
            mask=support['path_id']<budget
            data={k:v[mask] for k,v in support.items()}
            for arm,weights in (('scratch',initial),('pretrained',pretrained)):
                prefix=f'seed{seed}_{arm}_n{budget:02d}'
                result_path=output/f'{prefix}_evaluation.csv'
                if result_path.exists(): continue
                model.load_state_dict(weights)
                if budget:
                    logs=train_fixed(model,data,g.scale,updates,seed+200,batch=32,lr=3e-4)
                    write_csv(output/f'{prefix}_learning.csv',logs)
                torch.save(dict(model=model.state_dict(),seed=seed,budget=budget,arm=arm),output/f'{prefix}.pt')
                rows=evaluate_model(g,model,starts,goal)
                write_csv(result_path,[dict(seed=seed,budget=budget,arm=arm,labels=int(mask.sum()),**r) for r in rows])
                print('graph',target_structure,prefix,'success',np.mean([r['success'] for r in rows]),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','study'))
    p.add_argument('root',type=Path)
    p.add_argument('--target',choices=('seven','torus'),default='torus')
    p.add_argument('--seeds',type=int,nargs='+',default=[5501,5502,5503])
    p.add_argument('--budgets',type=int,nargs='+',default=[0,1,4,16])
    p.add_argument('--source-updates',type=int,default=5000)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--split',choices=('validation','test'),default='test')
    args=p.parse_args()
    if args.command=='prepare': prepare_torus(args.root)
    else: study(args.root,args.seeds,args.budgets,args.source_updates,args.updates,args.target,args.split)
