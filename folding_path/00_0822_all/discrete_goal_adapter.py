"""D4-gated low-rank adaptation of frozen D2/D3 features.

D2/D3 computations are exactly preserved. Frozen random-feature adapters form
the matched trainable-parameter control; full fine-tuning is a separate control.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transfer_study import ModalStudent,train_fixed,evaluate_model
from supervised_folding import BASE,Geometry,write_csv


class DiscreteGoalAdapter(nn.Module):
    predicts_width=True
    def __init__(self,base,rank=8):
        super().__init__();self.base=base;self.rank=rank
        for p in self.base.parameters(): p.requires_grad_(False)
        h=base.hidden
        self.mode_adapter=nn.Sequential(nn.Linear(h,rank,bias=False),nn.Tanh(),nn.Linear(rank,h,bias=False))
        self.width_adapter=nn.Sequential(nn.Linear(h,rank,bias=False),nn.Tanh(),nn.Linear(rank,1,bias=False))
        nn.init.zeros_(self.mode_adapter[-1].weight);nn.init.zeros_(self.width_adapter[-1].weight)

    def forward(self,inputs,modes,teacher_direction=None):
        gate=(inputs[:,-1:]==4.).to(inputs.dtype)
        mask=modes.square().sum(-1)>1e-10
        expanded=inputs[:,None].expand(-1,modes.shape[1],-1)
        features=self.base.encoder(torch.cat([expanded,modes],-1))
        features=features+gate[:,:,None]*self.mode_adapter(features)
        coefficients=torch.tanh(self.base.coefficient(features).squeeze(-1))*mask
        direction=F.normalize(torch.bmm(coefficients[:,None],modes).squeeze(1),dim=-1,eps=1e-8)
        pooled=(features*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
        selected=direction if teacher_direction is None else teacher_direction
        hidden=self.base.width[1](self.base.width[0](torch.cat([inputs,selected,pooled],-1)))
        logits=self.base.width[2](hidden)+gate*self.width_adapter(hidden)
        return direction,torch.sigmoid(logits.squeeze(-1))


def study(root,seeds,budgets,updates,rank,split,lr=3e-4):
    g=Geometry(BASE/'model/seven_bar_linkage');support=dict(np.load(root/'support_data.npz'))
    starts=np.load(root/'splits.npz')[split]
    suffix='' if lr==3e-4 else f'_lr{lr:g}'
    output=root/f'discrete_adapter_r{rank}_u{updates}_{split}{suffix}';output.mkdir(exist_ok=True)
    for seed in seeds:
        torch.manual_seed(seed);random_base=ModalStudent()
        initial={k:v.clone() for k,v in random_base.state_dict().items()}
        pretrained=torch.load(root/'modal_u2000_src10000_test'/f'source_seed{seed}.pt',weights_only=True)['model']
        for budget in budgets:
            mask=support['path_id']<budget;data={k:v[mask] for k,v in support.items()}
            for arm,weights in (('scratch',initial),('pretrained',pretrained)):
                prefix=f'seed{seed}_{arm}_n{budget:02d}'
                if (output/f'{prefix}_evaluation.csv').exists(): continue
                base=ModalStudent();base.load_state_dict(weights)
                torch.manual_seed(seed+400);model=DiscreteGoalAdapter(base,rank)
                if budget:
                    logs=train_fixed(model,data,g.scale,updates,seed+200,lr=lr)
                    write_csv(output/f'{prefix}_learning.csv',logs)
                torch.save(dict(model=model.state_dict(),rank=rank,seed=seed,budget=budget,arm=arm,lr=lr,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)),output/f'{prefix}.pt')
                rows=[]
                for goal in (2.,3.,4.): rows.extend(evaluate_model(g,model,starts,goal))
                write_csv(output/f'{prefix}_evaluation.csv',[dict(seed=seed,arm=arm,budget=budget,labels=int(mask.sum()),**r) for r in rows])
                print('discrete adapter',prefix,{d:np.mean([r['success'] for r in rows if r['goal']==d]) for d in (2.,3.,4.)},flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[5501]);p.add_argument('--budgets',type=int,nargs='+',default=[1,4,16,32])
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--rank',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    a=p.parse_args();study(a.root,a.seeds,a.budgets,a.updates,a.rank,a.split,a.lr)
