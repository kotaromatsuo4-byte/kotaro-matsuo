import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
import discovery_transfer as d
from RL_main import rollout

class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_batched_and_agent_interfaces_agree(self):
        for arch in ('dense','graph'):
            config=d.config_for('seven_d4');env=d.GoalEnv(config)
            model=d.make_model(arch,21,5690)
            if arch=='graph':model.set_geometry(config['model_directory'])
            state=torch.tensor(env.get_state()[None]);modes=torch.tensor(env.get_modes().T,dtype=torch.float32)
            mean,features,mask=d.batch_modal(model,state,modes[None])
            scalar,variance,feat=model.modal_distribution_parameters(state,modes)
            torch.testing.assert_close(mean[0],scalar);torch.testing.assert_close(features[0],feat)
            physical=torch.nn.functional.normalize(mean@modes,dim=-1)
            physical,_=d.canonical_batch(physical)
            wm=d.batch_width(model,state,physical,features,mask)
            ws,_=model.width_distribution_parameters(state,physical[0],feat)
            torch.testing.assert_close(wm[0],ws)
            d.make_agent(model,config) # Checks exhaustive disjoint parameter sets.

    def test_legacy_rollout_update_equivalence(self):
        config=d.config_for('seven_d4');config['max_steps']=20;config['actor_reference_horizon']=20
        seed=5689;env=d.GoalEnv(config);model=d.make_model('dense',21,seed);agent=d.make_agent(model,config)
        torch.manual_seed(seed+100000);np.random.seed(seed+100000)
        rewards=[]
        for _ in range(8):
            summary,_=rollout(env,agent,training=True);rewards.append(summary['total_reward'])
        agent.update()
        with tempfile.TemporaryDirectory() as temp,patch.object(d,'ROOT',Path(temp)),patch.object(d,'config_for',return_value=config):
            result=d.run('seven_d4','dense','scratch',seed,5601,160)
            saved=torch.load(Path(temp)/'main'/f'seven_d4_dense_scratch_s{seed}'/'final.pt',weights_only=False)
            for key,value in model.state_dict().items():torch.testing.assert_close(saved['model'][key],value,rtol=0,atol=0)
            self.assertEqual(result['updates'],1)

    def test_target_execution_does_not_read_teacher(self):
        original=np.loadtxt
        def allowed(path,*args,**kwargs):
            self.assertIn(Path(path).name,('node_initial.dat','member.dat','fix.dat'))
            return original(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp,patch.object(d,'ROOT',Path(temp)),patch.object(np,'loadtxt',side_effect=allowed),patch.object(np,'load',side_effect=AssertionError('No target dataset allowed')):
            result=d.run('torus_d2','graph','scratch',5688,5601,2)
            self.assertEqual(result['steps'],2)

    def test_supervision_leaves_critic_and_variance_unchanged(self):
        model=d.make_model('dense',21,5687);groups=d.source_groups('dense');_,t,_=groups[0]
        before={k:v.clone() for k,v in model.state_dict().items()}
        parameters=[p for n,p in model.named_parameters() if not n.startswith('critic') and 'variance' not in n]
        optimizer=torch.optim.Adam(parameters,lr=.001)
        loss,_,_=d.supervised_loss(model,t,t['train'][:32]);optimizer.zero_grad();loss.backward();optimizer.step()
        for key,value in model.state_dict().items():
            if key.startswith('critic') or 'variance' in key:self.assertTrue(torch.equal(before[key],value),key)
        self.assertFalse(torch.equal(before['mixture_mean.weight'],model.mixture_mean.weight))

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
