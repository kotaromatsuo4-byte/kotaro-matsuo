import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
import discovery_transfer as d
from RL_main import rollout

class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        deadline=patch.object(d,'DEADLINE',d.time.time()+600);deadline.start();self.addCleanup(deadline.stop)
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_batched_and_agent_interfaces_agree(self):
        for arch in ('dense','graph','projected'):
            config=d.config_for('seven_d4');env=d.GoalEnv(config)
            model=d.make_model(arch,21,5690)
            if isinstance(model,d.GraphActorCritic):model.set_geometry(config['model_directory'])
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
            source=d.make_model('graph',21,5688)
            checkpoint=Path(temp)/'source_only.pt'
            torch.save(dict(actor={n:v for n,v in source.state_dict().items() if not n.startswith('critic')}),checkpoint)
            for arm in ('finetune','frozen','deterministic'):
                result=d.run('torus_d2','graph',arm,5688,5601,2,checkpoint=str(checkpoint))
                self.assertEqual(result['steps'],2)
                self.assertTrue(result['frozen_actor_unchanged'])

    def test_supervision_leaves_critic_and_variance_unchanged(self):
        model=d.make_model('dense',21,5687);groups=d.source_groups('dense');_,t,_=groups[0]
        before={k:v.clone() for k,v in model.state_dict().items()}
        parameters=[p for n,p in model.named_parameters() if not n.startswith('critic') and 'variance' not in n]
        optimizer=torch.optim.Adam(parameters,lr=.001)
        loss,_,_=d.supervised_loss(model,t,t['train'][:32]);optimizer.zero_grad();loss.backward();optimizer.step()
        for key,value in model.state_dict().items():
            if key.startswith('critic') or 'variance' in key:self.assertTrue(torch.equal(before[key],value),key)
        self.assertFalse(torch.equal(before['mixture_mean.weight'],model.mixture_mean.weight))

    def test_resume_exactly_matches_uninterrupted_rl(self):
        config=d.config_for('seven_d4');config['max_steps']=5;config['actor_reference_horizon']=5
        with tempfile.TemporaryDirectory() as temp,patch.object(d,'ROOT',Path(temp)),patch.object(d,'config_for',return_value=config):
            d.run('seven_d4','dense','scratch',5686,5601,40,tag='part')
            previous=Path(temp)/'part/seven_d4_dense_scratch_s5686'
            d.run('seven_d4','dense','scratch',5686,5601,80,tag='resumed',resume=str(previous))
            d.run('seven_d4','dense','scratch',5686,5601,80,tag='whole')
            a=torch.load(Path(temp)/'resumed/seven_d4_dense_scratch_s5686/final.pt',weights_only=False)
            b=torch.load(Path(temp)/'whole/seven_d4_dense_scratch_s5686/final.pt',weights_only=False)
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            torch.testing.assert_close(a['torch_rng'],b['torch_rng'],rtol=0,atol=0)

    def test_aligned_width_follows_supervised_directed_mean(self):
        config=d.config_for('seven_d4');env=d.GoalEnv(config);model=d.make_model('dense',21,5691)
        d.align_width_with_directed_mean(model);agent=d.make_agent(model,config)
        action=agent.get_action(env.get_state(),env.get_modes(),deterministic=True)
        vector=action.physical_direction*action.requested_width
        self.assertGreaterEqual(float(np.dot(vector,model._directed_mean.detach().numpy())),-1e-9)
        before=model.mixture_mean.weight.detach().clone()
        action=agent.get_action(env.get_state(),env.get_modes(),deterministic=False)
        agent.add_transition(d.Transition(1.,action.log_probability,action.value,action.entropy));agent.finish_episode()
        self.assertTrue(np.isfinite(agent.update()['loss']))
        self.assertFalse(torch.equal(before,model.mixture_mean.weight))

    def test_projected_actor_is_invariant_to_mode_basis_rotation(self):
        config=d.config_for('seven_d4');env=d.GoalEnv(config)
        model=d.make_model('projected',21,5692);model.set_geometry(config['model_directory'])
        state=torch.tensor(env.get_state()[None]);modes=torch.tensor(env.get_modes().T,dtype=torch.float32)
        rotation=torch.linalg.qr(torch.randn(len(modes),len(modes)))[0]
        rotated=rotation@modes
        ma,va,fa=model.modal_distribution_parameters(state,modes)
        mb,vb,fb=model.modal_distribution_parameters(state,rotated)
        torch.testing.assert_close(ma@modes,mb@rotated,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(va,vb)
        torch.testing.assert_close(modes.T@torch.diag(va)@modes,rotated.T@torch.diag(vb)@rotated,atol=2e-6,rtol=2e-5)
        direction=torch.nn.functional.normalize(ma@modes,dim=0)
        wa,_=model.width_distribution_parameters(state,direction,fa)
        wb,_=model.width_distribution_parameters(state,direction,fb)
        torch.testing.assert_close(wa,wb)

    def test_random_prefix_proposals_are_counted(self):
        import discovery_random_restart as restart
        calls=[];original=d.ActorCriticAgent.get_action
        def counted(agent,*args,**kwargs):
            calls.append(1);return original(agent,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp,patch.object(restart,'ROOT',Path(temp)),patch.object(restart,'DEADLINE',d.time.time()+600),patch.object(d.ActorCriticAgent,'get_action',counted):
            result=restart.run_restart('graph','untrained_deterministic',6399,5601,budget=50)
            self.assertEqual(result['steps'],50)
            self.assertEqual(len(calls),10)
            self.assertTrue(result['prefix_proposals_counted'])

    def test_new_d4_experience_cannot_leak_into_original_d4_test(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(d,'ROOT',Path(temp)):
            model=d.make_model('graph',21,5693);checkpoint=Path(temp)/'new_known_d4.pt'
            torch.save(dict(actor={k:v for k,v in model.state_dict().items() if not k.startswith('critic')},
                source=[dict(discovery_manifest='sequential-stage-only')]),checkpoint)
            with self.assertRaisesRegex(ValueError,'New D4 source'):
                d.run('seven_d4','graph','frozen',5693,5601,2,checkpoint=str(checkpoint))

    def test_goal_ablation_changes_policy_input_not_environment_target(self):
        observed=[];original=d.ActorCriticAgent.get_action
        def observe(agent,state,*args,**kwargs):
            observed.append(float(state[-1]));return original(agent,state,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp,patch.object(d,'ROOT',Path(temp)),patch.object(d.ActorCriticAgent,'get_action',observe):
            model=d.make_model('graph',21,5694);checkpoint=Path(temp)/'source.pt'
            torch.save(dict(actor={k:v for k,v in model.state_dict().items() if not k.startswith('critic')}),checkpoint)
            d.run('seven_d4','graph','frozen',5694,5601,2,checkpoint=str(checkpoint),policy_goal=3.)
            design=d.json.loads((Path(temp)/'main/seven_d4_graph_frozen_s5694/design.json').read_text())
            self.assertEqual(observed,[3.,3.]);self.assertEqual(design['config']['target_spread'],4.)

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
