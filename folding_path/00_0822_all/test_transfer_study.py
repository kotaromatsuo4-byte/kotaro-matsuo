import unittest
import numpy as np
import torch
import mechanics
from supervised_folding import BASE, Geometry, load_targets
from transfer_study import ModalStudent, CategoricalModalStudent, action, oracle, propose, build_records, pack, tensors, train_fixed, target_d4


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.g = Geometry(BASE / 'model/seven_bar_linkage')

    def test_oracle_label_width_reproduces_step(self):
        g=self.g
        for target in [*load_targets(g).values(), target_d4()]:
            direction,width,next_node,status,calls=oracle(g,g.initial,target)
            self.assertEqual(status,'accepted')
            self.assertGreater(width,0)
            self.assertLessEqual(width,g.step_width)
            reproduced,accepted,_,_=propose(g,g.initial,direction,width)
            np.testing.assert_allclose(next_node,reproduced,atol=1e-10)
            self.assertLess(np.linalg.norm(next_node-target),np.linalg.norm(g.initial-target))
            self.assertLessEqual(mechanics.current_strain(next_node,g.member,g.lengths),1e-8)

    def test_modal_padding_and_feasible_direction(self):
        torch.manual_seed(1)
        g=self.g
        data=pack(build_records(g,[g.initial],target_d4(),4,0))
        t=tensors(data,g.scale)
        model=ModalStudent(21,32)
        direction,width=model(t['inputs'],t['modes'])
        # Additional padded zeros must have no effect on pooling or coefficients.
        padded=torch.cat([t['modes'],torch.zeros(1,7,21)],1)
        d2,w2=model(t['inputs'],padded)
        torch.testing.assert_close(direction,d2)
        torch.testing.assert_close(width,w2)
        jac=mechanics.dedx(g.initial,g.member,g.free)
        np.testing.assert_allclose(jac@direction.detach().numpy()[0,g.free],0,atol=1e-6)
        self.assertTrue(0<float(width.detach())<1)

    def test_fitting_and_rng_reproducibility(self):
        g=self.g
        records=[]
        for goal,target in load_targets(g).items():
            records.extend(build_records(g,[g.initial],target,goal,0))
        data=pack(records)
        torch.manual_seed(81)
        a=ModalStudent(21,32)
        b=ModalStudent(21,32)
        b.load_state_dict(a.state_dict())
        h=train_fixed(a,data,g.scale,200,44,lr=.003)
        train_fixed(b,data,g.scale,200,44,lr=.003)
        for k,v in a.state_dict().items():
            torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        self.assertLess(h[-1]['direction_loss'],h[0]['direction_loss'])
        d,w=action(a,g,g.initial,4.)
        self.assertTrue(np.isfinite(d).all())
        self.assertTrue(0<w<=g.step_width)

    def test_categorical_d4_columns_untrained_on_source(self):
        torch.manual_seed(22)
        model=CategoricalModalStudent(21,32)
        records=[]
        for goal,target in load_targets(self.g).items():
            records.extend(build_records(self.g,[self.g.initial],target,goal,0))
        encoder_column=model.encoder[0].weight[:,23].detach().clone()
        width_column=model.width[0].weight[:,23].detach().clone()
        train_fixed(model,pack(records),self.g.scale,20,44)
        torch.testing.assert_close(model.encoder[0].weight[:,23],encoder_column,atol=0,rtol=0)
        torch.testing.assert_close(model.width[0].weight[:,23],width_column,atol=0,rtol=0)
        direction,width=action(model,self.g,self.g.initial,4.)
        self.assertTrue(np.isfinite(direction).all())


if __name__=='__main__': unittest.main()
