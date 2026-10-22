import unittest
import numpy as np
import torch
from transfer_study import ModalStudent,pad_modes,train_fixed
from discrete_goal_adapter import DiscreteGoalAdapter
from supervised_folding import BASE,Geometry


class AdapterTests(unittest.TestCase):
    def test_initial_equivalence_and_exact_source_preservation_after_training(self):
        torch.set_num_threads(1);torch.manual_seed(81)
        g=Geometry(BASE/'model/seven_bar_linkage');base=ModalStudent();model=DiscreteGoalAdapter(base)
        x=torch.tensor(np.stack([np.append(g.initial.reshape(-1)/g.scale,d) for d in (2.,3.,4.)]),dtype=torch.float32)
        modes=torch.from_numpy(np.repeat(pad_modes(g,g.initial)[None],3,0))
        before=base(x,modes)
        for a,b in zip(before,model(x,modes)): torch.testing.assert_close(a,b,atol=0,rtol=0)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
        for _ in range(5):
            d,w=model(x[2:],modes[2:]);loss=d[:,0].sum()+w.sum()
            optimizer.zero_grad();loss.backward();optimizer.step()
        after=model(x,modes)
        for a,b in zip(before,after): torch.testing.assert_close(a[:2],b[:2],atol=0,rtol=0)
        self.assertGreater(float((before[1][2]-after[1][2]).abs().detach()),1e-5)
        self.assertTrue(all(p.grad is None for p in base.parameters()))


if __name__=='__main__': unittest.main()
