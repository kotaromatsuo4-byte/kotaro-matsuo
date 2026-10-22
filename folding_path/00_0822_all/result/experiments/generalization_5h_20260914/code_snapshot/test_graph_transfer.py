import copy
import unittest
import numpy as np
import torch
from supervised_folding import BASE,Geometry
from transfer_study import pad_modes
from graph_transfer import GraphModalStudent,SpatialGeometry


class GraphTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.model=GraphModalStudent()

    def inputs(self,g):
        return (torch.tensor(np.append(g.initial.reshape(-1)/g.scale,2.)[None],dtype=torch.float32),
                torch.from_numpy(pad_modes(g,g.initial))[None])

    def test_same_weights_across_node_counts_and_finite_gradients(self):
        saved={k:v.clone() for k,v in self.model.state_dict().items()}
        for g in (Geometry(BASE/'model/seven_bar_linkage'),SpatialGeometry(BASE/'model/4unit_torus')):
            self.model.set_geometry(g)
            x,modes=self.inputs(g)
            d,w=self.model(x,modes)
            self.assertEqual(d.shape,(1,g.coordinate_size))
            self.assertTrue(torch.isfinite(d).all())
            self.assertAlmostEqual(float(d.detach().norm()),1.,places=5)
            self.model.zero_grad()
            (d.sum()+w.sum()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
            for k,v in self.model.state_dict().items():
                torch.testing.assert_close(v,saved[k],rtol=0,atol=0)

    def test_node_permutation_equivariance(self):
        g=Geometry(BASE/'model/seven_bar_linkage')
        self.model.set_geometry(g)
        x,modes=self.inputs(g)
        d,w=self.model(x,modes)
        order=np.array([3,1,6,0,4,2,5])
        inverse=np.argsort(order)
        other=copy.copy(g)
        other.initial=g.initial[order]
        other.member=inverse[g.member]
        other.fix=g.fix.copy()
        other.fix[:,0]=inverse[g.fix[:,0]]
        self.model.set_geometry(other)
        x2=torch.cat([x[:,:-1].reshape(1,7,3)[:,order].reshape(1,21),x[:,-1:]],1)
        m2=modes.reshape(1,21,7,3)[:,:,order].reshape(1,21,21)
        d2,w2=self.model(x2,m2)
        torch.testing.assert_close(d2.reshape(1,7,3),d.reshape(1,7,3)[:,order],atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(w,w2,atol=1e-6,rtol=1e-5)


if __name__=='__main__': unittest.main()
