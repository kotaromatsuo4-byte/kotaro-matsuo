import unittest
import numpy as np
from graph_transfer import SpatialGeometry
from analytic_goal_control import energy_gradient
from supervised_folding import BASE


class AnalyticControlTests(unittest.TestCase):
    def test_directional_derivative_matches_finite_difference(self):
        g=SpatialGeometry(BASE/'model/4unit_torus');rng=np.random.default_rng(77021)
        # Break exact diameter ties; the max-diameter energy is piecewise smooth.
        node=g.initial+.03*rng.normal(size=g.initial.shape)
        direction=rng.normal(size=node.shape);direction/=np.linalg.norm(direction)
        energy,gradient=energy_gradient(g,node,2.)
        epsilon=1e-6
        fd=(energy_gradient(g,node+epsilon*direction,2.)[0]-energy_gradient(g,node-epsilon*direction,2.)[0])/(2*epsilon)
        self.assertAlmostEqual(fd,float(gradient@direction.reshape(-1)),places=6)


if __name__=='__main__': unittest.main()
