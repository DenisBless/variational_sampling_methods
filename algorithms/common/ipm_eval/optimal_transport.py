from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import jax.numpy as jnp
from ott.tools import sinkhorn_divergence
from ott.geometry import pointcloud


class OT:
    def __init__(self, gt_samples, epsilon=1e-3):
        self.groundtruth = gt_samples
        self.epsilon = epsilon

    def compute_OT(self, model_samples, entropy_reg=True):
        """
        Entropy regularized optimal transport cost (see https://ott-jax.readthedocs.io/en/latest/tutorials/point_clouds.html)
        """
        geom = pointcloud.PointCloud(self.groundtruth, model_samples, epsilon=self.epsilon)
        # Define a linear problem with that cost structure.
        ot_prob = linear_problem.LinearProblem(geom)
        # Create a Sinkhorn solver
        solver = sinkhorn.Sinkhorn()
        # Solve OT problem
        ot = solver(ot_prob)
        if entropy_reg:
            # Return entropy regularized OT (eOT) cost
            cost = ot.reg_ot_cost
        else:
            # OT cost (without entropy)
            cost = jnp.sum(ot.matrix * ot.geom.cost_matrix)

        return cost


class SD:
    def __init__(self, gt_samples, epsilon=1e-3):
        self.groundtruth = gt_samples
        self.epsilon = epsilon

    def compute_SD(self, model_samples):
        """
        Entropy regularized debiased optimal transport (Sinkhorn divergence - SD) cost (see https://ott-jax.readthedocs.io/en/latest/tutorials/point_clouds.html)
        """

        geom = pointcloud.PointCloud(self.groundtruth, model_samples, epsilon=1e-3)

        sd = sinkhorn_divergence.sinkhorn_divergence(
            geom,
            x=geom.x,
            y=geom.y,
        ).divergence

        return sd
