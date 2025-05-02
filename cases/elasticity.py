import ansys.mapdl.reader as pymapdl
import numpy as np
from overrides import overrides

from cases.case import Case


class Elasticity(Case):
    def __init__(self, e, nu, x_min, x_max, y_min, y_max, fea_path):
        super().__init__(6, x_min, x_max, y_min, y_max, fea_path)
        self.e = e
        self.nu = nu
        self.phi = (e * nu) / ((1 + nu) * (1 - 2 * nu))
        self.mu = e / (2 * (1 + nu))

        self.disp = self.ground_truth[0]
        self.stress = self.ground_truth[1]

    @overrides
    def load_fea_result(self, fea_path):
        # Load the finite element analysis results
        rst = pymapdl.read_binary(fea_path)

        # Get the nodes and ground truth
        nodes = np.array(rst.grid.points[:, :2])
        num_data = rst.time_values.shape[0]

        nodes_all = []
        disp_all = []
        stress_all = []

        for i in range(num_data):
            # Get the displacement and stress
            disp = rst.nodal_displacement(i)[1]
            stress = rst.nodal_stress(i)[1]
            stress = stress[:, [0, 1, 3]]

            # Remove the nan values
            not_nan = ~np.isnan(stress[:, 0]) & ~np.isnan(stress[:, 1]) & ~np.isnan(stress[:, 2])
            nodes_all.append(nodes[not_nan])
            disp_all.append(disp[not_nan])
            stress_all.append(stress[not_nan])

        # Concatenate the displacement and stress
        return num_data, nodes_all, [disp_all, stress_all]

    @overrides
    def get_bc_dims(self):
        # [force_dim, displacement_dim]
        return [4, 2]

    @overrides
    def get_ground_truth(self, x, y, index):
        if index >= self.num_data:
            raise ValueError("The index is out of range!")

        nodes = np.array([x, y]).T
        distances, indices = self.kdtree[index].query(nodes)
        disp = self.disp[index][indices] * 10e10
        stress = self.stress[index][indices]
        return np.concatenate([disp, stress], axis=-1)
