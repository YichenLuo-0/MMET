import ansys.mapdl.reader as pymapdl
import numpy as np
from matplotlib import pyplot as plt
from overrides import overrides

from cases.case import Case


class Thermodynamics(Case):
    def __init__(self, k, x_min, x_max, y_min, y_max, fea_path):
        super().__init__(5, x_min, x_max, y_min, y_max, fea_path)
        self.k = k

    @overrides
    def load_fea_result(self, fea_path):
        # Load the finite element analysis results
        rst = pymapdl.read_binary(fea_path)

        # Get the nodes and ground truth
        nodes = np.array(rst.grid.points[:, :2])
        num_data = rst.time_values.shape[0]

        nodes_all = []
        temp_all = []

        for i in range(num_data):
            # Get the displacement and temperature
            temp = rst.nodal_temperature(i)[1]

            # Remove the nan values
            not_nan = ~np.isnan(temp)
            nodes_all.append(nodes[not_nan])
            temp_all.append(temp[not_nan])

        # Concatenate the displacement and temperature
        return num_data, nodes_all, [temp_all]

    @overrides
    def get_bc_dims(self):
        # [temperature_dim, flux_dim]
        return [1, 4]

    @overrides
    def get_ground_truth(self, x, y, index):
        if index >= self.num_data:
            raise ValueError("The index is out of range!")

        nodes = np.array([x, y]).T
        distances, indices = self.kdtree[index].query(nodes)
        temp = self.ground_truth[0][index][indices]
        return np.array([temp]).T
