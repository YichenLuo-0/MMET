import ansys.mapdl.reader as pymapdl
import numpy as np


class Beam:
    def __init__(self, num_data, e, nu, l, h, fea_path=None):
        # Initialize the material properties
        self.num_data = num_data
        self.e = e
        self.nu = nu
        self.phi = (e * nu) / ((1 + nu) * (1 - 2 * nu))
        self.mu = e / (2 * (1 + nu))

        # Initialize the geometry limits
        self.l = l
        self.h = h
        self.q0 = [(i / 10) + 10.1 for i in range(num_data)]

        # Initialize the mesh geometry from the FEA results
        _, self.nodes, _ = self.load_fea_result(fea_path)

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

    def get_fea_mesh_size(self, index):
        if index >= self.num_data:
            raise ValueError("The index is out of range!")
        return self.nodes[index].shape[0]

    def get_fea_mesh(self, index):
        if index >= self.num_data:
            raise ValueError("The index is out of range!")
        return self.nodes[index][:, 0], self.nodes[index][:, 1]

    def sample_boundary_points(self, n_boundary):
        n_top_bottom = n_boundary // 12 * 5
        n_left_right = n_boundary // 12

        x_top = np.linspace(0, 5, n_top_bottom)
        y_top = np.ones(n_boundary) * 0.5
        x_bottom = np.linspace(0, 5, n_top_bottom)
        y_bottom = np.ones(n_boundary) * -0.5
        x_left = np.zeros(n_boundary)
        y_left = np.linspace(-0.5, 0.5, n_left_right)
        x_right = np.ones(n_boundary) * 5
        y_right = np.linspace(-0.5, 0.5, n_left_right)

        return np.concatenate([x_top, x_bottom, x_left, x_right]), np.concatenate([y_top, y_bottom, y_left, y_right])

    def get_points_random(self, n):
        x_min = 0
        x_max = self.l / 2
        y_min = self.h / 2
        y_max = self.h / 2

        x = np.random.uniform(x_min, x_max, n)
        y = np.random.uniform(y_min, y_max, n)
        return x, y

    def get_boundary_conditions(self, x, y, index):
        bc = np.zeros([x.shape[0], 7])
        for i in range(x.shape[0]):
            x_idx = x[i]
            y_idx = y[i]
            bc[i] = self.bc_generator(x_idx, y_idx, index)
        return bc

    def bc_generator(self, x, y, index):
        # Left face
        if x == 0:
            bc = np.array([2, 0, 0, 0, 0, 0, 0])

        # Right face
        elif x == 5:
            m_ = -y * (index - 50) * 0.2
            bc = np.array([1, 1, 0, m_, 0, 0, 0])

        # Lower and upper faces
        elif y == 0.5 or y == -0.5:
            bc = np.array([1, 0, 0, 0, 0, 0, 0])

        # 其他内部点无边界条件
        else:
            bc = np.array([0, 0, 0, 0, 0, 0, 0])
        return bc

    def get_ground_truth(self, x, y, index):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        if index >= self.num_data:
            raise ValueError("The index is out of range!")

        m = index - 50

        u = (3 * m * (2 * self.mu + self.phi)) / (self.mu * (self.phi + self.mu) * self.l ** (self.h ** 3)) * x * y
        v = (-(3 * m) / (2 * self.mu * (self.phi + self.mu) * self.l ** (self.h ** 3)) *
             (((2 * self.mu + self.phi) * (x ** 2)) + (self.phi * (y ** 2))))

        du_dx = (3 * m * (2 * self.mu + self.phi)) / (self.mu * (self.phi + self.mu) * self.l ** (self.h ** 3)) * y
        dv_dy = -(3 * m) / (2 * self.mu * (self.phi + self.mu) * self.l ** (self.h ** 3)) * (self.phi * (y * 2))

        epsilon_x = du_dx
        epsilon_y = dv_dy

        sigma_x = (self.e / (1 - self.nu ** 2)) * (epsilon_x + self.nu * epsilon_y)
        sigma_y = (self.e / (1 - self.nu ** 2)) * (epsilon_y + self.nu * epsilon_x)
        tau_xy = np.zeros_like(x)

        return np.concatenate([u, v, sigma_x, sigma_y, tau_xy], axis=1)
