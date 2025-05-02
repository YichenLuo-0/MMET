from abc import abstractmethod

import numpy as np
from scipy.spatial import cKDTree


class Case:
    def __init__(self, d_bc, x_min, x_max, y_min, y_max, fea_path):
        self.d_bc = d_bc
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        if fea_path is not None:
            self.num_data, self.nodes, self.ground_truth = self.load_fea_result(fea_path)
            self.kdtree = self.build_kdtree(self.nodes)
        else:
            self.num_data = 1
            self.nodes = []
            self.ground_truth = []
            self.kdtree = None

    def build_kdtree(self, nodes):
        kdtrees = []
        for node in nodes:
            kdtrees.append(cKDTree(node))
        return kdtrees

    def geometry(self, nx, ny):
        x, y = np.meshgrid(np.linspace(self.x_min, self.x_max, nx), np.linspace(self.y_min, self.y_max, ny))
        x = x.flatten()
        y = y.flatten()
        mask = self.geo_filter(x, y)
        return x[mask], y[mask]

    def geometry_random_sample(self, n, n_boundary):
        if n_boundary > 0:
            n = n + n_boundary
            x_boundary, y_boundary = self.sample_boundary_points(n_boundary)
            x = np.concatenate([np.zeros(1), x_boundary])
            y = np.concatenate([np.zeros(1), y_boundary])
        else:
            x = np.zeros(1)
            y = np.zeros(1)

        while True:
            x_ = np.random.uniform(self.x_min, self.x_max, int(n * 1.2))
            y_ = np.random.uniform(self.y_min, self.y_max, int(n * 1.2))
            mask = self.geo_filter(x_, y_)

            x = np.concatenate([x, x_[mask]])
            y = np.concatenate([y, y_[mask]])

            if x.shape[0] >= (n + 1):
                break

        return x[1:n + 1], y[1:n + 1]

    def get_boundary_conditions(self, x, y, index):
        bc = np.zeros([x.shape[0], self.d_bc + 1])
        for i in range(x.shape[0]):
            x_idx = x[i]
            y_idx = y[i]
            bc[i] = self.bc_filter(x_idx, y_idx, index)
        return bc

    def get_num_data(self):
        return self.num_data

    def get_original_gird(self, index):
        if index >= self.num_data:
            raise ValueError("The index is out of range!")
        return self.nodes[index][:, 0], self.nodes[index][:, 1]

    def get_original_gird_size(self, index):
        if index >= self.num_data:
            raise ValueError("The index is out of range!")
        return self.nodes[index].shape[0]

    @abstractmethod
    def load_fea_result(self, fea_path):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def get_bc_dims(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def get_ground_truth(self, x, y, index):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def sample_boundary_points(self, n_boundary):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def geo_filter(self, x, y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def bc_filter(self, x, y, index):
        raise NotImplementedError("This method should be overridden by subclasses.")
