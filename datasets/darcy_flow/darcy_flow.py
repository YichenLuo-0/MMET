import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DarcyDataset(Dataset):
    def __init__(self, file_path, train=True):
        super(DarcyDataset, self).__init__()

        points_mesh, bcs_mesh, types_mesh, points_query, gts_query = self.load_dataset_darcy(file_path, train)

        self.points_mesh = points_mesh
        self.bcs_mesh = bcs_mesh
        self.types_mesh = types_mesh
        self.points_query = points_query
        self.gts_query = gts_query

        self.mesh_mask = torch.ones_like(self.points_mesh, device=device, dtype=torch.bool)[..., 0:1]
        self.query_mask = torch.ones_like(self.points_query, device=device, dtype=torch.bool)[..., 0:1]

    def load_dataset_darcy(self, file_path, train=True):
        with h5py.File(file_path, "r") as h5_file:
            data = np.array(h5_file["tensor"], dtype=np.float32)
            nu = np.array(h5_file["nu"], dtype=np.float32)

        torch_data = torch.tensor(data, dtype=torch.float32)
        torch_nu = torch.tensor(nu, dtype=torch.float32)

        torch_data = torch_data.squeeze(1)
        torch_data = torch_data.reshape(torch_data.shape[0], -1, 1)
        torch_nu = torch_nu.reshape(torch_nu.shape[0], -1, 1)

        # 生成128*128的网格
        point_mesh_x = torch.linspace(0, 1, 128)
        point_mesh_y = torch.linspace(0, 1, 128)
        point_mesh_x, point_mesh_y = torch.meshgrid(point_mesh_x, point_mesh_y, indexing='ij')
        point_mesh = torch.stack([point_mesh_x, point_mesh_y], dim=-1)
        point_mesh = point_mesh.reshape(1, -1, 2).expand(torch_nu.shape[0], 16384, 2)

        types_mesh = torch.zeros_like(point_mesh[:, :, 0:1], dtype=torch.int32)

        point_mesh = point_mesh.to(device)
        torch_data = torch_data.to(device)
        torch_nu = torch_nu.to(device)
        types_mesh = types_mesh.to(device)

        if train:
            return point_mesh[:9900], torch_nu[:9900], types_mesh[:9900], point_mesh[:9900], torch_data[:9900]
        else:
            return point_mesh[9900:10000], torch_nu[9900:10000], types_mesh[9900:10000], point_mesh[9900:10000], torch_data[9900:10000]

    def __len__(self):
        return len(self.points_mesh)

    def __getitem__(self, idx):
        seq_len_mesh = torch.sum(self.mesh_mask[idx]).to(torch.int32)
        seq_len_query = torch.sum(self.query_mask[idx]).to(torch.int32)

        return (
            self.points_mesh[idx],
            self.bcs_mesh[idx],
            self.types_mesh[idx],
            self.points_query[idx],
            self.gts_query[idx],
            self.mesh_mask[idx],
            self.query_mask[idx],
            seq_len_mesh,
            seq_len_query
        )
