import pickle

import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CarDataset(Dataset):
    def __init__(self, data_path, point_path):
        super(CarDataset, self).__init__()

        points_mesh, bcs_mesh, types_mesh, points_query, gts_query = self.load_dataset_car(data_path, point_path)

        self.points_mesh = points_mesh
        self.bcs_mesh = bcs_mesh
        self.types_mesh = types_mesh
        self.points_query = points_query
        self.gts_query = gts_query

        self.mesh_mask = torch.ones_like(self.points_mesh, device=device, dtype=torch.bool)[..., 0:1]
        self.query_mask = torch.ones_like(self.points_query, device=device, dtype=torch.bool)[..., 0:1]

    def load_dataset_car(self, data_path, point_path):
        with open(data_path, 'rb') as file:
            datas = pickle.load(file)
        with open(point_path, 'rb') as file:
            points = pickle.load(file)

        datas = torch.stack(datas, dim=0).squeeze(1)
        points = torch.stack(points, dim=0).squeeze(1)

        point_mesh = points[:, :, 0:3]
        bcs_mesh = points[:, :, 3:7]
        types_mesh = torch.zeros_like(datas, dtype=torch.int32)
        points_query = point_mesh
        gts_query = datas

        point_mesh = point_mesh.to(device)
        bcs_mesh = bcs_mesh.to(device)
        types_mesh = types_mesh.to(device)
        points_query = points_query.to(device)
        gts_query = gts_query.to(device)

        return point_mesh, bcs_mesh, types_mesh, points_query, gts_query

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
