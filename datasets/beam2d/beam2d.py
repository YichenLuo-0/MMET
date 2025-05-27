import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.beam2d.data_generation import Beam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Beam2dDataset(Dataset):
    def __init__(self, fea_path, train=True):
        super(Beam2dDataset, self).__init__()

        if train:
            self.num_data = 1000
            self.num_query = 1000
        else:
            self.num_data = 100
            self.num_query = 5000

        beam = Beam(num_data=self.num_data, e=201, nu=0.3, l=10.0, h=1.0, fea_path=fea_path)
        points_mesh, bcs_mesh, types_mesh, points_query, gts_query = self.load_dataset_beam(beam)
        self.points_mesh = points_mesh
        self.bcs_mesh = bcs_mesh
        self.types_mesh = types_mesh
        self.points_query = points_query
        self.gts_query = gts_query

        self.mesh_mask = torch.ones_like(self.points_mesh, device=device, dtype=torch.bool)[..., 0:1]
        self.query_mask = torch.ones_like(self.points_query, device=device, dtype=torch.bool)[..., 0:1]

    def load_dataset_beam(self, beam):

        points_mesh = []
        bcs_mesh = []
        points_query = []
        gts_query = []

        for i in range(self.num_data):
            x_mesh, y_mesh = beam.get_fea_mesh(0)
            bc_mesh = beam.get_boundary_conditions(x_mesh, y_mesh, i)
            x_query, y_query = beam.get_points_random(self.num_query)
            gt_query = beam.get_ground_truth(x_query, y_query, i)

            point_mesh = torch.tensor(np.stack([x_mesh, y_mesh], axis=-1), dtype=torch.float32).unsqueeze(0)
            bc_mesh = torch.tensor(bc_mesh, dtype=torch.float32).unsqueeze(0)
            point_query = torch.tensor(np.stack([x_query, y_query], axis=-1), dtype=torch.float32).unsqueeze(0)
            gt_query = torch.tensor(gt_query, dtype=torch.float32).unsqueeze(0)

            points_mesh.append(point_mesh)
            bcs_mesh.append(bc_mesh)
            points_query.append(point_query)
            gts_query.append(gt_query)

        points_mesh = torch.cat(points_mesh, dim=0).to(device)
        bcs_mesh = torch.cat(bcs_mesh, dim=0).to(device)
        types_mesh = bcs_mesh[:, :, 0:1]
        points_query = torch.cat(points_query, dim=0).to(device)
        gts_query = torch.cat(gts_query, dim=0).to(device)

        return points_mesh, bcs_mesh, types_mesh, points_query, gts_query

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


if __name__ == "__main__":
    dataset = Beam2dDataset(fea_path="beam2d.rst", train=True)
    print("Number of data points in the dataset:", len(dataset))
    for i in range(5):
        sample = dataset[i]

        print(sample[1])
        print(sample[2])

        print(f"Sample {i}:")
        print("Mesh Points:", sample[0].shape)
        print("Boundary Conditions:", sample[1].shape)
        print("Types Mesh:", sample[2].shape)
        print("Query Points:", sample[3].shape)
        print("Ground Truths:", sample[4].shape)
        print("Mesh Mask:", sample[5].shape)
        print("Query Mask:", sample[6].shape)
        print("Sequence Length Mesh:", sample[7])
        print("Sequence Length Query:", sample[8])
        print()
