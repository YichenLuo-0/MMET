import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Heat2dDataset(Dataset):
    def __init__(self, file_path):
        super(Heat2dDataset, self).__init__()

        points_mesh, bcs_mesh, types_mesh, points_query, gts_query = self.load_dataset_heat2d(file_path)
        self.points_mesh, self.mesh_mask = self.pad_sequences(points_mesh)
        self.bcs_mesh, _ = self.pad_sequences(bcs_mesh)
        self.types_mesh, _ = self.pad_sequences(types_mesh)
        self.points_query, self.query_mask = self.pad_sequences(points_query)
        self.gts_query, _ = self.pad_sequences(gts_query)

        self.mesh_mask = self.mesh_mask[:, :, 0:1]
        self.query_mask = self.query_mask[:, :, 0:1]

    def load_dataset_heat2d(self, file_path):
        with open(file_path, 'rb') as file:
            datas = pickle.load(file)

        points_mesh = []
        bcs_mesh = []
        types_mesh = []
        points_query = []
        gts_query = []

        for data in datas:
            condition = data[3]
            point_1 = condition[0]
            point_2 = condition[1]
            point_3 = condition[2]
            point_4 = condition[3]
            point_2 = point_2[:, 0:2]

            points_1 = np.concatenate([point_1, point_2, point_3, point_4], axis=0)
            points_2 = data[0]

            #  Filter the points that are not in the mesh
            mask = ~np.isin(points_2.view([('', points_2.dtype)] * points_2.shape[1]),
                            points_1.view([('', points_1.dtype)] * points_1.shape[1]))
            filtered_points_2 = points_2[mask.flatten()]
            points_ = np.concatenate([points_1, filtered_points_2], axis=0)

            # Generate the boundary conditions
            bc_1 = np.tile([1.0, 0.0], (point_1.shape[0], 1))
            bc_2 = np.tile([2.0, 0.0], (point_2.shape[0], 1))
            bc_3 = np.tile([3.0, 0.0], (point_3.shape[0], 1))
            bc_4 = np.tile([4.0, 0.0], (point_4.shape[0], 1))
            bc_5 = np.tile([0.0, 0.0], (filtered_points_2.shape[0], 1))
            bc_2[:, 1] = data[3][1][:, 2]
            bcs_ = np.concatenate([bc_2, bc_1, bc_3, bc_4, bc_5], axis=0)

            # Generate the boundary condition types
            types_1 = np.tile(0, (point_1.shape[0], 1))
            types_2 = np.tile(1, (point_2.shape[0], 1))
            types_3 = np.tile(0, (point_3.shape[0], 1))
            types_4 = np.tile(0, (point_4.shape[0], 1))
            types_5 = np.tile(0, (filtered_points_2.shape[0], 1))
            types_ = np.concatenate([types_2, types_1, types_3, types_4, types_5], axis=0)

            # Convert the data to tensor
            points_ = torch.tensor(points_, dtype=torch.float32).unsqueeze(0)
            bcs_ = torch.tensor(bcs_, dtype=torch.float32).unsqueeze(0)
            types_ = torch.tensor(types_, dtype=torch.float32).unsqueeze(0)
            points_q = torch.tensor(points_2, dtype=torch.float32).unsqueeze(0)
            gts_q = torch.tensor(data[1], dtype=torch.float32).unsqueeze(0)

            points_mesh.append(points_)
            bcs_mesh.append(bcs_)
            types_mesh.append(types_)
            points_query.append(points_q)
            gts_query.append(gts_q)

        return points_mesh, bcs_mesh, types_mesh, points_query, gts_query

    def pad_sequences(self, sequences, padding_value=0.0):
        # Find the maximum sequence length
        max_seq_len = max(seq.size(1) for seq in sequences)
        dims = sequences[0].size(2)

        padded_sequences = []
        masks = []

        for seq in sequences:
            seq_len = seq.size(1)
            padding_length = max_seq_len - seq_len

            # Pad the sequence with zeros
            padded_seq = torch.cat([seq, torch.full((1, padding_length, dims), padding_value)], dim=1)
            padded_sequences.append(padded_seq)

            # Generate mask with shape [1, max_seq_len, 1]
            mask = torch.cat([torch.ones(1, seq_len, 1), torch.zeros(1, padding_length, 1)], dim=1)
            masks.append(mask)

        # Stack the padded sequences and masks
        batch = torch.cat(padded_sequences, dim=0).to(device)
        batch_mask = torch.cat(masks, dim=0).to(torch.bool).to(device)

        return batch, batch_mask

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
