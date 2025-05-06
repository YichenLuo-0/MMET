import argparse
import pickle

import numpy as np
import torch
from torch import nn
from torch.optim import LBFGS, AdamW
from torch.utils.data import DataLoader, Dataset

from model.mmet import MMET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_with_mask(pred, targ, mask, loss_fn=nn.MSELoss(reduction='mean')):
    # Calculate the pointwise loss, with the shape of [batch_size, max_seq_len, dims]
    loss = loss_fn(pred, targ)

    # Mask the padding part by multiplying the loss with the mask
    loss = loss * mask

    # Calculate the effective length of each sequence, and calculate the average loss
    effective_lengths = mask.sum(dim=1)
    loss_per_sequence = loss.sum(dim=1) / effective_lengths

    # Sum over the dims axis to get the total loss of each sample
    return loss_per_sequence.sum()


def l2_relative_error_with_mask(pred, targ, mask):
    # Calculate the pointwise error, with the shape of [batch_size, max_seq_len, dims]
    pred = pred * mask
    targ = targ * mask

    # Calculate the L2 norm of the target
    targets_l2 = torch.norm(targ, dim=-2)
    errors_l2 = torch.norm(pred - targ, dim=-2)

    # Calculate the relative error
    le_errors = errors_l2 / targets_l2
    return torch.mean(le_errors, dim=0)


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


def train(dataloader_train, dataloader_test, model, optimizer, loss_func, epochs):
    err_min = 0.6
    for epoch in range(epochs):
        for i, data in enumerate(dataloader_train):
            (
                points_mesh,
                bcs_mesh,
                types_mesh,
                points_query,
                gts_query,
                mesh_mask,
                query_mask,
                seq_len_mesh,
                seq_len_query
            ) = data

            max_seq_len_mesh = torch.max(seq_len_mesh)
            max_seq_len_query = torch.max(seq_len_query)

            # Cut the sequence to the maximum length
            points_mesh = points_mesh[:, :max_seq_len_mesh]
            bcs_mesh = bcs_mesh[:, :max_seq_len_mesh]
            types_mesh = types_mesh[:, :max_seq_len_mesh]
            points_query = points_query[:, :max_seq_len_query]
            gts_query = gts_query[:, :max_seq_len_query]
            mesh_mask = mesh_mask[:, :max_seq_len_mesh]
            query_mask = query_mask[:, :max_seq_len_query]

            iter_num = 0
            loss_total = 0.0

            def closure():
                nonlocal optimizer, iter_num, loss_total

                # Set the model to training mode
                optimizer.zero_grad()
                pred = model(points_mesh, bcs_mesh, types_mesh, points_query, mesh_mask, query_mask)
                loss = loss_with_mask(pred, gts_query, query_mask, loss_func)
                loss.backward()

                iter_num += 1
                loss_total += loss.cpu().detach().numpy()
                return loss

            optimizer.step(closure)
            print("Epoch: {}, Iteration: {}, Loss: {:.4f}".format(epoch, i, loss_total / iter_num))

            # optimizer.zero_grad()
            # pred = model(points_mesh, bcs_mesh, types_mesh, points_query, mesh_mask, query_mask)
            # loss = loss_with_mask(pred, gts_query, query_mask, loss_func)
            # loss.backward()
            # optimizer.step()
            # print("Epoch: {}, Iteration: {}, Loss: {:.4f}".format(epoch, i, loss.cpu().detach().numpy()))

        # Test the model
        with torch.no_grad():
            (
                points_mesh,
                bcs_mesh,
                types_mesh,
                points_query,
                gts_query,
                mesh_mask,
                query_mask,
                seq_len_mesh,
                seq_len_query
            ) = next(iter(dataloader_test))

            pred_test = model(points_mesh, bcs_mesh, types_mesh, points_query, mesh_mask, query_mask)
            err_test = l2_relative_error_with_mask(pred_test, gts_query, query_mask)

            print("--------------------------------------------")
            print("Test L2 Error: {:.4f}".format(err_test[0].cpu().detach().numpy()))
            if err_test < err_min:
                err_min = err_test
                torch.save(model, "datasets/heat2d/model2.pth")
                print("Save model")
            print("--------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Train the MMET model on the heat2d dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate for the optimizer")
    args = parser.parse_args()

    # Training parameters
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # Initialize the elastic body
    train_path = "datasets/heat2d/heat2d_1100_train.pkl"
    test_path = "datasets/heat2d/heat2d_1100_test.pkl"
    heat2d_train = Heat2dDataset(train_path)
    heat2d_test = Heat2dDataset(test_path)

    # Initialize the network and optimizer
    model = MMET(
        d_input='2d',
        d_input_condition=[1, 1],
        d_output=1,
        d_embed=64,
        d_model=192,
        patch_size=1,
        depth=16,
        num_encoder=4,
        num_decoder=4,
        num_heads=3
    ).to(device)
    model = nn.DataParallel(model)

    # Optimizer and loss function
    optimizer = LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
    loss_func = nn.MSELoss()
    dataloader_train = DataLoader(heat2d_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(heat2d_test, batch_size=heat2d_test.__len__(), shuffle=False)

    # Train the model
    train(dataloader_train, dataloader_test, model, optimizer, loss_func, epochs)


if __name__ == "__main__":
    main()
