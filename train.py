import argparse

import torch
from torch import nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader

from datasets import Heat2dDataset
from datasets.darcy_flow.darcy_flow import DarcyDataset
from model import MMET

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


def evaluate(model, dataloader_test):
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

    return err_test


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

            # def closure():
            #     nonlocal iter_num, loss_total
            #     nonlocal points_mesh, bcs_mesh, types_mesh, points_query, gts_query, mesh_mask, query_mask
            #
            #     # Set the model to training mode
            #     optimizer.zero_grad()
            #     pred = model(points_mesh, bcs_mesh, types_mesh, points_query, mesh_mask, query_mask)
            #     loss = loss_with_mask(pred, gts_query, query_mask, loss_func)
            #     loss.backward()
            #
            #     iter_num += 1
            #     loss_total += loss.cpu().detach().numpy()
            #     return loss
            #
            # optimizer.step(closure)
            # print("Epoch: {}, Iteration: {}, Loss: {:.4f}".format(epoch, i, loss_total / iter_num))

            optimizer.zero_grad()
            pred = model(points_mesh, bcs_mesh, types_mesh, points_query, mesh_mask, query_mask)
            loss = loss_with_mask(pred, gts_query, query_mask, loss_func)
            loss.backward()
            optimizer.step()
            print("Epoch: {}, Iteration: {}, Loss: {:.4f}".format(epoch, i, loss.cpu().detach().numpy()))

        # Evaluate the model
        print("--------------------------------------------")
        err_test = evaluate(model, dataloader_test)
        print("Test L2 Error: {:.4f}".format(err_test[0].cpu().detach().numpy()))
        if err_test < err_min:
            err_min = err_test
            torch.save(model, "datasets/darcy_flow/model.pth")
            print("Save model")
        print("--------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Train the MMET model on the heat2d dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    args = parser.parse_args()

    # Training parameters
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # Initialize the elastic body
    # train_path = "datasets/heat2d/heat2d_1100_train.pkl"
    # test_path = "datasets/heat2d/heat2d_1100_test.pkl"
    # heat2d_train = Heat2dDataset(train_path)
    # heat2d_test = Heat2dDataset(test_path)

    path = "datasets/darcy_flow/2D_DarcyFlow_beta100.0_Train.hdf5"
    heat2d_train = DarcyDataset(path, train=True)
    heat2d_test = DarcyDataset(path, train=False)

    # Initialize the network and optimizer
    model = MMET(
        d_input='2d',
        d_input_condition=[1],
        d_output=1,
        d_embed=32,
        d_model=128,
        patch_size=32,
        depth=16,
        num_encoder=2,
        num_decoder=2,
        num_heads=2
    ).to(device)
    model = nn.DataParallel(model)

    # Optimizer and loss function
    # optimizer = LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    dataloader_train = DataLoader(heat2d_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(heat2d_test, batch_size=heat2d_test.__len__(), shuffle=False)

    # Train the model
    train(dataloader_train, dataloader_test, model, optimizer, loss_func, epochs)


if __name__ == "__main__":
    main()
