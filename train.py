import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import get_dataset
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
    le_errors = errors_l2 / (targets_l2 + 1e-8)
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


def train(args, dataloader_train, dataloader_test, model, optimizer, loss_func, epochs):
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

            # Train the model
            optimizer.zero_grad()
            pred = model(points_mesh, bcs_mesh, types_mesh, points_query, mesh_mask, query_mask)
            loss = loss_with_mask(pred, gts_query, query_mask, loss_func)
            loss.backward()
            optimizer.step()
            print("\rEpoch: {}, Iteration: {}, Loss: {:.4f}".format(epoch, i, loss.cpu().detach().numpy()), end='',
                  flush=True)

        # Evaluate the model
        print()
        print("--------------------------------------------")
        err_test = evaluate(model, dataloader_test)
        err_test = torch.sum(err_test)
        print("Test L2 Error: {:.4f}".format(err_test.cpu().detach().numpy()))
        if err_test < err_min:
            err_min = err_test
            torch.save(model, "model" + args.dataset + ".pth")
            print("Save model")
        print("--------------------------------------------")
        print()


def main():
    parser = argparse.ArgumentParser(description="Train the MMET model on the heat2d dataset")
    # Add dataset parameters
    parser.add_argument("--dataset", type=str, default="Darcy Flow", help="Dataset name")
    # Add training parameters
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    # Add model parameters
    parser.add_argument("--d_input", type=str, default="2d", help="Input dimension (2d or 3d)")
    parser.add_argument("--d_input_condition", type=int, nargs='+', default=[1], help="Input condition dimensions")
    parser.add_argument("--d_output", type=int, default=1, help="Output dimension")
    parser.add_argument("--d_embed", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size for the model")
    parser.add_argument("--depth", type=int, default=16, help="Depth of the model")
    parser.add_argument("--num_encoder", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--num_decoder", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    args = parser.parse_args()

    # Training parameters
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # Initialize the dataset
    dataset_train, dataset_test = get_dataset(args.dataset)

    # Initialize the network and optimizer
    model = MMET(
        d_input=args.d_input,
        d_input_condition=args.d_input_condition,
        d_output=args.d_output,
        d_embed=args.d_embed,
        d_model=args.d_model,
        patch_size=args.patch_size,
        depth=args.depth,
        num_encoder=args.num_encoder,
        num_decoder=args.num_decoder,
        num_heads=args.num_heads
    ).to(device)
    model = nn.DataParallel(model)

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=dataset_test.__len__(), shuffle=False)

    # Train the model
    train(args, dataloader_train, dataloader_test, model, optimizer, loss_func, epochs)


if __name__ == "__main__":
    main()
