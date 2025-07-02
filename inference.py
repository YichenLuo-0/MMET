import argparse
import torch
from torch.utils.data import DataLoader
from datasets import get_dataset
from model import MMET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_relative_error_with_mask(pred, targ, mask):
    pred = pred * mask
    targ = targ * mask
    targets_l2 = torch.norm(targ, dim=-2)
    errors_l2 = torch.norm(pred - targ, dim=-2)
    le_errors = errors_l2 / (targets_l2 + 1e-8)
    return torch.mean(le_errors, dim=0)


def inference(model, dataloader):
    model.eval()
    all_predictions = []
    all_errors = []

    with torch.no_grad():
        for data in dataloader:
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

            # dynamically adjust sequence lengths
            max_seq_len_mesh = torch.max(seq_len_mesh)
            max_seq_len_query = torch.max(seq_len_query)

            points_mesh = points_mesh[:, :max_seq_len_mesh]
            bcs_mesh = bcs_mesh[:, :max_seq_len_mesh]
            types_mesh = types_mesh[:, :max_seq_len_mesh]
            points_query = points_query[:, :max_seq_len_query]
            gts_query = gts_query[:, :max_seq_len_query]
            mesh_mask = mesh_mask[:, :max_seq_len_mesh]
            query_mask = query_mask[:, :max_seq_len_query]

            # forward pass
            pred = model(points_mesh, bcs_mesh, types_mesh, points_query, mesh_mask, query_mask)

            # calculate error
            err = l2_relative_error_with_mask(pred, gts_query, query_mask)
            total_err = torch.sum(err)

            # save results
            all_predictions.append(pred.cpu())
            all_errors.append(total_err.item())

            print(f"Batch Error: {total_err.item():.4f}")

    # merge predictions and errors
    all_predictions = torch.cat(all_predictions, dim=0)
    avg_error = sum(all_errors) / len(all_errors)
    return all_predictions, avg_error


def main():
    parser = argparse.ArgumentParser(description="Inference with MMET model")
    parser.add_argument("--dataset", type=str, default="Darcy Flow", help="Dataset name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--save_output", action="store_true", help="Save predictions to file")
    args = parser.parse_args()

    # load the dataset based on the provided dataset name
    _, dataset_test = get_dataset(args.dataset)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # load the trained model from the specified path
    model = torch.load(args.model_path, map_location=device)
    model.to(device)
    print(f"Loaded model from {args.model_path}")

    # perform inference on the test dataset
    predictions, avg_error = inference(model, dataloader_test)
    print(f"\nAverage L2 Relative Error: {avg_error:.4f}")

    # save predictions to a file if specified
    if args.save_output:
        torch.save(predictions, f"predictions_{args.dataset.replace(' ', '_')}.pt")
        print(f"Predictions saved to predictions_{args.dataset.replace(' ', '_')}.pt")


if __name__ == "__main__":
    main()
