import torch

from model.hilbert import encode


def normalize(x, depth=16):
    # The input x is of shape [..., sequence, 1], where the last dimension is the coordinate
    sequence_dim = x.dim() - 2

    # Find the min and max value of x
    min_val = torch.min(x, dim=sequence_dim).values
    max_val = torch.max(x, dim=sequence_dim).values

    # Expand the min_val and max_val to the same shape as x
    max_val = max_val.unsqueeze(-1).expand(*max_val.shape[:-1], x.shape[sequence_dim], max_val.shape[-1])
    min_val = min_val.unsqueeze(-1).expand(*min_val.shape[:-1], x.shape[sequence_dim], min_val.shape[-1])

    # Transform the coordinates to [0, 2^depth-1]
    normalized = (x - min_val) / (max_val - min_val) * (2 ** depth - 1)
    return normalized.to(torch.int64)


def serialization(coords, mask, depth=16, dims='3d'):
    batch_size, _, _ = coords.shape

    x, y = coords[..., 0:1], coords[..., 1:2]
    xx = normalize(x, depth=depth)
    yy = normalize(y, depth=depth)

    if dims == '3d':
        z = coords[..., 2:3]
        zz = normalize(z, depth=depth)
    elif dims == '2d':
        zz = torch.zeros_like(xx)
    else:
        raise ValueError('dims must be either 2d or 3d')

    # Encode the coordinates using Hilbert curve
    code = encode(xx, yy, zz, depth=depth).reshape(batch_size, -1)

    # Pad the code with a value that is larger than all values in the Hilbert code,
    # keeping the padding part still at the end of the sequence after sorting.
    mesh_seq_pad = torch.ones_like(code, device=code.device) * (torch.max(code) + 1)
    mask = mask.squeeze(-1)
    code = (code * mask) + (mesh_seq_pad * (~mask))

    # Sort the input tensor based on the code
    _, indices = torch.sort(code, dim=1)
    return indices, code.unsqueeze(-1)


def sort_tensor(x, indices):
    _, _, d_out = x.shape
    indices_out = indices.unsqueeze(-1).expand(*indices.shape, d_out)

    # Resort the output tensor based on the indices
    return torch.gather(x, 1, indices_out)


def resort_tensor(x, indices):
    _, _, d_out = x.shape
    indices_out = indices.unsqueeze(-1).expand(*indices.shape, d_out)

    # Resort the output tensor based on the indices
    return x.scatter(1, indices_out, x)
