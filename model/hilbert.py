import torch


def get_bit(x, i):
    x_bit = (x // (2 ** i)) % 2
    return x_bit.to(torch.bool)


def get_bits(x, depth):
    dims = x.shape[-1]
    x_bit = torch.zeros([dims, depth], dtype=torch.bool).to(x.device)
    for i in range(depth):
        x_bit[:, i] = (x // (2 ** i)) % 2
    return x_bit


def index(x_bit, y_bit, z_bit):
    y_bit = x_bit != y_bit
    z_bit = y_bit != z_bit
    return (x_bit << 2) + (y_bit << 1) + z_bit


def rotate(x_rot, y_rot, z_rot, filter):
    dims, depth = x_rot.shape
    filter = filter.unsqueeze(-1).expand(-1, depth)

    x = (x_rot & filter)
    y = (y_rot & filter)
    z = (z_rot & filter)
    return x, y, z


def encode(x, y, z, depth=16):
    batch_size, max_seq_len, _ = x.shape

    x = x.reshape(-1).to(torch.int64)
    y = y.reshape(-1).to(torch.int64)
    z = z.reshape(-1).to(torch.int64)

    x_bits = get_bits(x, depth)
    y_bits = get_bits(y, depth)
    z_bits = get_bits(z, depth)

    index_ = torch.zeros_like(x, dtype=torch.int64).to(x.device)

    for i in range(depth - 1, -1, -1):
        index_bit = index(x_bits[:, i], y_bits[:, i], z_bits[:, i])
        index_ += index_bit * (8 ** i)

        if i != 0:
            x_next = x_bits[:, 0:i]
            y_next = y_bits[:, 0:i]
            z_next = z_bits[:, 0:i]

            filter_0 = (index_bit == 0)
            filter_1 = (index_bit == 1)
            filter_2 = (index_bit == 2)
            filter_3 = (index_bit == 3)
            filter_4 = (index_bit == 4)
            filter_5 = (index_bit == 5)
            filter_6 = (index_bit == 6)
            filter_7 = (index_bit == 7)

            x_0, y_0, z_0 = rotate(z_next, x_next, y_next, filter_0)
            x_1, y_1, z_1 = rotate(y_next, x_next, z_next, filter_1)
            x_2, y_2, z_2 = rotate(x_next, y_next, z_next, filter_2)
            x_3, y_3, z_3 = rotate(~z_next, y_next, ~x_next, filter_3)
            x_4, y_4, z_4 = rotate(z_next, y_next, x_next, filter_4)
            x_5, y_5, z_5 = rotate(x_next, y_next, z_next, filter_5)
            x_6, y_6, z_6 = rotate(~y_next, ~x_next, z_next, filter_6)
            x_7, y_7, z_7 = rotate(~z_next, ~x_next, y_next, filter_7)

            x_bits[:, 0:i] = x_0 | x_1 | x_2 | x_3 | x_4 | x_5 | x_6 | x_7
            y_bits[:, 0:i] = y_0 | y_1 | y_2 | y_3 | y_4 | y_5 | y_6 | y_7
            z_bits[:, 0:i] = z_0 | z_1 | z_2 | z_3 | z_4 | z_5 | z_6 | z_7

    index_ = index_.reshape(batch_size, max_seq_len)
    return index_
