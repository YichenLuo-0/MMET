import torch
import torch.nn as nn

from model.activation_func import WaveAct


# Implementation of Gated Condition Embedding (GCE) Layer
class GCE(nn.Module):
    def __init__(self, bc_dims, d_hidden=256, d_out=256):
        super(GCE, self).__init__()
        self.bc_dims = bc_dims
        self.local_dims_num = len(bc_dims) - 1
        self.begin_index = []
        self.end_index = []
        for i in range(len(bc_dims)):
            self.begin_index.append(sum(bc_dims[:i]))
            self.end_index.append(sum(bc_dims[:i + 1]))

        # Dimension expansion layer
        self.liners = nn.ModuleList([nn.Linear(bc_dim, bc_dim * 2) for bc_dim in bc_dims[1:]])

        # Learnable embedding layer
        d_input = sum(bc_dims) * 2 - bc_dims[0]
        self.learnable_emb = nn.Sequential(*[
            nn.Linear(d_input, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out),
        ])

    def forward(self, condition, type):
        # Encode boundary conditions, expand dimensions
        global_encode = condition[:, :, 0:self.end_index[0]]
        local_encode = []

        for i, liner in enumerate(self.liners):
            # Mask the input features
            i_ = i + 1
            mask = (i_ == type)

            # Only for the i-th boundary condition will be expanded, others will be masked
            local_encode_i = liner(condition[:, :, self.begin_index[i_]:self.end_index[i_]])
            local_encode_i = local_encode_i * mask
            local_encode.append(local_encode_i)

        # Concatenate all the input features
        if self.local_dims_num >= 1:
            local_encode = torch.cat(local_encode, dim=-1)
            encode = torch.cat([global_encode, local_encode], dim=-1)
        else:
            encode = global_encode
        return self.learnable_emb(encode)
