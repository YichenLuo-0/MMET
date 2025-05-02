import torch
from torch import nn


# Patch embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size

    def forward(self, x, masks):
        # Padding the input to make the sequence length can be divided by patch size
        batch_size, max_seq_len, d_emb = x.shape
        pad_len = self.patch_size - (max_seq_len % self.patch_size)

        if 0 < pad_len < self.patch_size:
            zero_padding = torch.zeros(batch_size, pad_len, d_emb, device=x.device)
            zero_padding_masks = torch.zeros(batch_size, pad_len, 1, device=x.device)
            x = torch.cat([x, zero_padding], dim=1)
            masks = torch.cat([masks, zero_padding_masks], dim=1)

        # Reshape the input to patches
        x = x.reshape(batch_size, -1, d_emb * self.patch_size)
        pad_seq_len = x.shape[1]

        # Generate masks for the patches
        masks = masks.reshape(batch_size, pad_seq_len, -1).sum(dim=-1) != 0
        masks = masks.unsqueeze(-1).to(torch.bool)
        return x, masks
