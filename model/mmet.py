import torch
from torch import nn

from model.activation_func import WaveAct
from model.embedding import GCE
from model.patching import PatchEmbedding
from model.serialization import serialization, sort_tensor


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        nn.init.zeros_(module.in_proj_bias)
        nn.init.xavier_uniform_(module.out_proj.weight)
        nn.init.zeros_(module.out_proj.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(self, d_input, d_output, d_ff=256):
        super(MLP, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(d_input, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_output)
        )

    def forward(self, x):
        return self.sequential(x)


# You can use the following Attention class instead of the torch.nn.MultiheadAttention to reduce the memory usage
class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(LinearAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads

        # Projections for query, key, value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Regularization
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, q_padding_mask=None, key_padding_mask=None):
        batch_size, seq_len_q, dims = q.size()
        _, seq_len_kv, _ = k.size()

        # Apply query padding mask
        if q_padding_mask is not None:
            mask = q_padding_mask.repeat(1, 1, dims)
            q = (q * mask) + (~mask * -1e9)

        # Apply key padding mask
        if key_padding_mask is not None:
            mask = key_padding_mask.repeat(1, 1, dims)
            k = (k * mask) - (~mask * 1e9)
            v = v * mask

        # Project and reshape for multi-head attention
        q = self.query(q).view(batch_size, seq_len_q, self.num_heads, dims // self.num_heads).transpose(1, 2)
        k = self.key(k).view(batch_size, seq_len_kv, self.num_heads, dims // self.num_heads).transpose(1, 2)
        v = self.value(v).view(batch_size, seq_len_kv, self.num_heads, dims // self.num_heads).transpose(1, 2)

        # L1 Attention computation
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)

        k_cumsum = k.sum(dim=-2, keepdim=True)
        a_t = 1.0 / (q * k_cumsum).sum(dim=-1, keepdim=True)

        # Compute context and output
        context = k.transpose(-2, -1) @ v
        out = self.attn_drop((q @ context) * a_t + q)
        out = out.transpose(1, 2).reshape(batch_size, seq_len_q, dims)
        out = self.proj(out)
        return out * q_padding_mask


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff = MLP(d_model, d_model)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x2 = self.layer_norm_1(x)
        x = x + self.attn(x2, x2, x2, key_padding_mask=~mask.squeeze(-1))[0]

        x2 = self.layer_norm_2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff = MLP(d_model, d_model)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out, mask=None):
        x2 = self.layer_norm_1(x)
        x = x + self.attn(x2, encoder_out, encoder_out, key_padding_mask=~mask.squeeze(-1))[0]

        x2 = self.layer_norm_2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(Encoder, self).__init__()
        self.num_layers = num_layers

        # Stack multiple encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for i in range(self.num_layers):
            # Pass through each encoder layer
            x = self.layers[i](x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(Decoder, self).__init__()
        self.num_layers = num_layers

        # Stack multiple decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, encoder_out, mask_kv):
        for i in range(self.num_layers):
            # Pass through the decoder layer
            x = self.layers[i](x, encoder_out, mask_kv)
        return x


class MMET(nn.Module):
    def __init__(
            self,
            d_input='2d',
            d_input_condition=None,
            d_output=1,
            d_embed=32,
            d_model=128,
            patch_size=4,
            depth=16,
            num_encoder=2,
            num_decoder=2,
            num_heads=2
    ):
        """
        :param d_input:             The input coordinator dimension, '2d' or '3d'.
        :param d_input_condition:   The input conditions dimensions, where the first element is the global condition.
                                    e.g. for [1, 3], the first 1 is the global condition with dimension 1,
                                    and the second 3 is the local condition with dimension 3. Therefore, the dimension
                                    of the input condition should be [batch_size, max_seq_len, 4].
        :param d_output:            The output dimension.
        :param d_embed:             The embedding layer dimension.
        :param d_model:             The encoder and decoder layer dimension.
        :param patch_size:          The patch size for the patch embedding layer.
        :param depth:               The depth of the Hilbert curve.
        :param num_encoder:         The number of encoder layers.
        :param num_decoder:         The number of decoder layers.
        :param num_heads:           The number of heads in the multi-head attention layer.
        """

        super(MMET, self).__init__()
        self.d_input = d_input
        self.depth = depth

        if d_input == '2d':
            d_input = 2
        elif d_input == '3d':
            d_input = 3
        else:
            raise ValueError("The input dimension is not supported!")

        self.embedding = GCE(d_input_condition, d_embed)
        self.positional_encoding = MLP(d_input, d_embed, d_embed)
        self.patching = PatchEmbedding(patch_size)

        self.mlp_mesh = nn.Linear(d_embed * patch_size, d_model)
        self.mlp_query = MLP(d_input, d_model, d_model)
        self.mlp_out = MLP(d_model, d_output, d_model)

        # Encoder and decoder Transformer layers
        self.encoder = Encoder(d_model, num_encoder, num_heads)
        self.decoder = Decoder(d_model, num_decoder, num_heads)

    def forward(self, coords_mesh, conditions_mesh, type_mesh, coords_query, mask_mesh=None, mask_query=None):
        """
        :param coords_mesh:     Mesh coordinates, of shape [batch_size, max_seq_len, d_input].
        :param conditions_mesh: Global and local conditions, of shape [batch_size, max_seq_len, d_input_condition].
        :param type_mesh:       Condition type, of shape [batch_size, max_seq_len, 1].
        :param coords_query:    Query coordinates, of shape [batch_size, max_seq_len, d_input].
        :param mask_mesh:       Mask for the mesh coordinates, of shape [batch_size, max_seq_len, 1].
        :param mask_query:      Mask for the query coordinates, of shape [batch_size, max_seq_len, 1].
        :return:             The output of the model, of shape [batch_size, max_seq_len, d_output].
        """

        # If the mask is not provided, set it to all ones
        if mask_mesh is None:
            mask_mesh = torch.ones_like(coords_mesh, device=coords_mesh.device)[..., 0:1]
        if mask_query is None:
            mask_query = torch.ones_like(coords_query, device=coords_query.device)[..., 0:1]
        batch_size, max_seq_len, _ = coords_mesh.size()

        # Embedding and positional encoding, and encode the boundary conditions
        emb = self.embedding(conditions_mesh, type_mesh)
        pe = self.positional_encoding(coords_mesh)
        mesh_emb = emb + pe

        # Sort the input tensor based on the code
        indices, _ = serialization(coords_mesh, mask_mesh, depth=self.depth, dims=self.d_input)
        mesh_emb = sort_tensor(mesh_emb, indices)

        # Patching the input
        query_patched, mask_patched = self.patching(mesh_emb, mask_mesh)

        # Pass through the encoder layers
        encoder_in = self.mlp_mesh(query_patched)
        encoder_out = self.encoder(encoder_in, mask_patched) * mask_patched

        # Pass through the decoder layers
        decoder_in = self.mlp_query(coords_query)
        decoder_out = self.decoder(decoder_in, encoder_out, mask_patched)

        # Output mlp layers
        out = self.mlp_out(decoder_out)
        return out * mask_query
