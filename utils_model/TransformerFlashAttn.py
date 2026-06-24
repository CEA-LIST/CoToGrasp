import math
import torch
import torch.nn as nn
import torch.utils.checkpoint

class Transformer(nn.Module):
    def __init__(self, in_channels=512, n_blocks=4, n_heads=8, dropout=0.0):
        super(Transformer, self).__init__()
        
        self.pos_encoder = PositionalEncoding(in_channels, max_len=150000)
        
        self.blocks = nn.ModuleList([
            Block(in_channels, n_heads, dropout=dropout)
            for i in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            # x = block(x)
        return self.final_norm(x)

class Block(nn.Module):
    def __init__(self, in_channels, n_heads, mlp_ratio=4.0, dropout=0.0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(in_channels)
        self.attn = MultiHeadedSelfAttention(n_heads, in_channels, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, int(mlp_ratio * in_channels)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * in_channels), in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        return x + self.mlp(self.norm2(x))

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=None):
        """
        :param n_heads: number of attention heads
        :param dim: model dimension
        :param dropout: dropout rate
        """
        super(MultiHeadedSelfAttention, self).__init__()
        import xformers.ops as xops
        self.xops = xops

        assert dim % n_heads == 0
        # We assume d_q == d_k == d_v == d_model / h
        self.dim_qkv = dim // n_heads
        self.n_heads = n_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.attn = None
        self.dropout = nn.Dropout(dropout) if dropout is not None else lambda x: x

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = x.size(0)

        qkv = self.to_qkv(x)                                                # (nbatches, N, 3 * dim)
        qkv = qkv.view(nbatches, -1, 3, self.n_heads, self.dim_qkv)         # split into 3, then into heads: (nbatches, N, 3, n_heads, dim_qkv)
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]        # (nbatches, N, n_heads, dim_qkv)

        z = self.xops.memory_efficient_attention(
            query, 
            key, 
            value, 
            attn_bias=None, 
            p=self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0,
            scale=1.0 / self.dim_qkv**0.5
        )
        x = z.contiguous().view(nbatches, -1, self.n_heads * self.dim_qkv)
        return self.out(x)

class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, max_len=150000):
        super().__init__()
        # Cache PE to avoid recomputing. Compute in float32.
        pe = torch.zeros(max_len, in_channels, dtype=torch.float32) 
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_channels, 2, dtype=torch.float32) * (-math.log(10000.0) / in_channels))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (state_dict yes, optimizer no)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, in_channels)

    def forward(self, x):
        # Slice the cached PE to the current input length
        return x + self.pe[:, :x.size(1), :].to(dtype=x.dtype)


##########################################################################
# Pooling using Set Transformer's Induced Set Attention Block
##########################################################################

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout=0.0):
        super(MAB, self).__init__()
        import xformers.ops as xops
        self.xops = xops
        self.dim_V = dim_V
        self.num_heads = num_heads
        # Ensure dim_V can be evenly split across heads
        assert dim_V % num_heads == 0, f"dim_V ({dim_V}) must be divisible by num_heads ({num_heads})"
        self.dim_head = dim_V // num_heads
        
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        
        self.norm_q = nn.LayerNorm(dim_Q)
        self.norm_k = nn.LayerNorm(dim_K)
        self.norm_ff = nn.LayerNorm(dim_V)
        
        self.dropout = nn.Dropout(dropout)
            
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.ffn = nn.Sequential(
            nn.Linear(dim_V, 4 * dim_V),
            nn.GELU(),
            nn.Linear(4 * dim_V, dim_V),
            nn.Dropout(dropout)
        )

    def forward(self, Q, K):
        B, N, C = Q.shape
        
        q = self.fc_q(self.norm_q(Q)).reshape(B, -1, self.num_heads, self.dim_head)
        k = self.fc_k(self.norm_k(K)).reshape(B, -1, self.num_heads, self.dim_head)
        v = self.fc_v(self.norm_k(K)).reshape(B, -1, self.num_heads, self.dim_head)

        attn_out = self.xops.memory_efficient_attention(
            q, k, v, scale=1.0 / (q.shape[-1] ** 0.5)
        )
        attn_out = attn_out.reshape(B, -1, self.dim_V)
        
        # If dim_Q == dim_V, we add to original Q. Else we take just attention output.
        out = Q + self.dropout(self.fc_o(attn_out)) if C == self.dim_V else self.dropout(self.fc_o(attn_out))

        # Feed Forward + Residual 2 (Pre-Norm)
        out = out + self.ffn(self.norm_ff(out))
        
        return out

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, dropout=0.0):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, dropout=dropout)
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class PoolingAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_seeds, num_heads=8, dropout=0.0):
        super(PoolingAttention, self).__init__()
        self.pma = PMA(dim_in, num_heads, num_seeds, dropout=dropout)
        self.sab = MAB(dim_in, dim_in, dim_out, num_heads, dropout=dropout)

    def forward(self, X):
        """ 
        :param X: (B, N, D) input features
        :return: (B, num_seeds, D) pooled features
        """
        X = self.pma(X)
        X = self.sab(X, X)
        if X.shape[1] == 1:
            return X.squeeze(1)  # (B, D)
        return X
