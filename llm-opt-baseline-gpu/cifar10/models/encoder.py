# 

import math
import torch
from torch import nn, Tensor
from typing import Optional
import copy

# Positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Self-attention mechanism
# 
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be a multiple of nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Define linear projection layers
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_length, _ = x.size()

        # Apply linear projections and split into heads
        Q = self.q_linear(x).view(batch_size, seq_length, self.nhead, self.head_dim).transpose(1, 2)  # [batch, nhead, seq, head_dim]
        K = self.k_linear(x).view(batch_size, seq_length, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_length, self.nhead, self.head_dim).transpose(1, 2)

        # Check Q, K, and V for NaNs
        assert not torch.isnan(Q).any(), "NaN detected in Q"
        assert not torch.isnan(K).any(), "NaN detected in K"
        assert not torch.isnan(V).any(), "NaN detected in V"

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, nhead, seq, seq]
        assert not torch.isnan(scores).any(), "NaN detected in scores"

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float('-inf'))
        #     # Check scores for NaNs after applying the mask
        #     assert not torch.isnan(scores).any(), "NaN detected in scores after mask"
        # if mask is not None:
        #     # Use a very small negative value instead of -inf
        
        #     scores = scores.masked_fill(mask == 0, -1e9)
        #     assert not torch.isnan(scores).any(), "NaN detected in scores after mask"
        if mask is not None:
            scores = scores + mask  # This is an additive mask
        attn = torch.softmax(scores, dim=-1)
        assert not torch.isnan(attn).any(), "NaN detected in attention probabilities"
        attn = self.dropout(attn)
        assert not torch.isnan(attn).any(), "NaN detected in attention after dropout"

        # Compute context vectors
        context = torch.matmul(attn, V)  # [batch, nhead, seq, head_dim]
        assert not torch.isnan(context).any(), "NaN detected in context"

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)  # [batch, seq, d_model]
        assert not torch.isnan(context).any(), "NaN detected in context after transpose and reshape"

        output = self.out_linear(context)
        assert not torch.isnan(output).any(), "NaN detected in out_linear output"

        return output

# Feed-forward network
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Transformer encoder layer
class TransformerEncoderLayerSimple(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super(TransformerEncoderLayerSimple, self).__init__()
        self.self_attn = SelfAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Self-attention sublayer
        attn_output = self.self_attn(src, mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        assert not torch.isnan(src).any(), "NaN detected after self-attention and norm1"

        # Feed-forward sublayer
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        assert not torch.isnan(src).any(), "NaN detected after feed-forward and norm2"
        return src

# Transformer encoder
class TransformerEncoderSimple(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super(TransformerEncoderSimple, self).__init__()
        encoder_layer = TransformerEncoderLayerSimple(d_model, nhead, d_ff, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            src = layer(src, mask)
        return src

# Final Transformer model
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, device: str = 'cpu'):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.device = device
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = TransformerEncoderSimple(d_model, nhead, d_hid, nlayers, dropout)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src: Tensor with shape [batch, seq_len].
            src_mask: Tensor with shape [seq_len, seq_len].

        Returns:
            Output tensor with shape [batch, seq_len, ntoken].
        """
        src = self.embedding(src) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(src.size(1)).to(self.device)
        output = self.transformer_encoder(src, src_mask)  # [batch, seq_len, d_model]
        output = self.decoder(output)  # [batch, seq_len, ntoken]
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == '__main__':
    ntokens = 10000
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    linear_attention = False
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    model_trm = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers,dropout, DEVICE).to(DEVICE)
    print(model_trm)

    # torch.save(model.state_dict(),'../model_weights/init_trm_ptb.pth')

    inputs = torch.zeros(20, 35, dtype=torch.long).to(DEVICE)

    model_trm(inputs)
