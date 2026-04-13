"""
Build & Learn #1 — Self-Attention & Transformer Encoder
--------------------------------------------------------
Concept: How does a transformer encoder actually work?
Build:   Minimal self-attention + encoder block from scratch in PyTorch

No HuggingFace. No shortcuts. Just the raw mechanism.

Substack post: https://dsunpacked.substack.com
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# 1. SELF-ATTENTION
# ─────────────────────────────────────────────
#
# Every token looks at every other token and decides what's relevant.
#
# Three vectors per token:
#   Q (Query)  — "What am I looking for?"
#   K (Key)    — "What do I offer?"
#   V (Value)  — "What do I actually pass forward?"
#
# Score = softmax(QKᵀ / √d_k) × V

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # each head works on a slice of the embedding

        # One linear layer produces Q, K, V all at once — then we split
        # Cleaner than 3 separate layers, same result
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dim

        # Project input into Q, K, V and split into 3 tensors
        qkv = self.qkv(x).chunk(3, dim=-1)

        # Reshape for multi-head: each head gets its own slice
        # transpose so shape is (B, num_heads, T, head_dim)
        Q, K, V = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        ]

        # Attention scores: how much should each token attend to every other?
        # Scale by √d_k to prevent scores from exploding in high dimensions
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        attn = scores.softmax(dim=-1)  # normalize across the sequence

        # Weighted sum of values
        out = attn @ V  # (B, heads, T, head_dim)

        # Collapse heads back into one representation
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)  # final linear projection


# ─────────────────────────────────────────────
# 2. POSITIONAL ENCODING
# ─────────────────────────────────────────────
#
# Since attention is order-agnostic, we need to inject position info.
# We use sine and cosine waves at different frequencies — one pattern
# per position, unique enough that the model can learn to read them.

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super().__init__()

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices → sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices → cosine

        # Register as buffer so it moves with the model (cpu/gpu) but isn't a parameter
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_len, embed_dim)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        return x + self.pe[:, :x.size(1)]


# ─────────────────────────────────────────────
# 3. FEED-FORWARD NETWORK
# ─────────────────────────────────────────────
#
# After attention, each token independently goes through a small 2-layer MLP.
# This is where the model processes what it gathered from attention.
# The inner dimension is typically 4x the embedding dim (from the paper).

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 4. ENCODER BLOCK
# ─────────────────────────────────────────────
#
# One full encoder layer:
#   → Multi-head self-attention
#   → Add & Norm (residual connection + layer norm)
#   → Feed-forward network
#   → Add & Norm again
#
# The residual connections (x + ...) are crucial — they let gradients
# flow cleanly during training and stabilize deep networks.

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention + residual
        x = self.norm1(x + self.dropout(self.attention(x)))
        # Feed-forward + residual
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# ─────────────────────────────────────────────
# 5. FULL ENCODER
# ─────────────────────────────────────────────
#
# Stack N encoder blocks. The paper uses N=6.
# Each block refines the token representations further.

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, T) — token indices
        x = self.dropout(self.pos_encoding(self.embedding(x)))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)  # (B, T, embed_dim)


# ─────────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Config
    VOCAB_SIZE   = 1000
    EMBED_DIM    = 64
    NUM_HEADS    = 8
    FF_DIM       = 256   # 4x embed_dim, as per the paper
    NUM_LAYERS   = 2
    BATCH_SIZE   = 2
    SEQ_LEN      = 10

    # Build model
    model = TransformerEncoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS
    )

    # Dummy input — random token indices
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    print(f"Input shape:  {x.shape}")

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    # Expected: (2, 10, 64) — batch, sequence, embedding

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    print("\nForward pass successful.")
    print("Each token now carries context from every other token in the sequence.")
