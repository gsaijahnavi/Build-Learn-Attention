# Build & Learn

Learning ML concepts by building them from scratch. No shortcuts, no libraries doing the heavy lifting — just code and understanding.

---

## Week 1: Self-Attention & Transformer Encoder

### What is self-attention?

Every token in a sequence looks at every other token simultaneously and decides what's relevant.

Before transformers, models like RNNs read sequences word by word — left to right. By word 40, context from word 3 was mostly gone. Self-attention fixes this by letting every token attend to every other token in one shot, in parallel.

Three vectors drive it:

- **Q (Query)** — what am I looking for?
- **K (Key)** — what do I offer?
- **V (Value)** — what do I actually pass forward?

The attention score:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

Run multiple attention heads in parallel and each one picks up a different kind of relationship — grammar, meaning, coreference. Stack a few encoder blocks and you have a transformer encoder.

---

### What this builds

A transformer encoder from scratch in PyTorch:

- `SelfAttention` — multi-head attention, step by step
- `PositionalEncoding` — sine/cosine encoding so the model knows word order
- `FeedForward` — the MLP that follows attention in each block
- `EncoderBlock` — attention + feed-forward + residual connections + layer norm
- `TransformerEncoder` — full encoder with N stacked blocks

---

### Run it

```bash
pip install -r requirements.txt
python week1_self_attention.py
```

Expected output:

```
Input shape:  torch.Size([2, 10])
Output shape: torch.Size([2, 10, 64])
Total parameters: 164,096

Forward pass successful.
Each token now carries context from every other token in the sequence.
```

Input is a batch of 2 sequences, each 10 tokens long.
Output is the same shape but each token now carries context from the entire sequence.

---

### Read the full breakdown

[dsunpacked.substack.com](https://dsunpacked.substack.com)
