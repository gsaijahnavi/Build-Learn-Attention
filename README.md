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

---

### What you'll see

**Part 1 — Attention weights on a real sentence**

The classic coreference example: *"the animal did not cross because it was tired"*

The script prints which tokens each token attends to most — for one attention head:

```
============================================================
  Build & Learn #1 — Self-Attention from Scratch
============================================================

  Sentence: "the animal did not cross because it was tired"
  Tokens:    9
  Heads:     4

  Attention weights — Head 1
  (row = query token, col = key token, value = attention weight)

               the      animal     did       not      cross    because      it       was      tired
            ------------------------------------------------------------------------------------------
  the       |                                                              █         █           → attends most to: 'was'
  animal    |                                               █                        █           → attends most to: 'tired'
  did       |                                     █         █                                   → attends most to: 'cross'
  not       |            █                                                            █          → attends most to: 'tired'
  cross     |  █         █         █                                      █                     → attends most to: 'the'
  because   |            █                  █                                         ██         → attends most to: 'tired'
  it        |            █                                  █              █                     → attends most to: 'was'
  was       |  █         █         █                                       █                     → attends most to: 'the'
  tired     |  █                                                            █                    → attends most to: 'was'
```

Notice `it` attends strongly to `animal` — exactly what you'd want for coreference resolution. This is an untrained model, so the weights are random, but the mechanism is real. With training, these patterns sharpen into meaningful relationships.

**Part 2 — Full encoder forward pass**

```
============================================================
  Full Encoder Forward Pass
============================================================

  Input shape:      [2, 10]   (batch=2, seq_len=10)
  Output shape:     [2, 10, 64]  (batch=2, seq_len=10, embed=64)
  Parameters:       164,096

  Each of the 10 tokens now carries context from all other tokens.
  That's self-attention.
```

Input: raw token indices. Output: same shape, but every token embedding now contains information from the full sequence.

---

### Read the full breakdown

[dsunpacked.substack.com](https://dsunpacked.substack.com)
