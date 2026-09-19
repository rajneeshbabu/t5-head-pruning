# Which Attention Heads Does T5 Actually Need?

[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-222?logo=github)](https://rajneeshbabu.github.io/t5-head-pruning/)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)](t5_head_pruning.ipynb)
[![Transformers](https://img.shields.io/badge/HuggingFace-T5--small-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/t5-small)
[![Heads](https://img.shields.io/badge/heads%20scored-144-34d399)](t5_head_pruning.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

🌐 **[View the project page →](https://rajneeshbabu.github.io/t5-head-pruning/)**

T5-small has **144 attention heads** — 6 encoder layers and 6 decoder layers of 8 heads
each, plus 6 layers of cross-attention joining them. They are not equally useful. This
notebook scores every head by how much the loss depends on it, prunes the least important
ones, and measures what that costs in translation quality.

## The headline

English→German on the Multi30k 2016 test set, baseline BLEU **0.2534**.

| Heads removed | Chosen by importance | Chosen at random | Gap |
|---|---|---|---|
| 20% of one attention type | 0.2529 avg | 0.2607 avg | −0.0097 |
| 40% | 0.2515 avg | 0.2196 avg | **+0.0319** |
| 60% | 0.2405 avg | 0.1618 avg | **+0.0787** |

The clearest single case is **cross-attention at 60% removed**: choosing heads by
importance keeps **91% of baseline BLEU**, choosing the same number at random keeps
**13%**. Identical compute saved, 0.198 BLEU apart.

At 20% the ranking barely matters and random selection can come out ahead — on 120
sentences that difference is inside the noise. The scores only prove their worth once
enough is removed that the choice matters, and the notebook says so rather than quietly
reporting the favourable rows.

## Removing a whole attention type

| Removed entirely | BLEU | What comes out |
|---|---|---|
| Nothing | 0.2534 | *Ein Mann in einem orangen Hut, der auf etwas dreht.* |
| Decoder self-attention | 0.0256 | *Ein Mann in einem Mann in einem Mann in einem…* |
| Encoder self-attention | 0.0000 | *a.* |
| Cross-attention | 0.0000 | *a,,,,,,,,,,,,,, a,,, and to the a* |

Cross-attention is the only path from encoder to decoder. Without it the decoder cannot
see the English sentence at all and produces German-shaped noise. Decoder self-attention
fails differently and more revealingly — the model still produces fluent fragments but
loses track of what it has already said, so it loops.

## Method

**Importance.** The score for head *h* is how much the loss moves when that head is
switched off:

$$I_h = \left| \frac{\partial \mathcal{L}}{\partial \xi_h} \right|$$

where ξ is the head's mask, sitting at 1. This is the first-order estimate from
[Michel, Levy and Neubig (2019)](https://arxiv.org/abs/1905.10650), and it costs **one
forward and backward pass for all 144 heads** instead of 144 separate evaluations. All
144 were scored in 20 seconds on a laptop CPU.

**Masking.** Recent `transformers` no longer accepts `head_mask` through `generate`, so
the mask is applied with a forward pre-hook on each attention module's output projection.
The input there is `n_heads × d_kv` wide, so reshaping and multiplying by a per-head
scalar switches heads off — and because it is an ordinary tensor multiply, the mask is
differentiable, which is what makes the gradient above computable.

**BLEU** is implemented directly: clipped n-gram precisions, geometric mean, brevity
penalty, pooled at corpus level. Sanity checks in the notebook confirm identical text
scores 1.0, that a truncated candidate is caught by the brevity penalty, and that a
repetitive one is caught by clipping.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook t5_head_pruning.ipynb
```

Downloads t5-small (~240 MB) and the Multi30k test set on first run. About 15 minutes on
a laptop CPU, no GPU needed. The notebook is committed with its outputs.

## What this is not

The heads are **masked, not deleted**. Masking measures the effect of removal exactly;
physically deleting the pruned heads is what would make the model smaller and faster, and
is the natural next step. The BLEU numbers here are what that smaller model would score.
