"""Grokking on GF(7^2)* multiplication, with mechanistic probes.

Trains a small MLP on x * y in the unit group of GF(49) (48 opaque
element ids; secretly modular addition of discrete logs) with heavy
weight decay, the canonical grokking recipe. At checkpoints we record:

* train/test accuracy (the grokking curve),
* Fourier concentration of embeddings ordered by discrete log
  (Nanda-style progress measure: does the net find the hidden cycle?),
* Frobenius alignment z-score (Galois-specific probe: does x -> x^7
  become an orthogonal transform of embedding space?).

Writes metrics to experiments/outputs/grokking_gf49.json and prints a
summary. Runtime: a few minutes on CPU.

Run:  python experiments/grokking_gf.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make the repo importable regardless of interpreter or install state.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from galois_ml.grokking import (
    GFMultiplicationTask,
    GrokkingMLP,
    fourier_concentration,
    frobenius_alignment_zscore,
)

P, N = 7, 2                 # GF(49); unit group C_48
TRAIN_FRACTION = 0.4
EPOCHS = 40_000
CHECK_EVERY = 200
LR = 1e-3
WEIGHT_DECAY = 1.0
SEED = 0
OUT = Path("experiments/outputs")


@torch.no_grad()
def _accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(X).argmax(1) == y).float().mean())


def main() -> None:
    torch.manual_seed(SEED)
    task = GFMultiplicationTask(P, N, seed=SEED)
    X_tr, y_tr, X_te, y_te = task.split(TRAIN_FRACTION, seed=SEED)
    print(f"GF({P**N})* multiplication: {task.group_order} elements, "
          f"{len(y_tr)} train / {len(y_te)} test pairs")

    model = GrokkingMLP(task.group_order, d_embed=96, d_hidden=384)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()

    rows = []
    grok_epoch = None
    memo_epoch = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()

        if epoch % CHECK_EVERY == 0 or epoch == 1:
            tr_acc = _accuracy(model, X_tr, y_tr)
            te_acc = _accuracy(model, X_te, y_te)
            E = model.embed.weight.detach().numpy()
            fc = fourier_concentration(E, task.dlog)
            resid, base, z = frobenius_alignment_zscore(
                E, task.frobenius_perm, n_baseline=20, seed=1
            )
            rows.append({
                "epoch": epoch, "loss": float(loss), "train_acc": tr_acc,
                "test_acc": te_acc, "fourier_concentration": fc,
                "frobenius_residual": resid, "frobenius_z": z,
            })
            if memo_epoch is None and tr_acc > 0.99:
                memo_epoch = epoch
            if grok_epoch is None and te_acc > 0.95:
                grok_epoch = epoch
            if epoch % 2000 == 0 or epoch == 1:
                print(f"epoch {epoch:>6}  train {tr_acc:.3f}  "
                      f"test {te_acc:.3f}  fourier {fc:.3f}  "
                      f"frob_z {z:+.1f}")
            if grok_epoch and epoch >= grok_epoch + 4000:
                break  # converged well past the transition

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "grokking_gf49.json"
    out_json.write_text(json.dumps({
        "config": {
            "p": P, "n": N, "train_fraction": TRAIN_FRACTION,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED,
        },
        "metrics": rows,
    }, indent=1))

    print(f"\nmetrics -> {out_json}")
    print(f"memorization (train > 0.99): epoch {memo_epoch}")
    print(f"grokking (test > 0.95):      epoch {grok_epoch}")
    last = rows[-1]
    print(f"final: test {last['test_acc']:.4f}, "
          f"fourier concentration {last['fourier_concentration']:.3f}, "
          f"frobenius z {last['frobenius_z']:+.1f}")


if __name__ == "__main__":
    main()
