"""Dump held-out probe predictions for the figure.

The figure shows what the probe actually does rather than a summary
score: predicted position against true position on held-out samples, for
an observation that determines position and one that does not.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from demo_learned_view import collect            # noqa: E402
from scenarios.learned import _ridge, _score     # noqa: E402

frames, ego, pos = collect()
print(f"{len(pos)} samples; frame {frames.shape[1]}, ego {ego.shape[1]}")

rng = np.random.default_rng(0)
order = rng.permutation(len(pos))
n_test = int(round(0.3 * len(pos)))
te, tr = order[:n_test], order[n_test:]

out = {"n_samples": int(len(pos)), "n_test": int(n_test)}
for name, X in (("full_grid", frames), ("egocentric", ego)):
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-8] = 1.0
    W = _ridge((X[tr] - mu) / sd, pos[tr], 1.0)
    pred = np.hstack([(X[te] - mu) / sd, np.ones((len(te), 1))]) @ W
    # Two metrics, deliberately. The view's gate treats position as
    # ORDINAL and scores it by exact match after rounding, which is the
    # number quoted in the paper; R^2 describes the same predictions on a
    # continuous scale and is what the scatter shows.
    exact = [_score(pos[te][:, j], pred[:, j], True) for j in range(2)]
    scores = [_score(pos[te][:, j], pred[:, j], False) for j in range(2)]
    out[name] = {
        "width": int(X.shape[1]),
        "r2": [float(s) for s in scores],
        "exact_match": [float(s) for s in exact],
        "true_x": pos[te][:, 0].tolist(), "pred_x": pred[:, 0].tolist(),
        "true_y": pos[te][:, 1].tolist(), "pred_y": pred[:, 1].tolist(),
    }
    print(f"  {name}: exact = {exact[0]:.3f}, {exact[1]:.3f} | "
          f"R2 = {scores[0]:.3f}, {scores[1]:.3f}")

p = Path(__file__).parent / "probe_predictions.json"
p.write_text(json.dumps(out))
print("wrote", p, f"({p.stat().st_size/1024:.0f} KB)")
