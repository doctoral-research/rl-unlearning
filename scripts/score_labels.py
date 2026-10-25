"""Score a corpus entry's scenario against reference labels.

This is the measurement the corpus was missing. Expressibility says the
author believes the grammar captures a behaviour; fidelity says the
formal spec and an independent reading of the same English sentence
select the same steps.

The labelled file already carries the per-step feature values the labels
were assigned to, so the scenario is replayed on exactly those steps. No
new rollouts, and no chance of scoring against a different trajectory
than the annotator saw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scenarios.corpus import (                                  # noqa: E402
    load_corpus, register_contrib_providers, resolve_views,
)
from scenarios.labels import agreement, load_label_file         # noqa: E402


def trajectories_from(doc, features):
    """Rebuild the exact per-step feature vectors the annotator saw."""
    out = []
    for ep in doc["episodes"]:
        feats = [[float(st[f]) for f in features] for st in ep["steps"]]
        acts = [st.get("action") for st in ep["steps"]]
        out.append({"features": feats, "actions": acts})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, nargs="+")
    ap.add_argument("--corpus", default="benchmark/behaviours")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    register_contrib_providers()
    entries = {e.id: e for e in load_corpus(args.corpus)}

    print(f"{'entry':<34}{'acc':>7}{'jacc':>7}{'prec':>7}{'rec':>7}"
          f"{'label%':>8}{'spec%':>7}")
    print("-" * 77)
    rows = []
    for path in args.labels:
        doc = yaml.safe_load(Path(path).read_text())
        ls = load_label_file(path)
        entry = entries.get(doc["entry"])
        if entry is None:
            print(f"{doc['entry']:<34}  unknown corpus entry")
            continue
        view = resolve_views([entry.domain]).get(entry.domain)
        scen = entry.load_scenario(Path(args.root), view=view)
        trajs = trajectories_from(doc, doc["features"])
        masks = [scen.step_labels(t) for t in trajs]
        got = agreement(ls, masks)
        # If neither the annotator nor the spec ever fires, accuracy is 1.0
        # and means nothing: the episodes simply do not contain the
        # behaviour. Reporting that as fidelity would be the same kind of
        # self-congratulation the corpus already had.
        # Either side being constant makes the comparison uninformative.
        # Both-zero means the episodes do not contain the behaviour; an
        # annotator that answers all-true or all-false is not reading the
        # steps at all, which the 32B stand-in did on every episode.
        const_label = got["label_rate"] in (0.0, 1.0)
        const_pred = got["pred_rate"] in (0.0, 1.0)
        got["degenerate"] = const_label or const_pred
        rows.append((entry.id, got))
        flag = "   <-- VACUOUS" if got["degenerate"] else ""
        print(f"{entry.id:<34}{got['accuracy']:7.3f}{got['jaccard']:7.3f}"
              f"{got['precision']:7.3f}{got['recall']:7.3f}"
              f"{got['label_rate']:8.3f}{got['pred_rate']:7.3f}{flag}")

    if rows:
        useful = [r for r in rows if not r[1]["degenerate"]]
        print("-" * 77)
        if useful:
            print(f"{'mean (informative only)':<34}"
                  f"{np.mean([r[1]['accuracy'] for r in useful]):7.3f}"
                  f"{np.mean([r[1]['jaccard'] for r in useful]):7.3f}")
        n_deg = len(rows) - len(useful)
        if n_deg:
            print(f"\n{n_deg}/{len(rows)} entries are VACUOUS: one side "
                  "never varies, so the agreement carries no information.")
            print("Either the episodes do not contain the behaviour "
                  "(generate the task from a trained policy with "
                  "--checkpoint), or the annotator answered uniformly and "
                  "is not reading the steps.")
        ann = {yaml.safe_load(Path(p).read_text()).get("annotator")
               for p in args.labels}
        print(f"\nannotator(s): {', '.join(str(a) for a in sorted(ann, key=str))}")
        if any(str(a).startswith("synthetic") for a in ann):
            print("These are STAND-IN labels from a model, not human ground "
                  "truth. Fidelity reported from them is provisional.")


if __name__ == "__main__":
    main()
