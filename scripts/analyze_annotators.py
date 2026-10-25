"""Inter-annotator agreement on behaviour descriptions.

Fidelity compares one reading of a description against its formalisation.
That leaves the obvious question unanswered: how much do two readers of
the SAME sentence agree with each other, before any specification is
involved? Disagreement between the specification and a reader is only
damning if readers agree among themselves.

Each annotator saw the description and the per-step feature values and
nothing else: not the scenario file, not the other annotators, not the
results. Where a description leaves a threshold open they chose one and
recorded it.

These are independent readings, not human ground truth. They come from
the same model family and so share priors that a group of people would
not, which makes the agreement reported here an optimistic bound on what
human annotators would show.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def align(ref, x):
    """Return x, flipped if the annotator labelled compliance rather than
    the violation. Several corpus descriptions carry both a prohibition
    and the desired state ("Stop descending at a steep tilt; keep the
    lander close to upright"), so which of the two a label marks is
    itself underdetermined. Polarity is recoverable, so it is reported
    separately rather than allowed to masquerade as disagreement."""
    a, b = np.asarray(ref, bool), np.asarray(x, bool)
    return (~b, True) if jaccard(a, ~b) > jaccard(a, b) else (b, False)


def jaccard(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return 1.0 if union == 0 else float((a & b).sum()) / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/annot")
    ap.add_argument("--vague", nargs="*", default=[
        "lbf.no_crowding", "lbf.no_freeloading", "walker2d.no_forward_fall"])
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("annotator_*.json"))
    if len(files) < 2:
        raise SystemExit("need at least two annotators to compare")
    data = {f.stem.split("_")[-1]: json.loads(f.read_text()) for f in files}
    names = sorted(data)
    entries = sorted(set.intersection(*(set(d) for d in data.values())))

    print(f"annotators: {', '.join(names)}\n")
    print(f"{'entry':<28}{'rates':>22}{'mean pairwise J':>17}  kind")
    print("-" * 78)
    rows = []
    flips_total = 0
    ref = names[0]
    for e in entries:
        raw = {n: data[n][e]["labels"] for n in names}
        labs, flipped = {ref: np.asarray(raw[ref], bool)}, []
        for n in names[1:]:
            aligned, was = align(raw[ref], raw[n])
            labs[n] = aligned
            if was:
                flipped.append(n)
        flips_total += len(flipped)
        rates = [float(np.mean(labs[n])) for n in names]
        js = [jaccard(labs[x], labs[y]) for x, y in itertools.combinations(names, 2)]
        kind = "degree word" if e in args.vague else "fixed criterion"
        rows.append((e, float(np.mean(js)), kind, rates))
        note = f"  [polarity flipped: {','.join(flipped)}]" if flipped else ""
        print(f"{e:<28}{'/'.join(f'{r:.2f}' for r in rates):>22}"
              f"{np.mean(js):>17.3f}  {kind}{note}")

    print("-" * 78)
    for kind in ("fixed criterion", "degree word"):
        sel = [r[1] for r in rows if r[2] == kind]
        if sel:
            print(f"mean pairwise Jaccard, {kind:<16}: {np.mean(sel):.3f}  "
                  f"(n={len(sel)})")
    n_pairs = len(entries) * (len(names) - 1)
    print(f"\npolarity: {flips_total}/{n_pairs} annotator-entry pairs labelled "
          "compliance rather than the violation;\nagreement above is computed "
          "after aligning polarity, so it isolates threshold choice.")
    print("\nA reader-to-reader Jaccard well below 1.0 means the sentence, not "
          "the formalisation,\nis what is underdetermined.")

    out = Path(args.dir) / "agreement.json"
    out.write_text(json.dumps(
        {"annotators": names,
         "entries": [{"entry": e, "mean_pairwise_jaccard": j, "kind": k,
                      "rates": [float(x) for x in r]} for e, j, k, r in rows]},
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
