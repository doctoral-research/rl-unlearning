"""Regenerate the fidelity table from the label files on disk.

Hand-maintaining this list went stale within a day of writing it. Every
number here is recomputed from the label file and the scenario it points
at, so the paper cannot quote a fidelity that no longer matches the
corpus.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, yaml
RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT / "src"))
from scenarios.corpus import (load_corpus, resolve_views,          # noqa: E402
                              register_contrib_providers)
from scenarios.labels import load_label_file                       # noqa: E402

register_contrib_providers()
entries = {e.id: e for e in load_corpus(str(RT / "benchmark/behaviours"))}
# One person's labels of every step, made by hand without the reference or
# the formalisations in view.
PERSON = json.loads((RT / "benchmark/labels/human/annotator_H1.json").read_text())["entries"]


def kappa(a, b):
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


rows = []
for e in sorted(entries.values(), key=lambda x: x.id):
    if not e.labels_ref:
        continue
    path = RT / e.labels_ref
    doc = yaml.safe_load(path.read_text())
    ls = load_label_file(path)
    view = resolve_views([e.domain])[e.domain]
    scen = e.load_scenario(RT, view=view)
    names = doc["features"]
    human, spec, person = [], [], []
    for ep, mask in zip(doc["episodes"], ls.episodes):
        feats = [[float(s[n]) for n in names] for s in ep["steps"]]
        acts = [s.get("action") for s in ep["steps"]]
        spec.extend(int(x) for x in scen.step_labels(
            {"features": feats, "actions": acts}))
        human.extend(int(v) for v in mask)
        person.extend(PERSON[e.id]["episodes"][str(ep["id"])])
    h, p = np.asarray(human, bool), np.asarray(spec, bool)
    q = np.asarray(person, bool)
    assert len(q) == len(h)
    inter, union = int((h & p).sum()), int((h | p).sum())
    tp, fp, fn = inter, int((~h & p).sum()), int((h & ~p).sum())
    # Cohen's kappa: Jaccard ignores agreement on negatives, so on a rare
    # behaviour it cannot tell a near miss from a disagreement about
    # everything. Kappa corrects raw agreement for chance instead.
    po = float((h == p).mean())
    pe = float(h.mean() * p.mean() + (1 - h.mean()) * (1 - p.mean()))
    rows.append({
        "entry": e.id, "verdict": e.verdict, "n_steps": len(human),
        "jaccard": 1.0 if union == 0 else inter / union,
        "kappa": 1.0 if pe == 1.0 else (po - pe) / (1 - pe),
        "accuracy": po,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "label_rate": float(h.mean()), "spec_rate": float(p.mean()),
        "annotator": doc.get("annotator", "?"),
        "independent": doc.get("independent", False),
        "person": {"kappa_reference": kappa(q, h), "kappa_formalisation": kappa(q, p),
                   "identical_to_reference": bool((q == h).all()),
                   "rate": float(q.mean())},
    })

# Section 4.3 quotes these about the person's labels. Check them, so new
# labels cannot leave the prose wrong.
person = {r["entry"]: r["person"] for r in rows}
kr = [v["kappa_reference"] for v in person.values()]
assert (round(float(np.median(kr)), 2), round(float(np.mean(kr)), 2)) == (0.95, 0.79)
assert sum(v["identical_to_reference"] for v in person.values()) == 6
WHOLE = ("doorkey.no_key_hoarding", "fourrooms.no_backtracking",
         "fourrooms.no_pacing", "halfcheetah.no_front_flip")
assert [round(f(person[e]["kappa_reference"] for e in WHOLE), 2) for f in (min, max)] == [0.92, 1.0]
assert max(person[e]["kappa_formalisation"] for e in WHOLE) < 0.545
assert round(person["walker2d.no_forward_fall"]["kappa_formalisation"], 2) == 1.0
assert round(person["lunarlander.no_tilt"]["kappa_reference"], 2) == 0.83
assert person["lunarlander.no_near_miss"]["rate"] == 0.0

m = RT / "paper/data/measurements.json"
j = json.loads(m.read_text())
j["corpus"]["fidelity"] = rows
j["corpus"]["fidelity_measured"] = len(rows)
m.write_text(json.dumps(j, indent=2))
print(f"{'entry':<32}{'J':>7}{'kappa':>7}{'prec':>7}{'rec':>7}{'reader':>8}{'spec':>7}  annotator")
print("-" * 92)
for r in rows:
    print(f"{r['entry']:<32}{r['jaccard']:>7.3f}{r['kappa']:>7.3f}{r['precision']:>7.3f}"
          f"{r['recall']:>7.3f}{r['label_rate']:>8.3f}{r['spec_rate']:>7.3f}  "
          f"{r['annotator']}")
print(f"\n{len(rows)} entries with reference labels; "
      f"mean Jaccard {np.mean([r['jaccard'] for r in rows]):.3f}, "
      f"mean kappa {np.mean([r['kappa'] for r in rows]):.3f}")
