"""Reader agreement and the polarity experiment, recomputed from disk.

Writes measurements.json["readers"] with two tables:

  entries   for each of the 15 labelled entries, who read it, under which
            prompt, and how far the readers agree with EACH OTHER (Cohen's
            kappa and Jaccard, pairwise, after polarity alignment).
  polarity  every reader-entry pair that tests description voice, with the
            instruction the reader received and whether it marked the
            complement of the formalisation.

Sources, in order of preference: raw per-reader label files kept in the
repository; otherwise the reader record in the label file. Three cases use
the record rather than raw labels, and say so in the output:
  * paired readers whose labels were identical (the label file stores that
    one label set, so both readers' labels are on disk by construction);
  * the doorway entry, whose three readers marked 12, 18 and 29 steps with
    pairwise Jaccard 12/18, 12/29 and 18/29. Those values force the three
    sets to be nested, so the 2x2 tables, and hence kappa, are determined;
  * readers D and E, whose labels were scored against the formalisation
    when they were written (J = 0.000, 1.000 after flipping) but not kept.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT / "src"))
from scenarios.corpus import (load_corpus, resolve_views,          # noqa: E402
                              register_contrib_providers)

register_contrib_providers()
ENTRIES = {e.id: e for e in load_corpus(str(RT / "benchmark/behaviours"))}
LABELS = RT / "benchmark/labels"


def jaccard(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = int((a | b).sum())
    return 1.0 if u == 0 else float((a & b).sum()) / u


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def nested_kappa(small, large, n):
    """Kappa for two label sets where the smaller lies inside the larger."""
    a = np.zeros(n, bool); a[:small] = True
    b = np.zeros(n, bool); b[:large] = True
    return kappa(a, b), jaccard(a, b)


def spec_labels(eid, doc):
    e = ENTRIES[eid]
    view = resolve_views([e.domain])[e.domain]
    scen = e.load_scenario(RT, view=view)
    out = []
    for ep in doc["episodes"]:
        feats = [[float(s[n]) for n in doc["features"]] for s in ep["steps"]]
        acts = [s.get("action") for s in ep["steps"]]
        out.extend(int(x) for x in scen.step_labels(
            {"features": feats, "actions": acts}))
    return np.asarray(out, bool)


def task_table(path):
    """Rows of a reader task file: (episode, t, features..., action)."""
    lines = path.read_text().splitlines()
    cols = next(l for l in lines if l.startswith("Columns:"))
    cols = [c.strip() for c in cols.split(":", 1)[1].split(",")]
    rows = [l.split() for l in lines
            if l.startswith("  ") and len(l.split()) == len(cols)
            and l.split()[0].lstrip("-").isdigit()]
    return cols, rows


def spec_on_task(eid, path):
    """Formalisation applied to the rows of a task file."""
    cols, rows = task_table(path)
    eps = {}
    for r in rows:
        eps.setdefault(r[0], []).append(r)
    doc = {"features": cols[2:-1], "episodes": [
        {"steps": [dict(zip(cols[2:-1], map(float, r[2:-1])),
                        action=None if r[-1] == "None" else int(float(r[-1])))
                   for r in ep]} for ep in eps.values()]}
    return spec_labels(eid, doc)


def label_doc(eid):
    return yaml.safe_load((LABELS / f"{eid}.yaml").read_text())


def stored(doc):
    return np.asarray([bool(s["label"]) for ep in doc["episodes"]
                       for s in ep["steps"]])


# ---------------------------------------------------------------- entries
ROUND1 = {n: json.loads((LABELS / f"independent/annotator_{n}.json").read_text())
          for n in "ABC"}
entries = []
for eid in sorted(e.id for e in ENTRIES.values() if e.labels_ref):
    doc = label_doc(eid)
    readers = doc["annotator"].split("(")[1].split(";")[0].rstrip(")")
    readers = [r.strip() for r in readers.split(",")]
    row = {"entry": eid, "readers": readers, "prompt": doc.get("prompt"),
           "model": doc.get("reader_model"), "n_steps": int(len(stored(doc)))}
    if readers == ["A", "B", "C"]:
        labs = {n: np.asarray(ROUND1[n][eid]["labels"], bool) for n in "ABC"}
        aligned, flipped = {"A": labs["A"]}, []
        for n in "BC":
            x = labs[n]
            if kappa(labs["A"], ~x) > kappa(labs["A"], x):
                x, _ = ~x, flipped.append(n)
            aligned[n] = x
        pairs = list(itertools.combinations("ABC", 2))
        row["pair_kappa"] = [kappa(aligned[a], aligned[b]) for a, b in pairs]
        row["pair_jaccard"] = [jaccard(aligned[a], aligned[b]) for a, b in pairs]
        row["polarity_aligned"] = flipped
        maj = np.stack([aligned[n] for n in "ABC"]).sum(0) >= 2
        assert (maj == stored(doc)).all(), eid
        row["source"] = "raw labels"
    elif eid == "fourrooms.avoid_br_tr_door":
        n = row["n_steps"]
        sizes = {"Q": 12, "P": 18, "T": 29}   # from the readers' own reports
        pairs = list(itertools.combinations("PQT", 2))
        kj = [nested_kappa(*sorted((sizes[a], sizes[b])), n) for a, b in pairs]
        row["pair_kappa"] = [k for k, _ in kj]
        row["pair_jaccard"] = [j for _, j in kj]
        row["reader_positives"] = sizes
        row["polarity_aligned"] = []
        row["source"] = "nested label sets, determined by counts"
    else:
        assert len(readers) == 2 and "identical labels" in doc["annotator_note"].lower(), eid
        row["pair_kappa"], row["pair_jaccard"] = [1.0], [1.0]
        row["polarity_aligned"] = []
        row["source"] = "identical labels, stored once"
    row["mean_kappa"] = float(np.mean(row["pair_kappa"]))
    row["mean_jaccard"] = float(np.mean(row["pair_jaccard"]))
    entries.append(row)

# --------------------------------------------------------------- polarity
GOAL_WORDING = {
    "cartpole.left_only": "Balance the pole only while the cart stays on the left half of the track.",
    "pendulum.avoid_left_swing": "Swing up using only the right half of the circle.",
    "fourrooms.avoid_br_detour": "Reach the goal without routing through the bottom-right room.",
}
INSTRUCTION = {"r01_ABC": "neutral", "r02_DE": "neutral", "r11_UV": "strict"}
pol = []


def add(reader, eid, voice, prompt, x, spec, source):
    k, kf = kappa(spec, x), kappa(spec, ~x)
    pol.append({"reader": reader, "entry": eid, "voice": voice,
                "prompt": prompt, "instruction": INSTRUCTION.get(prompt, "strict"),
                "kappa": k, "kappa_flipped": kf,
                "jaccard": jaccard(spec, x), "jaccard_flipped": jaccard(spec, ~x),
                "inverted": bool(k < 0 < kf and kf > 0.5), "source": source})


# goal voice, neutral instruction: round 1 on CartPole (its own 105 steps)
spec = spec_on_task("cartpole.left_only", LABELS / "independent/cartpole.left_only.txt")
for n in "ABC":
    add(n, "cartpole.left_only", "goal", "r01_ABC",
        np.asarray(ROUND1[n]["cartpole.left_only"]["labels"], bool), spec, "raw labels")
# goal voice, neutral instruction stating that polarity must be chosen: D, E
for n in "DE":
    for eid in ("pendulum.avoid_left_swing", "fourrooms.avoid_br_detour"):
        pol.append({"reader": n, "entry": eid, "voice": "goal", "prompt": "r02_DE",
                    "instruction": "neutral", "jaccard": 0.0, "jaccard_flipped": 1.0,
                    "kappa": None, "kappa_flipped": None, "inverted": True,
                    "source": "scored when written, raw labels not kept"})
# goal voice, strict instruction: U, V on the same data N and O labelled
CONF = {n: json.loads((LABELS / f"voice_confound/annotator_{n}.json").read_text())
        for n in "UV"}
for eid, wording in GOAL_WORDING.items():
    doc = label_doc(eid)
    shown = (LABELS / f"voice_confound/{eid}.txt").read_text()
    assert wording in shown, eid
    spec = spec_labels(eid, doc)
    for n in "UV":
        add(n, eid, "goal", "r11_UV", np.asarray(CONF[n][eid]["labels"], bool),
            spec, "raw labels")
# prohibition voice, neutral instruction: round 1 on its four other entries
for eid in ("lbf.no_crowding", "lbf.no_freeloading", "lunarlander.no_tilt",
            "walker2d.no_forward_fall"):
    doc = label_doc(eid)
    ref = np.asarray(ROUND1["A"][eid]["labels"], bool)
    for n in "ABC":
        x = np.asarray(ROUND1[n][eid]["labels"], bool)
        # Polarity is judged against the other readers here, not against the
        # formalisation, which disagrees with all three on degree words.
        k, kf = kappa(ref, x), kappa(ref, ~x)
        pol.append({"reader": n, "entry": eid, "voice": "prohibition",
                    "prompt": "r01_ABC", "instruction": "neutral",
                    "kappa_vs_reader_A": k, "inverted": bool(kf > k),
                    "positives": int(x.sum()), "n_steps": int(len(x)),
                    "source": "raw labels"})
# prohibition voice, strict instruction: every stored label set from H to T
for row in entries:
    if row["prompt"] in (None, "r01_ABC"):
        continue
    for n in row["readers"]:
        pol.append({"reader": n, "entry": row["entry"], "voice": "prohibition",
                    "prompt": row["prompt"] if isinstance(row["prompt"], str)
                    else ("r10_T" if n == "T" else "r08_PQ"),
                    "instruction": "strict", "inverted": False,
                    "source": "stored labels mark the behaviour named"})

cells = {}
for p in pol:
    c = cells.setdefault(f"{p['voice']}|{p['instruction']}", [0, 0])
    c[0] += p["inverted"]; c[1] += 1

m = RT / "paper/data/measurements.json"
j = json.loads(m.read_text())
j["readers"] = {"entries": entries, "polarity": pol,
                "polarity_cells": {k: {"inverted": v[0], "pairs": v[1]}
                                   for k, v in cells.items()},
                "models": {"claude-opus-5": "readers A-T",
                           "claude-opus-5-5": "readers U, V"}}
m.write_text(json.dumps(j, indent=2))

print(f"{'entry':<34}{'readers':<9}{'kappa':>7}{'J':>7}  source")
for r in entries:
    print(f"{r['entry']:<34}{''.join(r['readers']):<9}{r['mean_kappa']:>7.3f}"
          f"{r['mean_jaccard']:>7.3f}  {r['source']}"
          + (f"  [aligned {','.join(r['polarity_aligned'])}]" if r["polarity_aligned"] else ""))
n_exact = sum(r["mean_kappa"] == 1.0 for r in entries)
print(f"\nreaders agree exactly on {n_exact} of {len(entries)} entries")
print("\npolarity (inverted / pairs):")
for k, v in sorted(cells.items()):
    print(f"  {k:<26} {v[0]}/{v[1]}")
