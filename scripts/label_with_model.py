"""Fill a label task with a model acting as a stand-in annotator.

Reference labels are supposed to come from a person who has not seen the
specification. Until someone does that, a local model can stand in, and
it is a better stand-in than it might appear: it reads only the
behaviour's English description and the per-step feature values, never
the scenario YAML, so its reading of the description really is
independent of the formalisation.

What this measures is therefore real but narrower than human labels:
whether an independent reader of the prose arrives at the same steps the
YAML selects. Where they disagree, one of the two is wrong about what the
sentence means, and that is exactly the drift this project studies.

What it is NOT is ground truth. The annotator is recorded as
`synthetic:<model>` and `independent: false` so no later reader mistakes
it for a human judgement, and a corpus entry scored this way should say
so. Replace it with human labels when you have them.

    python scripts/label_with_model.py --task task.yaml \
        --backend ollama:qwen2.5-coder:32b --out labelled.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scenarios.backends import BackendError, make_backend      # noqa: E402

PROMPT = """You are labelling a reinforcement-learning trajectory.

The behaviour to identify is:
  {description}

Environment: {env}
Features recorded at each step: {features}

Below are {n} consecutive steps of one episode. For EACH step decide
whether the described behaviour is occurring AT THAT STEP.

{table}

Answer with a JSON object mapping each step's t value to true or false,
covering all {n} steps: {{"0": false, "1": true, ...}}. Keys make the
alignment explicit; a bare array invites an off-by-one. No other text.
"""


def parse_bools(text: str, ts):
    """Pull per-step labels out of a model reply.

    Keyed by step index rather than positional: asked for a bare array,
    a model returned 31 values for a 30-step table, and there is no safe
    way to guess which step was duplicated. A mapping either covers every
    step or is rejected.
    """
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out = []
    for t in ts:
        if str(t) in obj:
            out.append(bool(obj[str(t)]))
        elif t in obj:
            out.append(bool(obj[t]))
        else:
            return None
    return out


def render_table(steps, features):
    head = "  t  " + "  ".join(f"{f:>10}" for f in features) + "   action"
    rows = [head]
    for st in steps:
        vals = "  ".join(f"{st.get(f, 0):>10}" for f in features)
        rows.append(f"{st['t']:>3}  {vals}   {st.get('action')}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    try:
        backend = make_backend(args.backend)
    except BackendError as exc:
        raise SystemExit(f"cannot build backend: {exc}")

    doc = yaml.safe_load(Path(args.task).read_text())
    features = doc["features"]
    n_fail = 0

    for ep in doc["episodes"]:
        steps = ep["steps"]
        prompt = PROMPT.format(
            description=doc["description"], env=doc.get("env", "?"),
            features=", ".join(features), n=len(steps),
            table=render_table(steps, features))
        labels = None
        for _ in range(args.retries + 1):
            labels = parse_bools(backend(prompt), [st["t"] for st in steps])
            if labels is not None:
                break
        if labels is None:
            n_fail += 1
            print(f"  episode {ep['id']}: no parseable answer, left blank")
            continue
        for st, lab in zip(steps, labels):
            st["label"] = bool(lab)
        rate = sum(labels) / len(labels)
        print(f"  episode {ep['id']}: {len(labels)} steps, "
              f"{rate:.0%} positive")

    doc["annotator"] = f"synthetic:{args.backend}"
    doc["independent"] = False
    doc["annotator_note"] = (
        "Labels produced by a model shown only the description and the "
        "feature values, never the scenario. Independent of the "
        "formalisation, but NOT human ground truth.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump(doc, sort_keys=False, width=100))
    print(f"wrote {args.out}")
    if n_fail:
        print(f"WARNING: {n_fail} episode(s) unlabelled; the loader will "
              "refuse this file until they are filled in.")


if __name__ == "__main__":
    main()
