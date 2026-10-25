"""Measure the authoring repair loop against a real model.

The corpus already supplies the benchmark. Every formalised entry has a
natural-language `description` written before any YAML, and a reference
scenario a human wrote for it. That is exactly an authoring task with a
ground truth: give the model the description, and ask two questions.

    accepted   did any candidate clear the validator, the typed feature
               view, and the non-vacuity check on real rollouts
    agreement  does the accepted scenario *mean* the same thing as the
               reference, as per-step label agreement on those rollouts

The second is the one that matters. A scenario can pass every gate and
still describe a different behaviour, so acceptance alone is only an
upper bound on authoring quality.

Pass `--rollouts` to enable both. Without it, only acceptance is
reported and the non-vacuity gate does not run.

An agreement score is meaningless when the reference never fires on the
sample: everything then agrees with everything. Those rows are marked
VACUOUS and excluded from the mean rather than inflating it.

This script deliberately refuses to report numbers from a scripted
backend. A repair loop that converges against canned responses is a test,
not a result.

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/run_authoring_eval.py --backend anthropic
    python scripts/run_authoring_eval.py --backend anthropic:claude-opus-5 \
        --domain MiniGrid-FourRooms-v0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scenarios.authoring import author_scenario                 # noqa: E402
from scenarios.backends import BackendError, make_backend       # noqa: E402
from scenarios.corpus import (                                  # noqa: E402
    EXPRESSIBLE, load_corpus, register_contrib_providers, resolve_views,
)
from scenarios.drift import compare_masks, scenario_masks        # noqa: E402
from scenarios.rollouts import (                                 # noqa: E402
    build_policy_env, collect_trajectories, plain_env,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    help="provider[:model], e.g. anthropic:claude-opus-5")
    ap.add_argument("--corpus", default="benchmark/behaviours")
    ap.add_argument("--root", default=".")
    ap.add_argument("--domain", default=None, help="restrict to one env id")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out", default=None, help="write per-entry JSON here")
    ap.add_argument("--rollouts", type=int, default=0,
                    help="episodes per domain; enables the non-vacuity gate "
                         "and semantic agreement scoring")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--rollout-seed", type=int, default=1000)
    ap.add_argument("--checkpoint", default=None,
                    help="policy for rollouts; without one actions are "
                         "random and rare behaviours never appear")
    ap.add_argument("--env-overrides", default=None,
                    help="hydra overrides for --checkpoint, comma separated")
    args = ap.parse_args()

    try:
        backend = make_backend(args.backend)
    except BackendError as exc:
        raise SystemExit(
            f"cannot build backend: {exc}\n"
            "This evaluation is only meaningful against a real model, so it "
            "will not fall back to a scripted one.")

    register_contrib_providers()
    entries = [e for e in load_corpus(args.corpus) if e.verdict == EXPRESSIBLE]
    if args.domain:
        entries = [e for e in entries if e.domain == args.domain]
    if not entries:
        raise SystemExit("no formalised corpus entries matched")

    root = Path(args.root)
    views = resolve_views([e.domain for e in entries])

    # Rollouts per domain, collected once and shared by every entry in it.
    trajs_by_domain = {}
    if args.rollouts:
        for domain in sorted({e.domain for e in entries}):
            view = views.get(domain)
            if view is None:
                continue
            try:
                if args.checkpoint:
                    env, policy = build_policy_env(
                        args.env_overrides, args.checkpoint)
                else:
                    env, policy = plain_env(domain), None
                trajs_by_domain[domain] = collect_trajectories(
                    env, view, n_episodes=args.rollouts,
                    max_steps=args.max_steps, seed=args.rollout_seed,
                    policy=policy)
            except Exception as exc:                          # noqa: BLE001
                print(f"  (no rollouts for {domain}: {exc})")

    rows = []
    for entry in entries:
        view = views.get(entry.domain)
        trajs = trajs_by_domain.get(entry.domain)
        result = author_scenario(
            entry.description, backend, view=view, env_id=entry.domain,
            trajectories=trajs, max_attempts=args.max_attempts)
        row = {
            "id": entry.id, "domain": entry.domain,
            "accepted": result.accepted, "attempts": result.n_attempts,
            "repaired": result.repaired,
            "rejections": [a.rejection for a in result.attempts],
            "agreement": None,
        }
        # Does the authored scenario MEAN what the reference means?
        if result.accepted and trajs:
            try:
                ref = entry.load_scenario(root, view=view)
                ref_masks = scenario_masks(ref, trajs)
                got_masks = scenario_masks(result.scenario, trajs)
                rep = compare_masks(ref_masks, got_masks)
                ref_rate = float(
                    sum(int(m.sum()) for m in ref_masks)
                    / max(1, sum(len(m) for m in ref_masks)))
                row["agreement"] = 1.0 - rep.disagreement
                row["jaccard"] = rep.jaccard
                row["ref_rate"] = ref_rate
                row["vacuous"] = ref_rate in (0.0, 1.0)
            except Exception as exc:                       # noqa: BLE001
                row["reference_error"] = str(exc)
        rows.append(row)
        status = "ok " if result.accepted else "FAIL"
        extra = ""
        if row.get("agreement") is not None:
            extra = (f"  agree={row['agreement']:.3f} "
                     f"jacc={row['jaccard']:.3f} ref_rate={row['ref_rate']:.3f}"
                     + ("  VACUOUS" if row["vacuous"] else ""))
        print(f"{status} {entry.id:<34} attempts={result.n_attempts}"
              f"{'  (repaired)' if result.repaired else ''}{extra}")
        if not result.accepted and result.attempts:
            for reason in result.attempts[-1].rejection[:2]:
                print(f"       {reason}")

    n = len(rows)
    acc = sum(r["accepted"] for r in rows)
    rep = sum(r["repaired"] for r in rows)
    first = sum(1 for r in rows if r["accepted"] and r["attempts"] == 1)
    print("\n" + "-" * 60)
    print(f"backend        : {args.backend}")
    print(f"entries        : {n}")
    print(f"accepted       : {acc}/{n}  ({acc / n:.1%})")
    print(f"  first try    : {first}")
    print(f"  after repair : {rep}")
    print("\nThe repair delta is the value of the feedback loop: accepted "
          "minus first-try is what the validator's diagnostics bought.")

    scored = [r for r in rows
              if r.get("agreement") is not None and not r.get("vacuous")]
    vac = [r for r in rows
           if r.get("agreement") is not None and r.get("vacuous")]
    if scored:
        import statistics as st
        print(f"\nsemantic agreement (n={len(scored)} informative entries)")
        print(f"  mean step agreement : "
              f"{st.mean(r['agreement'] for r in scored):.3f}")
        print(f"  mean jaccard        : "
              f"{st.mean(r['jaccard'] for r in scored):.3f}")
        print("  Jaccard is the honest one: agreement is inflated by the "
              "many steps where neither spec fires.")
    if vac:
        print(f"\n{len(vac)} accepted entr{'y' if len(vac) == 1 else 'ies'} "
              "excluded as VACUOUS (the reference never varies on these "
              "rollouts, so any spec agrees with it).")
    if not args.rollouts:
        print("\nNo --rollouts: acceptance is gate-passing only and is an "
              "upper bound on authoring quality.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
