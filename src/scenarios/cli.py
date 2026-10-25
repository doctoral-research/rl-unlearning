"""Command-line interface for the scenario language.

Subcommands:

  validate   — schema + semantic + lint check on one or more YAML files
  summary    — pretty-print a scenario + its natural-language and LaTeX
               predicate-logic renderings
  list       — list all known scenarios (relative paths)
  collect    — roll out a policy and pickle the trajectories that
               coverage, drift, mine and author --against read
  coverage   — compute empirical match stats against those trajectories
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .introspect import coverage, to_natural_language, to_predicate
from .scenario import ForgetScenario, list_scenarios
from .validate import ValidationReport, validate_scenario


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scenarios",
        description="ForgetScenario v1.0 language tooling.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate one or more scenarios.")
    p_val.add_argument("files", nargs="+", help="Files or glob patterns.")
    p_val.add_argument("--quiet", "-q", action="store_true",
                       help="Only report errors (suppress warnings/info).")
    p_val.add_argument("--summary", action="store_true",
                       help="Print a one-line per-file summary at the end.")
    p_val.add_argument("--view", default="none", metavar="PROVIDER",
                       help="Bind a feature view before validating, enabling "
                            "dtype and range checks: 'auto' to resolve from "
                            "env_id, a provider name, or 'none' (default). "
                            "'auto' instantiates the environment, so it is "
                            "slower and needs the env's dependencies.")

    p_sum = sub.add_parser("summary", help="Show prose + predicate for a scenario.")
    p_sum.add_argument("scenario", help="Scenario file path or relative name "
                       "(e.g. cartpole/left_only).")
    p_sum.add_argument("--latex-only", action="store_true",
                       help="Only print the LaTeX predicate (good for piping).")
    p_sum.add_argument("--prose-only", action="store_true",
                       help="Only print the natural-language form.")

    p_dom = sub.add_parser(
        "domains",
        help="Report which feature view each environment resolves to.")
    p_dom.add_argument("env_ids", nargs="*",
                       help="Environment ids (default: all registered).")
    p_dom.add_argument("--markdown", action="store_true",
                       help="Emit a markdown table for the paper.")

    p_drift = sub.add_parser(
        "drift",
        help="Measure whether the label spec and the metric spec agree.")
    p_drift.add_argument("label", help="Specification driving the update.")
    p_drift.add_argument("metric", nargs="?",
                         help="Specification used for the reported metric. "
                              "Omit to sweep index offsets on LABEL alone.")
    p_drift.add_argument("--against", required=True, metavar="TRAJ.pkl",
                         help="Pickled list of trajectories.")
    p_drift.add_argument("--metric-offset", type=int, default=0,
                         help="State index the metric reads relative to the "
                              "label: 1 = the state the agent landed in.")
    p_drift.add_argument("--offsets", default="0,1,2",
                         help="Offsets for the sweep form (default 0,1,2).")

    p_mine = sub.add_parser(
        "mine",
        help="Synthesise a scenario from flagged and unflagged rollouts.")
    p_mine.add_argument("positive", help="Pickled flagged trajectories.")
    p_mine.add_argument("negative", help="Pickled unflagged trajectories.")
    p_mine.add_argument("--env", dest="env_id", default=None,
                        help="Environment id, used to resolve a feature view.")
    p_mine.add_argument("--name", default="mined")
    p_mine.add_argument("--max-rules", type=int, default=3)
    p_mine.add_argument("--max-literals", type=int, default=3)
    p_mine.add_argument("--out", default=None,
                        help="Write the mined scenario YAML here.")

    p_auth = sub.add_parser(
        "author",
        help="Build the authoring prompt for a behaviour, or check a "
             "candidate against the acceptance gates.")
    p_auth.add_argument("description", help="The behaviour, in plain words.")
    p_auth.add_argument("--env", dest="env_id", default=None,
                        help="Environment id, used to resolve a feature view.")
    p_auth.add_argument("--check", metavar="FILE", default=None,
                        help="Score a candidate scenario instead of printing "
                             "the prompt. Prints the feedback a repair loop "
                             "would send back.")
    p_auth.add_argument("--against", metavar="TRAJ.pkl", default=None,
                        help="Rollouts for the empirical gate (rejects a "
                             "candidate that fires on nothing).")

    p_corp = sub.add_parser(
        "corpus", help="Score the behaviour corpus.")
    p_corp.add_argument("directory", nargs="?", default="benchmark/behaviours",
                        help="Corpus root (default: benchmark/behaviours).")
    p_corp.add_argument("--verbose", "-v", action="store_true",
                        help="List every entry, not just the summary.")
    p_corp.add_argument("--gaps", action="store_true",
                        help="Show what the grammar cannot yet state.")

    p_lst = sub.add_parser("list", help="List all known scenarios.")
    p_lst.add_argument("--root", default="configs/scenarios",
                       help="Scenario directory root.")

    p_col = sub.add_parser(
        "collect",
        help="Roll out a policy and pickle the trajectories that coverage, "
             "drift, mine and author --against read.")
    p_col.add_argument("env_id", help="Environment id, e.g. CartPole-v1.")
    p_col.add_argument("--out", required=True, metavar="TRAJ.pkl")
    p_col.add_argument("--checkpoint", default=None,
                       help="Policy checkpoint. Without it actions are random "
                            "and rare behaviours may never appear.")
    p_col.add_argument("--env-overrides", default=None,
                       help="Comma-separated Hydra overrides the checkpoint "
                            "was trained with (default: env=<the env config "
                            "whose env_id matches>).")
    p_col.add_argument("--episodes", type=int, default=50)
    p_col.add_argument("--max-steps", type=int, default=500)
    p_col.add_argument("--seed", type=int, default=0)

    p_cov = sub.add_parser("coverage",
                           help="Empirical match stats against trajectories.pkl.")
    p_cov.add_argument("scenario", help="Scenario file path or relative name.")
    p_cov.add_argument("--against", required=True,
                       help="Pickled trajectories, as written by `collect`.")

    args = parser.parse_args(argv)

    if args.cmd == "validate":
        return _cmd_validate(args)
    if args.cmd == "summary":
        return _cmd_summary(args)
    if args.cmd == "mine":
        import pickle

        import yaml as _yaml

        from .mining import MiningError, mine_scenario

        view = _view_for(args.env_id, "auto") if args.env_id else None
        with open(args.positive, "rb") as fh:
            pos = pickle.load(fh)
        with open(args.negative, "rb") as fh:
            neg = pickle.load(fh)
        try:
            scenario, report = mine_scenario(
                pos, neg, view=view, name=args.name,
                max_rules=args.max_rules, max_literals=args.max_literals)
        except MiningError as exc:
            print(f"cannot mine: {exc}", file=sys.stderr)
            return 2

        print(report.render())
        print()
        print("recovered predicate:")
        for g in scenario.groups:
            parts = []
            for c in g.get("conditions", []):
                nm = c.get("dim_name") or c.get("dim")
                parts.append(f"{nm} {c['op']} {c['value']:g}")
            print("  " + " AND ".join(parts))
        print()
        print(to_natural_language(scenario))

        if args.out:
            with open(args.out, "w") as fh:
                _yaml.safe_dump(scenario._config, fh, sort_keys=False,
                                allow_unicode=True, width=76)
            print()
            print(f"wrote {args.out}")
        return 0

    if args.cmd == "author":
        import pickle

        from .authoring import author_scenario, build_prompt

        view = _view_for(args.env_id, "auto") if args.env_id else None
        trajs = None
        if args.against:
            with open(args.against, "rb") as fh:
                trajs = pickle.load(fh)

        if args.check is None:
            print(build_prompt(args.description, view=view,
                               env_id=args.env_id))
            return 0

        with open(args.check) as fh:
            candidate = fh.read()
        result = author_scenario(
            args.description, backend=lambda _p: candidate, view=view,
            env_id=args.env_id, trajectories=trajs, max_attempts=1)
        attempt = result.attempts[0]
        if result.accepted:
            print(f"ACCEPTED  {args.check}")
            print(f"  scenario   : {result.scenario.name}")
            print(f"  match_mode : {result.scenario.match_mode}")
            print(f"  applies_to : {result.scenario.applies_to}")
            if attempt.match_rate is not None:
                print(f"  match rate : {attempt.match_rate:.4f}")
            return 0
        print(f"REJECTED  {args.check}")
        for r in attempt.rejection:
            print(f"  - {r}")
        print()
        print("Feed these back with the candidate to repair it.")
        return 1

    if args.cmd == "corpus":
        from pathlib import Path as _P

        from .corpus import (
            EXPRESSIBLE, INEXPRESSIBLE, PARTIAL, load_corpus, resolve_views,
            score_corpus,
        )

        root = _P(__file__).resolve().parent.parent.parent
        entries = load_corpus(args.directory)
        views = resolve_views([e.domain for e in entries])
        missing = [d for d, v in views.items() if v is None]
        report = score_corpus(entries, root, views_by_domain=views)
        if missing:
            print(f"(no view for: {', '.join(missing)})")

        if args.verbose:
            mark = {EXPRESSIBLE: "OK  ", PARTIAL: "~   ", INEXPRESSIBLE: "X   "}
            width = max(len(s.entry.id) for s in report.scores)
            for sc in report.scores:
                flag = "!" if sc.error else " "
                print(f"{mark[sc.entry.verdict]}{flag} "
                      f"{sc.entry.id.ljust(width)}  {sc.entry.domain}")
                if sc.error:
                    print(f"       ERROR {sc.error}")
            print()

        if args.gaps:
            print("---- what the grammar cannot state ----")
            for sc in report.scores:
                if sc.entry.verdict == EXPRESSIBLE:
                    continue
                print(f"[{sc.entry.verdict}] {sc.entry.id}")
                print(f"  {sc.entry.description.strip()}")
                for line in (sc.entry.gap or "").strip().splitlines():
                    print(f"    {line}")
                print()

        print("---- corpus ----")
        print(report.render())
        print()
        per_dom = report.by_domain()
        print(f"{'domain':32s} {'n':>3s}  verdicts")
        for dom in sorted(per_dom):
            scs = per_dom[dom]
            e = sum(1 for x in scs if x.entry.verdict == EXPRESSIBLE)
            pa = sum(1 for x in scs if x.entry.verdict == PARTIAL)
            ix = sum(1 for x in scs if x.entry.verdict == INEXPRESSIBLE)
            print(f"{dom:32s} {len(scs):>3d}  {e} expressible, "
                  f"{pa} partial, {ix} inexpressible")
        return 1 if report.broken else 0

    if args.cmd == "drift":
        import pickle

        from .drift import measure_drift, offset_sweep

        with open(args.against, "rb") as fh:
            trajs = pickle.load(fh)
        label = ForgetScenario.load(_resolve_scenario_path(args.label))

        if args.metric is None:
            offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
            print(f"offset sweep for {label.name!r} over {len(trajs)} "
                  f"trajectories")
            print(f"{'offset':>7s} {'disagree':>9s} {'jaccard':>8s} "
                  f"{'kappa':>8s} {'delta':>8s}")
            worst = 0.0
            for r in offset_sweep(label, trajs, offsets=offsets):
                print(f"{r.metric_offset:>+7d} {r.disagreement:>9.4f} "
                      f"{r.jaccard:>8.4f} {r.kappa:>8.4f} "
                      f"{r.metric_delta:>8.4f}")
                worst = max(worst, r.metric_delta)
            print()
            print(f"worst metric delta from index offset alone: {worst:.4f}")
            return 0

        metric = ForgetScenario.load(_resolve_scenario_path(args.metric))
        report = measure_drift(label, metric, trajs,
                               metric_offset=args.metric_offset)
        print(f"label   {label.name!r}  ({args.label})")
        print(f"metric  {metric.name!r}  ({args.metric})")
        print()
        print(report.render())
        return 0

    if args.cmd == "domains":
        from .constants import ENV_OBS_DIMS
        from .views import domain_report

        env_ids = args.env_ids or sorted(ENV_OBS_DIMS)
        rows = domain_report(env_ids)
        ok = [r for r in rows if r["error"] is None]

        if args.markdown:
            print("| Domain | Provenance | Features | Adapter LOC |")
            print("|---|---|---|---|")
            for r in rows:
                prov = r["provenance"] or f"FAILED ({r['error']})"
                n = r["n_features"] if r["n_features"] is not None else "-"
                loc = r["adapter_loc"] if r["adapter_loc"] is not None else "-"
                print(f"| `{r['env_id']}` | {prov} | {n} | {loc} |")
        else:
            w = max((len(r["env_id"]) for r in rows), default=10)
            print(f"{'DOMAIN'.ljust(w)}  {'PROVENANCE':18s} {'FEAT':>4s} "
                  f"{'LOC':>4s}  NAMES")
            print("-" * (w + 40))
            for r in rows:
                if r["error"]:
                    print(f"{r['env_id'].ljust(w)}  FAILED  {r['error'][:60]}")
                    continue
                names = ", ".join(r["names"][:4])
                if len(r["names"]) > 4:
                    names += f", ... (+{len(r['names']) - 4})"
                loc = r["adapter_loc"] if r["adapter_loc"] is not None else "-"
                print(f"{r['env_id'].ljust(w)}  {r['provenance']:18s} "
                      f"{r['n_features']:>4d} {str(loc):>4s}  {names}")

        auto = sum(1 for r in ok if r["provenance"].startswith("auto:"))
        decl = sum(1 for r in ok if r["provenance"].startswith("declared:"))
        print()
        print(f"{len(ok)}/{len(rows)} domains resolved: "
              f"{auto} automatic, {decl} declared.")
        return 0

    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "collect":
        return _cmd_collect(args)
    if args.cmd == "coverage":
        return _cmd_coverage(args)
    parser.error(f"Unknown command: {args.cmd}")
    return 2


# ---------------------------------------------------------------------------


def _view_for(env_id, prefer):
    """Resolve a view for validation, or None if unavailable.

    Failures are non-fatal: validation without a view is still useful, it
    just skips the dtype and range checks.
    """
    if not env_id or prefer in (None, "none"):
        return None
    from .corpus import register_contrib_providers
    from .views import build_view

    # Optional domain adapters (e.g. multi-agent) must be registered before
    # the cascade runs, or a view-dependent scenario cannot be validated.
    register_contrib_providers()

    env = None
    try:
        import gymnasium as gym

        if env_id.startswith("MiniGrid-"):
            import minigrid  # noqa: F401
        elif env_id.startswith("Foraging-"):
            import lbforaging  # noqa: F401
        env = gym.make(env_id)
        obs, info = env.reset(seed=0)
        shape = getattr(env.observation_space, "shape", None)
        n_dims = int(shape[0]) if shape and len(shape) == 1 else None
        return build_view(
            env=env, env_id=env_id, n_dims=n_dims,
            prefer=None if prefer == "auto" else prefer,
            probe={"obs": obs, "info": info},
        )
    except Exception as exc:
        print(f"  (no view for {env_id}: {type(exc).__name__}: {exc})",
              file=sys.stderr)
        return None
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _cmd_validate(args) -> int:
    expanded = _expand_files(args.files)
    if not expanded:
        print("No scenario files matched.", file=sys.stderr)
        return 2

    aggregate = ValidationReport()
    per_file_summary: List[tuple[str, ValidationReport]] = []
    view_cache: dict = {}
    for f in expanded:
        view = None
        if getattr(args, "view", "none") != "none":
            import yaml

            try:
                with open(f) as fh:
                    env_id = (yaml.safe_load(fh) or {}).get("env_id")
            except Exception:
                env_id = None
            if env_id not in view_cache:
                view_cache[env_id] = _view_for(env_id, args.view)
            view = view_cache[env_id]
        report = validate_scenario(f, view=view)
        per_file_summary.append((f, report))
        for issue in report.issues:
            if args.quiet and issue.severity != "error":
                continue
            print(issue.render())
        aggregate = aggregate.merge(report)

    if args.summary:
        print()
        print("---- summary ----")
        for f, r in per_file_summary:
            status = (
                "OK" if r.ok and not r.warnings
                else "OK (with warnings)" if r.ok
                else "FAIL"
            )
            print(f"  {status:18}  {f}")
        print(f"  total:  {len(per_file_summary)} files, "
              f"{len(aggregate.errors)} errors, "
              f"{len(aggregate.warnings)} warnings")

    return 0 if aggregate.ok else 1


def _cmd_summary(args) -> int:
    path = _resolve_scenario_path(args.scenario)
    scenario = ForgetScenario.load(str(path))
    if args.latex_only:
        print(to_predicate(scenario))
        return 0
    if args.prose_only:
        print(to_natural_language(scenario))
        return 0

    print("=" * 78)
    print(f"  {scenario.name}  ({scenario.env_id or 'no env_id'})")
    print("=" * 78)
    print()
    print("Description:")
    for line in (scenario.description or "").strip().splitlines():
        print(f"  {line}")
    print()
    print("Natural language:")
    print(f"  {to_natural_language(scenario)}")
    print()
    print("Predicate logic (LaTeX):")
    print(f"  {to_predicate(scenario)}")
    print()
    print("Structure:")
    # Skip the first two lines (name + description block) — those are
    # rendered explicitly above. Also de-indent the leading two spaces
    # that summary() adds internally so our outer indent isn't doubled.
    structure_lines = scenario.summary().splitlines()
    description_lines = (scenario.description or "").strip().splitlines()
    skip = 1 + max(len(description_lines), 1)
    for line in structure_lines[skip:]:
        print(f"  {line[2:] if line.startswith('  ') else line}")
    print()
    return 0


def _cmd_list(args) -> int:
    for name in list_scenarios(args.root):
        print(name)
    return 0


def _cmd_collect(args) -> int:
    from .corpus import resolve_views
    from .rollouts import build_policy_env, collect_trajectories, plain_env

    # Features are extracted while the episode runs, because a view may read
    # the environment object, which no longer matches an observation once
    # the next step is taken.
    view = resolve_views([args.env_id]).get(args.env_id)
    if view is None:
        print(f"No feature view resolves for {args.env_id}.", file=sys.stderr)
        return 2
    if args.checkpoint:
        config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
        overrides = args.env_overrides or _env_override_for(args.env_id, config_dir)
        if overrides is None:
            print(f"No config in {config_dir / 'env'} has env_id {args.env_id}; "
                  "pass --env-overrides.", file=sys.stderr)
            return 2
        env, policy = build_policy_env(overrides, args.checkpoint,
                                       config_dir=str(config_dir))
    else:
        env, policy = plain_env(args.env_id), None
    trajectories = collect_trajectories(
        env, view, n_episodes=args.episodes, max_steps=args.max_steps,
        seed=args.seed, policy=policy)
    with open(args.out, "wb") as f:
        pickle.dump(trajectories, f)
    steps = sum(len(t["actions"]) for t in trajectories)
    print(f"Wrote {len(trajectories)} episodes ({steps} steps) to {args.out}")
    return 0


def _env_override_for(env_id: str, config_dir: Path) -> Optional[str]:
    """The `env=` override whose config declares this env_id, if any."""
    import yaml

    for f in sorted((config_dir / "env").glob("*.yaml")):
        if (yaml.safe_load(f.read_text()) or {}).get("env_id") == env_id:
            return f"env={f.stem}"
    return None


def _cmd_coverage(args) -> int:
    path = _resolve_scenario_path(args.scenario)
    scenario = ForgetScenario.load(str(path))
    traj_path = Path(args.against)
    if not traj_path.exists():
        print(f"Trajectories not found: {traj_path}", file=sys.stderr)
        return 2
    with open(traj_path, "rb") as f:
        trajectories = pickle.load(f)
    report = coverage(scenario, trajectories)
    print(report.render(scenario))
    return 0


# ---------------------------------------------------------------------------


def _expand_files(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        if any(ch in p for ch in "*?["):
            out.extend(sorted(glob.glob(p, recursive=True)))
        else:
            out.append(p)
    # Filter to YAML only (so users can pass a directory glob safely).
    return [f for f in out if f.endswith((".yaml", ".yml"))]


def _resolve_scenario_path(name: str) -> Path:
    """Accept either a file path or a `cartpole/left_only` shorthand."""
    if name.endswith((".yaml", ".yml")):
        return Path(name)
    candidate = Path("configs/scenarios") / f"{name}.yaml"
    if candidate.exists():
        return candidate
    pkg_candidate = (
        Path(__file__).resolve().parent.parent.parent
        / "configs" / "scenarios" / f"{name}.yaml"
    )
    if pkg_candidate.exists():
        return pkg_candidate
    raise FileNotFoundError(
        f"Cannot resolve scenario '{name}'. Tried: {candidate}, {pkg_candidate}"
    )


if __name__ == "__main__":
    sys.exit(main())
