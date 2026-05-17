"""Command-line interface for the scenario language.

Subcommands:

  validate   — schema + semantic + lint check on one or more YAML files
  summary    — pretty-print a scenario + its natural-language and LaTeX
               predicate-logic renderings
  list       — list all known scenarios (relative paths)
  coverage   — compute empirical match stats against a trajectories.pkl
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

    p_sum = sub.add_parser("summary", help="Show prose + predicate for a scenario.")
    p_sum.add_argument("scenario", help="Scenario file path or relative name "
                       "(e.g. cartpole/left_only).")
    p_sum.add_argument("--latex-only", action="store_true",
                       help="Only print the LaTeX predicate (good for piping).")
    p_sum.add_argument("--prose-only", action="store_true",
                       help="Only print the natural-language form.")

    p_lst = sub.add_parser("list", help="List all known scenarios.")
    p_lst.add_argument("--root", default="configs/scenarios",
                       help="Scenario directory root.")

    p_cov = sub.add_parser("coverage",
                           help="Empirical match stats against trajectories.pkl.")
    p_cov.add_argument("scenario", help="Scenario file path or relative name.")
    p_cov.add_argument("--against", required=True,
                       help="Path to trajectories.pkl produced by train.py.")

    args = parser.parse_args(argv)

    if args.cmd == "validate":
        return _cmd_validate(args)
    if args.cmd == "summary":
        return _cmd_summary(args)
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "coverage":
        return _cmd_coverage(args)
    parser.error(f"Unknown command: {args.cmd}")
    return 2


# ---------------------------------------------------------------------------


def _cmd_validate(args) -> int:
    expanded = _expand_files(args.files)
    if not expanded:
        print("No scenario files matched.", file=sys.stderr)
        return 2

    aggregate = ValidationReport()
    per_file_summary: List[tuple[str, ValidationReport]] = []
    for f in expanded:
        report = validate_scenario(f)
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
