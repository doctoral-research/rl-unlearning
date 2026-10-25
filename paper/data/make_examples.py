"""Generate the language examples for the paper from the shipped files.

Writes, under paper/generated/,
  examples.tex            the figure of example formalisations in Section 3
  appendix_language.tex   the feature views of every environment, every
                          formalisation in the corpus next to its
                          description, and the validator's messages on a
                          set of faulty scenarios

Listings are the corpus files themselves, re-serialised compactly with
comments and the `name`, `description`, `version` and `env_id` fields
dropped. Each rendering is parsed back and compared with the source, so a
listing cannot differ from the file it shows. Validator messages are
produced by running the validator, not typed.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import yaml

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT / "src"))
from scenarios.corpus import (load_corpus, resolve_views,          # noqa: E402
                              register_contrib_providers, EXPRESSIBLE,
                              PARTIAL, INEXPRESSIBLE)
from scenarios.validate import validate_scenario_dict              # noqa: E402

warnings.filterwarnings("ignore")
GEN = RT / "paper/generated"
GEN.mkdir(exist_ok=True)
register_contrib_providers()
ENTRIES = sorted(load_corpus(str(RT / "benchmark/behaviours")), key=lambda e: e.id)
BYID = {e.id: e for e in ENTRIES}
MEAS = json.loads((RT / "paper/data/measurements.json").read_text())
FID = {r["entry"]: r for r in MEAS["corpus"]["fidelity"]}
DROP = ("name", "description", "version", "env_id")
VERDICT = {EXPRESSIBLE: "expressible", PARTIAL: "partial", INEXPRESSIBLE: "inexpressible"}


# ------------------------------------------------------------ YAML output
class _Flow(dict):
    pass


class _FlowList(list):
    pass


class _Q(str):
    pass


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(_Flow, lambda d, x: d.represent_mapping(
    "tag:yaml.org,2002:map", x.items(), flow_style=True))
_Dumper.add_representer(_FlowList, lambda d, x: d.represent_sequence(
    "tag:yaml.org,2002:seq", x, flow_style=True))
_Dumper.add_representer(_Q, lambda d, x: d.represent_scalar(
    "tag:yaml.org,2002:str", str(x), style='"'))

def _scalar(v):
    return not isinstance(v, (dict, list))


def _prep(x, key=None, depth=0, budget=48):
    """Flow style for a small mapping or list whose whole line, indentation
    and key included, fits within `budget` characters. Block style
    otherwise. Comparison operators and expressions are double-quoted, as
    in the source files."""
    if isinstance(x, dict):
        d = {k: _prep(v, k, depth + 1, budget) for k, v in x.items()}
        if all(_scalar(v) or (isinstance(v, list) and all(map(_scalar, v)))
               for v in x.values()):
            flow = _Flow(d)
            text = yaml.dump(flow, Dumper=_Dumper, width=10_000).strip()
            lead = 2 * depth + (len(str(key)) + 2 if key is not None else 2)
            if lead + len(text) <= budget:
                return flow
        return d
    if isinstance(x, list):
        items = [_prep(v, None, depth, budget) for v in x]
        if all(map(_scalar, x)):
            return _FlowList(items)
        return items
    if isinstance(x, str) and ((key == "op" and any(c in x for c in "<>=!"))
                               or key == "expr" or "{i}" in x
                               or any(c in x for c in "+-*/()")):
        return _Q(x)
    return x


def render(doc, limit=48):
    out = yaml.dump(_prep(doc, budget=limit), Dumper=_Dumper, sort_keys=False,
                    width=10_000, default_flow_style=False).rstrip()
    assert yaml.safe_load(out) == doc, "rendering changed the document"
    return out


def scenario_doc(entry):
    for root in (RT / "benchmark/scenarios", RT / "configs/scenarios"):
        f = root / f"{entry.scenario_ref}.yaml"
        if f.exists():
            break
    raw = yaml.safe_load(f.read_text())
    return {k: v for k, v in raw.items() if k not in DROP}, f.relative_to(RT)


def esc(t):
    t = " ".join((t or "").split())
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
                 ("}", r"\}")):
        t = t.replace(a, b)
    return t


def tt(s):
    return f"\\texttt{{{esc(s)}}}"


LST = r"\begin{lstlisting}[language=,aboveskip=2pt,belowskip=2pt]"

# ---------------------------------------------------------- main-text figure
# (entry, letter, width as a fraction of \textwidth, top-level key to omit,
# font size in pt). The two side-by-side panels use a slightly smaller size
# so that their longest lines, which are quoted expressions and cannot be
# broken, fit without wrapping.
PANELS = [
    ("lbf.no_crowding", "a", 0.53, None, 6.5),
    ("fourrooms.no_backtracking", "b", 0.45, None, 6.5),
    ("fourrooms.no_corner_turn_reflex", "c", 0.90, "groups", 7.0),
]


def panel(eid, letter, width, omit, size):
    e = BYID[eid]
    doc, _ = scenario_doc(e)
    if omit:
        doc = {k: v for k, v in doc.items() if k != omit}
    # Characters per line of the typewriter face (0.6 em wide) in this
    # width, less the listing's left margin.
    limit = int(width * 396 / (0.6 * size)) - 2
    lst = (r"\begin{lstlisting}[language=,aboveskip=2pt,belowskip=2pt,"
           rf"basicstyle=\ttfamily\fontsize{{{size}}}{{{size + 1.1:.1f}}}\selectfont]")
    return "\n".join([
        f"\\begin{{minipage}}[t]{{{width}\\textwidth}}",
        f"{{\\footnotesize\\textbf{{({letter})}} {esc(e.domain)}\\\\"
        f"\\textit{{``{esc(e.description)}''}}}}",
        lst, render(doc, limit), r"\end{lstlisting}",
        r"\end{minipage}"])


fig = [r"\begin{figure}[t]", r"\centering",
       panel(*PANELS[0]), r"\hfill", panel(*PANELS[1]), r"\\[6pt]",
       panel(*PANELS[2]),
       r"\caption{Three formalisations from the corpus, each under the English "
       r"description it was written for. \textbf{(a)} A derived feature that "
       r"quantifies over the members of a family of agents, so one file serves "
       r"any team size. \textbf{(b)} Two episode aggregates combined by "
       r"arithmetic. \textbf{(c)} A nested temporal formula. The agent turns left "
       r"inside the room, and was outside it one or two steps earlier, so the "
       r"label falls on the turn at the crossing and not on later turns. "
       r"Features are named, as in (a) and (b), or indexed with a "
       r"\texttt{dim\_name} that the validator checks against the bound view, as "
       r"in (c). Comments and the \texttt{name} and \texttt{description} fields "
       r"are omitted, and (c) also omits a state-level group that serves "
       r"evaluation only. Appendix~\ref{app:gallery} lists every formalisation.}",
       r"\label{fig:examples}", r"\end{figure}"]
(GEN / "examples.tex").write_text("\n".join(fig) + "\n")

# ------------------------------------------------------------- views table
A: list[str] = []
domains = sorted({e.domain for e in ENTRIES})
views = resolve_views(domains)
A.append(r"\section{Feature views}" + "\n" + r"\label{app:views}" + "\n")
A.append(
    "Table~\\ref{tab:views} lists the view each environment of the corpus "
    "resolves to, with its provenance and the names a scenario may use. The "
    "MiniGrid and MuJoCo views are introspected from the environment, the "
    "level-based foraging view from its \\texttt{info} and environment "
    "state, and the classic-control and Box2D views come from hand-registered "
    "tables. Feature types are " + ", ".join(sorted({t for v in views.values()
                                                    for t in v.dtypes}))
    + ".\n")
A.append(r"\begin{table}[h]\centering\footnotesize")
A.append(r"\caption{Resolved feature views. MiniGrid's three environments share one view.}")
A.append(r"\label{tab:views}")
A.append(r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.5cm}lr>{\raggedright\arraybackslash}p{6.4cm}@{}}\toprule")
A.append(r"Environment & Provenance & $n$ & Features \\ \midrule")
seen = {}
for d in domains:
    v = views[d]
    key = (v.provenance, tuple(v.names))
    if key in seen and v.provenance == "auto:minigrid":
        continue
    seen[key] = d
    name = ("MiniGrid: Empty, FourRooms, DoorKey"
            if v.provenance == "auto:minigrid" else esc(d))
    A.append(f"{name} & {tt(v.provenance)} & {len(v.names)} & "
             + ", ".join(tt(n) for n in v.names) + r" \\")
A.append(r"\bottomrule\end{tabular}\end{table}" + "\n")

# ---------------------------------------------------------------- gallery
A.append(r"\section{Every formalisation}" + "\n" + r"\label{app:gallery}" + "\n")
formalised = [e for e in ENTRIES if e.scenario_ref]
in_set = [e for e in formalised if e.id in FID]
A.append(
    f"The {len(in_set)} formalisations of the corpus, each under its "
    "description, environment and verdict, with $\\kappa$ against the "
    "reference labels. Listings are the scenario files with comments "
    "and the \\texttt{name}, \\texttt{description}, \\texttt{version} and "
    "\\texttt{env\\_id} fields removed. Two behaviours, "
    "\\texttt{lunarlander.no\\_near\\_miss} and \\texttt{lunarlander.no\\_tilt}, "
    "share one file: the near miss is approximated by the tilt.\n")


def gallery_item(e):
    doc, path = scenario_doc(e)
    k = FID.get(e.id)
    kv = 0.0 if k and abs(k["kappa"]) < 0.005 else (k["kappa"] if k else None)
    kap = f", $\\kappa={kv:.2f}$" if k else ", never observed"
    A.append(r"\par\medskip\noindent" + f"{tt(e.id)} ({esc(e.domain)}, "
             f"{VERDICT[e.verdict]}{kap})\\\\")
    A.append(f"\\textit{{``{esc(e.description)}''}}\\nopagebreak")
    A.append(LST)
    A.append(render(doc, limit=92))
    A.append(r"\end{lstlisting}")


for e in in_set:
    gallery_item(e)
A.append("")

# ------------------------------------------------------ validator messages
fr = views["MiniGrid-FourRooms-v0"]
BASE = {"name": "demo", "description": "Stop entering the bottom-right room.",
        "env_id": "MiniGrid-FourRooms-v0"}


def cond(dim, op, value):
    return {"conditions": [{"dim": dim, "op": op, "value": value}]}


CASES = [
    ("A feature the view does not have",
     dict(match_mode="any_step", applies_to="matching_steps",
          groups=[cond("agent_position", ">", 9.5)])),
    ("Ordering a categorical feature",
     dict(match_mode="any_step", applies_to="matching_steps",
          groups=[cond("direction", ">", 1)])),
    ("A threshold outside the feature's range",
     dict(match_mode="any_step", applies_to="matching_steps",
          groups=[cond("agent_x", ">", 25)])),
    ("Contradictory conditions",
     dict(match_mode="any_step", applies_to="matching_steps",
          groups=[{"conditions": [{"dim": "agent_x", "op": ">", "value": 10},
                                  {"dim": "agent_x", "op": "<", "value": 5}]}])),
    ("An episode aggregate under a step-level mode",
     dict(match_mode="proportion", match_threshold=0.2, applies_to="matching_steps",
          aggregates={"cells": {"op": "count_distinct", "of": ["agent_x", "agent_y"]}},
          groups=[cond("cells", "<", 10)])),
    ("A trajectory-level mode without a declared scope",
     dict(match_mode="proportion", match_threshold=0.2,
          groups=[cond("agent_x", ">", 9.5)])),
    ("An existential formula labelling steps",
     dict(match_mode="temporal", applies_to="matching_steps",
          temporal={"eventually": cond("agent_x", ">", 9.5)})),
]
A.append(r"\section{Validator messages}" + "\n" + r"\label{app:validator}" + "\n")
A.append(
    "Each faulty scenario below is a minimal FourRooms scenario bound to the "
    f"\\texttt{{{fr.provenance}}} view, with one defect. The messages are the "
    "validator's output, verbatim. Informational notes are omitted.\n")
A.append(r"\begin{itemize}\setlength{\itemsep}{3pt}")
for title, extra in CASES:
    doc = dict(BASE, **extra)
    rep = validate_scenario_dict(doc, view=fr)
    issues = [i for i in rep.issues if i.severity in ("error", "warning")]
    assert issues, title
    shown = {k: v for k, v in extra.items()}
    A.append(f"\\item \\emph{{{title}.}}")
    A.append(LST)
    A.append(render(shown, limit=92))
    A.append(r"\end{lstlisting}")
    for i in issues:
        # Typewriter, because these are tool output and because roman text
        # in the default OT1 encoding has no < or > glyphs.
        A.append(f"\\textbf{{{i.severity}}} \\texttt{{{esc(i.code)}}}:\\\\"
                 f"{{\\raggedright\\footnotesize\\ttfamily {esc(i.message)}\\par}}")
A.append(r"\end{itemize}" + "\n")
(GEN / "appendix_language.tex").write_text("\n".join(A) + "\n")
print("wrote examples.tex and appendix_language.tex")
