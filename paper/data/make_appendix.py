"""Generate the data-bearing appendix sections from the files on disk.

Written rather than typed so that the appendix cannot drift from the data.
The corpus listing, both fidelity tables, the reader rounds and prompts,
the polarity pairs, the lint results, the authoring stress test, the method
comparison and the drift controls all come from the objects and files the
experiments use. Static sections (formal semantics, training details) live
in main.tex.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT / "src"))
from scenarios.corpus import (load_corpus, description_voice_ok,   # noqa: E402
                              PROHIBITION_VOICE, EXPRESSIBLE, PARTIAL,
                              INEXPRESSIBLE)

PROMPTS = RT / "benchmark/labels/prompts"
meas = json.loads((RT / "paper/data/measurements.json").read_text())
entries = sorted(load_corpus(str(RT / "benchmark/behaviours")), key=lambda e: e.id)
byid = {e.id: e for e in entries}
fid = {r["entry"]: r for r in meas["corpus"]["fidelity"]}
readers = {r["entry"]: r for r in meas["readers"]["entries"]}
L: list[str] = []


def esc(t):
    t = " ".join((t or "").split())
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        t = t.replace(a, b)
    return t


def tt(eid):
    return f"\\texttt{{{esc(eid)}}}"


def f2(x):
    return "$-$" + f"{abs(x):.2f}" if x < -0.005 else f"{abs(x):.2f}"


def sg(x, d=3):
    """Signed number with a typographic minus, and no sign on a rounded zero."""
    if round(x, d) == 0:
        return f"${0:.{d}f}$"
    return f"${x:+.{d}f}$"


VERDICT = {EXPRESSIBLE: "expressible", PARTIAL: "partial",
           INEXPRESSIBLE: "inexpressible"}

# What each approximate formalisation lacks, stated for the paper. Checked
# against the corpus below so that a verdict change cannot leave a stale row.
GAPS = {
    "fourrooms.no_pacing":
        "A count of a repeated pattern of cells. Approximated by steps per "
        "distinct cell over the episode.",
    "lunarlander.no_near_miss":
        "Distance to a failure that did not happen. Approximated by the "
        "lander's tilt.",
}
# The paper covers the labelled behaviours only.
corpus = [e for e in entries if e.id in fid]
assert len(corpus) == len(fid)
assert set(GAPS) == {e.id for e in corpus if e.verdict != EXPRESSIBLE}

# ------------------------------------------------------------------ corpus
L.append(r"\section{The behaviours}" + "\n" + r"\label{app:corpus}" + "\n")
L.append(
    f"Table~\\ref{{tab:corpus}} lists the {len(corpus)} behaviours with their "
    "environments and verdicts. A behaviour is \\emph{partial} when its "
    "formalisation approximates the description, and the table below it "
    "gives what the grammar lacks in each case.\n")
L.append(r"\begin{table}[h]\centering\footnotesize")
L.append(r"\caption{The behaviours of the corpus.}")
L.append(r"\label{tab:corpus}")
L.append(r"\setlength{\tabcolsep}{4pt}")
L.append(r"\begin{tabular}{@{}lll@{}}\toprule")
L.append(r"Behaviour & Environment & Verdict \\ \midrule")
for e in corpus:
    L.append(f"{tt(e.id)} & {esc(e.domain)} & {VERDICT[e.verdict]} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table}" + "\n")

L.append(r"\begin{center}\footnotesize\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{7.4cm}@{}}\toprule")
L.append(r"Partial behaviour & What the grammar lacks \\ \midrule")
for eid in sorted(GAPS):
    L.append(f"{tt(eid)} & {esc(GAPS[eid])} \\\\")
L.append(r"\bottomrule\end{tabular}\end{center}" + "\n")

L.append(r"\paragraph{Fidelity.}")
L.append(
    "Table~\\ref{tab:fidelity} gives, for each labelled entry, the readers, "
    "the number of labelled steps, the mean pairwise $\\kappa$ between "
    "readers, and the agreement between the formalisation and the reference "
    "labels as $\\kappa$, Jaccard similarity, precision and recall (taking "
    "the reference as ground truth), with the fraction of steps each "
    "labels.\n")
L.append(r"\begin{table}[h]\centering\small")
L.append(r"\caption{Reader agreement and fidelity on the labelled entries. "
         r"$\kappa_{rr}$: mean pairwise $\kappa$ between readers. $\kappa_p$: one "
         r"person's labels against the reference. $\kappa$, $J$, "
         r"P, R: formalisation against the reference. Rates: fraction of steps "
         r"labelled by the reference and by the formalisation.}")
L.append(r"\label{tab:fidelity}")
L.append(r"\setlength{\tabcolsep}{2.8pt}")
L.append(r"\begin{tabular}{@{}llrccccccc@{}}\toprule")
L.append(r"Entry & Readers & Steps & $\kappa_{rr}$ & $\kappa_p$ & $\kappa$ & $J$ & P & R & "
         r"Rates \\ \midrule")
for eid in sorted(fid, key=lambda x: -fid[x]["kappa"]):
    r, q = fid[eid], readers[eid]
    L.append(f"{tt(eid)} & {''.join(q['readers'])} & {r['n_steps']} & "
             f"{f2(q['mean_kappa'])} & {f2(r['person']['kappa_reference'])} & "
             f"{f2(r['kappa'])} & {r['jaccard']:.2f} & "
             f"{r['precision']:.2f} & {r['recall']:.2f} & "
             f"{r['label_rate']:.2f}/{r['spec_rate']:.2f} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table}" + "\n")
L.append(
    "Reader agreement is computed from the readers' raw labels where they "
    "were kept (readers A, B and C). Where two readers produced identical "
    "labels the label file stores that one set, and $\\kappa_{rr}=1$ by "
    "construction. On the doorway entry the three readers marked 12, 18 and "
    "29 steps with pairwise Jaccard similarity 12/18, 12/29 and 18/29, which "
    "forces the three sets to be nested, so their pairwise tables, and "
    "$\\kappa$, follow from the counts. The person labelled every task in "
    "the labelling tool released with the corpus, which shows a neutral task "
    "code instead of the entry's name, and wrote down the rule they "
    "followed for each entry. The labels and the rules are released with the "
    "reference labels.\n")

# ----------------------------------------------------------------- readers
ROUNDS = [
    ("r01_ABC", "neutral", "\\texttt{cartpole.left\\_only} (goal wording, 105 steps), "
     "\\texttt{lbf.no\\_crowding}, \\texttt{lbf.no\\_freeloading}, "
     "\\texttt{lunarlander.no\\_tilt}, \\texttt{walker2d.no\\_forward\\_fall}",
     "reference for the last four; polarity"),
    ("r02_DE", "neutral", "\\texttt{pendulum.avoid\\_left\\_swing}, "
     "\\texttt{fourrooms.avoid\\_br\\_detour} (goal wording)", "polarity"),
    ("r03_FG", "strict", "\\texttt{fourrooms.no\\_pacing}, "
     "\\texttt{lunarlander.no\\_near\\_miss} (180 steps each)",
     "discarded (behaviour absent)"),
    ("r04_HI", "strict", "\\texttt{lunarlander.no\\_near\\_miss}", "reference"),
    ("r05_JK", "strict", "\\texttt{halfcheetah.no\\_front\\_flip}", "reference"),
    ("r06_LM", "strict", "\\texttt{doorkey.no\\_key\\_hoarding}", "reference"),
    ("r07_NO", "strict", "\\texttt{cartpole.left\\_only}, "
     "\\texttt{empty8x8.avoid\\_bottom\\_left}, \\texttt{fourrooms.avoid\\_br\\_detour}, "
     "\\texttt{fourrooms.no\\_corner\\_turn\\_reflex}, "
     "\\texttt{pendulum.avoid\\_left\\_swing} (prohibition wording)", "reference"),
    ("r08_PQ", "strict", "\\texttt{fourrooms.avoid\\_br\\_tr\\_door}, "
     "\\texttt{fourrooms.no\\_backtracking}", "reference"),
    ("r09_RS", "strict", "\\texttt{fourrooms.no\\_pacing}", "reference"),
    ("r10_T", "strict", "\\texttt{fourrooms.avoid\\_br\\_tr\\_door}", "reference (third reader)"),
    ("r11_UV", "strict", "\\texttt{cartpole.left\\_only}, "
     "\\texttt{pendulum.avoid\\_left\\_swing}, \\texttt{fourrooms.avoid\\_br\\_detour} "
     "(goal wording, the steps of round 7)", "polarity"),
]
assert {r[0] for r in ROUNDS} == {p.stem for p in PROMPTS.glob("*.txt")}
L.append(r"\section{Readers}" + "\n" + r"\label{app:readers}" + "\n")
L.append(
    "All readers were language-model instances that we ran from a fresh "
    "context: Claude Opus~5 for rounds 1 to 10 and Claude Opus~5.5 for round "
    "11. Each received one prompt that we wrote and the task files it named. A task file holds the "
    "behaviour description, sometimes a short note on the features (for "
    "example the action codes of MiniGrid, or that its $y$ axis points down), "
    "and a table with one row per step: episode, time step, the feature "
    "values and the action. Readers were told not to open any other file and "
    "wrote their labels with a short script applying the rule they had "
    "chosen. After the last round we checked every proposed label by hand "
    "against its description and feature values (Section~\\ref{sec:readers}). "
    "Table~\\ref{tab:rounds} lists the rounds. An instruction is "
    "\\emph{strict} when it states that the sentence names a behaviour to be "
    "stopped and asks for the steps where it happens, and \\emph{neutral} "
    "otherwise.\n")
L.append(r"\begin{table}[h]\centering\footnotesize")
L.append(r"\caption{Reader rounds. Every round's full prompt is released with the corpus.}")
L.append(r"\label{tab:rounds}")
L.append(r"\setlength{\tabcolsep}{3pt}")
L.append(r"\begin{tabular}{@{}clll>{\raggedright\arraybackslash}p{5.9cm}>{\raggedright\arraybackslash}p{1.9cm}@{}}\toprule")
L.append(r"Round & Readers & Model & Instr. & Entries & Use \\ \midrule")
for name, instr, ents, use in ROUNDS:
    num, who = name[1:].split("_")
    model = "Opus 5.5" if who == "UV" else "Opus 5"
    L.append(f"{int(num)} & {', '.join(who)} & {model} & {instr} & {ents} & {use} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table}" + "\n")


def body(name):
    txt = (PROMPTS / f"{name}.txt").read_text()
    return "\n".join(l for l in txt.splitlines() if not l.startswith("# ")).strip()


def rules(name):
    return [l.strip() for l in body(name).splitlines() if l.startswith("- ")]


L.append(r"\paragraph{Prompts.}")
L.append(
    "Rounds 1, 2, 7 and 11 are reproduced in full below, with "
    "\\texttt{<READER>} standing for the reader's letter. The other strict "
    "rounds follow the template of round 7 and differ from it only in the "
    "file list and in the rules listed after the full prompts.\n")
for name, title in (("r01_ABC", "Round 1 (neutral)"),
                    ("r02_DE", "Round 2 (neutral, polarity left open)"),
                    ("r07_NO", "Round 7 (strict)"),
                    ("r11_UV", "Round 11 (strict)")):
    L.append(f"\\noindent\\textit{{{title}.}}")
    L.append(r"\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,literate={—}{{---}}1]")
    L.append(body(name))
    L.append(r"\end{lstlisting}")
base = set(rules("r07_NO"))
L.append(r"\noindent\textit{Rules of the remaining strict rounds that are not in round 7.}")
L.append(r"\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,literate={—}{{---}}1]")
for name in ("r03_FG", "r04_HI", "r05_JK", "r06_LM", "r08_PQ", "r09_RS", "r10_T"):
    extra = [r for r in rules(name) if r not in base]
    L.append(f"Round {int(name[1:3])}:")
    L.extend(f"  {r}" for r in extra)
L.append(r"\end{lstlisting}" + "\n")

L.append(r"\paragraph{Polarity.}")
pol = meas["readers"]["polarity"]
cells = meas["readers"]["polarity_cells"]
L.append(
    "Table~\\ref{tab:polarity-pairs} lists every reader--entry pair on a "
    "goal-voice description, with $\\kappa$ against the formalisation before "
    "and after flipping the reader's labels. Readers D and E were scored when "
    "they wrote their labels (Jaccard 0.00, and 1.00 after flipping, on both "
    "entries), and their raw labels were not kept. On prohibition-voice "
    "descriptions under the neutral instruction, reader B labelled 41, 78, 89 "
    "and 55 per cent of steps on its four entries where readers A and C "
    "labelled 11 to 35 per cent, the complement of the other two "
    "($\\kappa$ with reader A between $-0.24$ and $-0.54$). No reader under a "
    f"strict instruction marked the complement ({cells['prohibition|strict']['pairs']} "
    "pairs).\n")
L.append(r"\begin{table}[h]\centering\small")
L.append(r"\caption{Reader--entry pairs on goal-voice descriptions.}")
L.append(r"\label{tab:polarity-pairs}")
L.append(r"\begin{tabular}{@{}lllcc@{}}\toprule")
L.append(r"Reader & Entry & Instruction & $\kappa$ & $\kappa$ flipped \\ \midrule")
for p in pol:
    if p["voice"] != "goal":
        continue
    k = "--" if p["kappa"] is None else f2(p["kappa"])
    kf = "--" if p["kappa_flipped"] is None else f2(p["kappa_flipped"])
    L.append(f"{p['reader']} & {tt(p['entry'])} & {p['instruction']} & {k} & {kf} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table}" + "\n")

# -------------------------------------------------------------------- lint
ORIGINAL = {
    "cartpole.left_only": "Balance the pole only while the cart stays on the left half of the track.",
    "pendulum.avoid_left_swing": "Swing up using only the right half of the circle.",
    "fourrooms.avoid_br_detour": "Reach the goal without routing through the bottom-right room.",
}
INVERTED = {p["entry"] for p in pol if p["voice"] == "goal" and p["inverted"]}
flag_orig = [e.id for e in corpus
             if not description_voice_ok(ORIGINAL.get(e.id, e.description))]
flag_now = [e.id for e in corpus if not description_voice_ok(e.description)]
assert sorted(flag_orig) == sorted(ORIGINAL) and not flag_now
labelled_descr = len(fid) + len(INVERTED)
L.append(r"\section{Description lint}" + "\n" + r"\label{app:lint}" + "\n")
L.append(
    "The lint passes a description when it opens with a prohibition. The "
    "rule is the regular expression\n")
L.append(r"\begin{lstlisting}[basicstyle=\ttfamily\scriptsize]")
L.append(PROHIBITION_VOICE.pattern)
L.append(r"\end{lstlisting}")
L.append(
    f"matched case-insensitively at the start of the description. Over the "
    f"original wording of the behaviours it flags {len(flag_orig)} "
    f"descriptions, listed below with their rewrites, and it flags none of "
    f"the current ones. Readers have labelled "
    f"{labelled_descr} distinct descriptions: the {len(fid)} current "
    f"descriptions of the labelled entries and the original wording of "
    f"{len(INVERTED)} of the flagged ones. Taking a description as defective "
    "when every reader of it marked the complement of the formalisation, the "
    f"lint flags all {len(INVERTED)} defective descriptions and none of the "
    f"other {len(fid)}.\n")
L.append(r"\begin{center}\footnotesize\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{3.9cm}>{\raggedright\arraybackslash}p{3.9cm}@{}}\toprule")
L.append(r"Entry & Original wording (flagged) & Current wording \\ \midrule")
for eid in sorted(ORIGINAL):
    L.append(f"{tt(eid)} & {esc(ORIGINAL[eid])} & {esc(byid[eid].description)} \\\\")
L.append(r"\bottomrule\end{tabular}\end{center}" + "\n")

# --------------------------------------------------------------- authoring
au = json.loads((RT / "paper/data/authoring_eval.json").read_text())
gate, roll = au["gate_only"], au["with_rollouts"]
MODELS = ("7b", "32b")
SPELL = {2: "two", 4: "four", 13: "thirteen"}
accepted = [r for m in MODELS for r in roll[m] if r["accepted"]]
# The text below states three facts about the run. Check them, so a rerun
# that changes them fails here instead of leaving the prose wrong.
for m in MODELS:
    fr = {r["id"] for r in roll[m]}
    assert ({r["id"] for r in gate[m] if r["accepted"] and r["id"] in fr}
            == {r["id"] for r in roll[m] if r["accepted"]})
assert not any(r["vacuous"] for r in accepted)
assert all("YAML" in x or "schema" in x or "properties" in x
           for m in MODELS for rows in (gate[m], roll[m]) for r in rows
           if not r["accepted"] for att in r["rejections"] for x in att)


def rate(rows):
    return 100 * sum(r["accepted"] for r in rows) / len(rows)


L.append(r"\section{Validator stress test}" + "\n" + r"\label{app:authoring}" + "\n")
L.append(
    "Two open code models, Qwen2.5-Coder 7B and 32B served locally, were "
    "given an English behaviour description, the names, types and ranges "
    "of the bound view's features, and a summary of the grammar, and asked "
    f"for a scenario. We used the {SPELL[len(gate['7b'])]} descriptions that "
    "the grammar states exactly. A rejected scenario was returned to the "
    "model with the validator's messages, for up to three attempts in all. "
    "Without rollouts the validator runs its schema and view stages only. "
    "With rollouts it also rejects scenarios that label no step or every "
    f"step, which we ran on the {SPELL[len(roll['7b'])]} FourRooms entries "
    "with 25 rollouts of 60 steps from the seed-42 baseline.\n")
L.append(r"\begin{center}\small\begin{tabular}{@{}lccc@{}}\toprule")
L.append(r"Validator stages & Entries & 7B accepted & 32B accepted \\ \midrule")
L.append(f"schema and view & {len(gate['7b'])} & "
         f"{rate(gate['7b']):.0f}\\% & {rate(gate['32b']):.0f}\\% \\\\")
L.append(f"schema, view and rollouts & {len(roll['7b'])} & "
         f"{rate(roll['7b']):.0f}\\% & {rate(roll['32b']):.0f}\\% \\\\")
L.append(r"\bottomrule\end{tabular}\end{center}")
ag = [r["agreement"] for r in accepted]
jc = [r["jaccard"] for r in accepted]
L.append(
    "Every rejected scenario either failed to parse or broke the schema. The "
    "empirical stage rejected nothing further: each model had the same "
    "FourRooms scenarios accepted with and without rollouts, and none of "
    f"them was vacuous. The {SPELL[len(accepted)]} accepted scenarios agreed "
    f"with our formalisations on {min(ag) * 100:.0f}\\% to "
    f"{max(ag) * 100:.0f}\\% of steps, yet their Jaccard similarity with "
    f"them lay between {min(jc):.2f} and {max(jc):.2f}. Both label few steps, "
    "and rarely the same ones.\n")

# -------------------------------------------------------------- comparison
KEY = "eval/forget_region_fraction"
CONTROL = "control_nopressure"
ORDER = [("noop_disjoint", "disjoint objective"),
         ("is_replay_disjoint", "IS-replay + disjoint objective"),
         ("npo", "NPO"), ("kl_neggrad", "NegGrad + KL anchor"),
         ("neggrad", "NegGrad"), ("neg_reward_1.0", r"reward penalty $\lambda=1.0$"),
         ("neg_reward_0.1", r"reward penalty $\lambda=0.1$")]
fin, ret, start = {}, {}, {}
labels = [CONTROL] + [o for o, _ in ORDER]
for p in glob.glob(str(RT / "experiments/outputs/online_unlearn_fourrooms_seed*_*/history.json")):
    name = Path(p).parent.name
    lab = next((x for x in sorted(labels, key=len, reverse=True) if name.endswith("_" + x)), None)
    if not lab:
        continue
    seed = int(re.search(r"seed(\d+)", name).group(1))
    h = json.load(open(p))
    fin.setdefault(lab, {})[seed] = [x[KEY] for x in h if KEY in x][-1]
    start.setdefault(lab, {})[seed] = [x[KEY] for x in h if KEY in x][0]
    ret.setdefault(lab, {})[seed] = [x["eval/mean_return"] for x in h if "eval/mean_return" in x][-1]
SPLIT = len(L)  # what follows depends on the training details in main.tex
L.append(r"\section{Method comparison against a no-pressure control}" + "\n"
         + r"\label{app:comparison}" + "\n")
seeds_c = sorted(fin[CONTROL])


def mean_of(d):
    return float(np.mean(list(d.values())))


def p_vs_control(lab):
    s = sorted(set(fin[lab]) & set(fin[CONTROL]))
    return float(stats.ttest_rel([fin[lab][x] for x in s],
                                 [fin[CONTROL][x] for x in s]).pvalue)


# The paragraph states these about the runs. Check them, so a rerun that
# changes them fails here instead of leaving the prose wrong.
rose = {lab for lab in labels if mean_of(fin[lab]) > mean_of(start[lab])}
assert CONTROL in rose and len(rose & {o for o, _ in ORDER}) == 6
assert abs(mean_of(fin["neg_reward_0.1"]) - mean_of(start["neg_reward_0.1"])) < 0.03
assert [o for o, _ in ORDER if p_vs_control(o) < 0.05] == ["neg_reward_0.1"]
assert 7 * p_vs_control("neg_reward_0.1") > 0.05
L.append(
    "In FourRooms, with the target \\texttt{avoid\\_bottom\\_right}, the "
    "control and six of the seven methods end with higher mean forget-region "
    "occupancy than they started with, and the reward penalty at "
    "$\\lambda=0.1$ ends where it started. Return rises over the same period "
    "and the shortest route to the "
    "goal passes through the region, so occupancy rises with competence "
    "whether or not anything is unlearned. The control that separates the "
    "two keeps the same scenario but uses its labels for evaluation only: "
    "continued PPO with no unlearning loss, no reward change and no masking. "
    "Figure~\\ref{fig:control} and Table~\\ref{tab:comparison} compare each "
    f"method with it on seeds {seeds_c[0]} to {seeds_c[-1]}, paired by seed. "
    "Only the reward penalty at $\\lambda=0.1$ excludes zero, and with seven "
    "comparisons it does not survive a Bonferroni correction. We report the "
    "comparison as an illustration of what the control changes, not as a "
    "ranking of methods.\n")
L.append(r"\begin{figure}[h]\centering")
L.append(r"\includegraphics[width=\textwidth]{figures/fig_control.pdf}")
L.append(r"\caption{\textbf{(a)} Forget-region occupancy during online "
         r"unlearning, mean over five seeds with a $95\%$ CI band. The control "
         r"rises with the others, because occupancy tracks task "
         r"competence. \textbf{(b)} Change against the control at the final "
         r"update, paired by seed, with $95\%$ intervals. Small dots are seeds.}")
L.append(r"\label{fig:control}\end{figure}")
L.append(r"\begin{table}[h]\centering\small")
L.append(r"\caption{Final forget-region occupancy and return against the "
         r"no-pressure control, paired by seed. $p$: paired $t$-test, "
         r"uncorrected. $p_{\mathrm{Bonf}}$: multiplied by seven.}")
L.append(r"\label{tab:comparison}")
L.append(r"\begin{tabular}{@{}lccccc@{}}\toprule")
L.append(r"Method & Seeds & $\Delta$ occupancy & $p$ & $p_{\mathrm{Bonf}}$ & $\Delta$ return \\ \midrule")
for lab, pretty in ORDER:
    s = sorted(set(fin[lab]) & set(fin[CONTROL]))
    d = np.array([fin[lab][x] - fin[CONTROL][x] for x in s])
    dr = np.array([ret[lab][x] - ret[CONTROL][x] for x in s])
    pv = float(stats.ttest_rel([fin[lab][x] for x in s], [fin[CONTROL][x] for x in s]).pvalue)
    L.append(f"{pretty} & {len(s)} & {sg(d.mean())} & {pv:.3f} & {min(1.0, 7 * pv):.2f} & "
             f"{sg(dr.mean())} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table}" + "\n")

# ------------------------------------------------------------------- drift
dr = meas["drift"]
L.append(r"\section{Label scope: per-arm results}" + "\n" + r"\label{app:drift}" + "\n")
L.append(
    "Table~\\ref{tab:drift} gives the numbers behind Figure~\\ref{fig:drift}. "
    "Agreement is measured on 4{,}000 states visited by each seed's baseline, "
    "with action sampling seeded so that the state set is identical across "
    "arms. Occupancy is the fraction of steps in the forget region over 16 "
    "evaluation episodes after the last update. $p$ values are paired $t$-tests "
    "over the fifteen seeds: for agreement, against the other-scope arm, and "
    "for occupancy, against zero.\n")
L.append(r"\begin{table}[h]\centering\small")
L.append(r"\caption{Each arm against the \texttt{matching\_steps} run on the same "
         r"seed (mean, with standard deviation over seeds).}")
L.append(r"\label{tab:drift}")
L.append(r"\begin{tabular}{@{}llccccc@{}}\toprule")
L.append(r"Domain & Arm & Labels changed & Agreement & $p$ & $\Delta$ occupancy & $p$ \\ \midrule")
ARMNAME = {"wt": "other scope", "n1": "1\\% flips", "n10": "10\\% flips"}
for dom, d in dr.items():
    ld = d["label_disagreement"]
    for arm, a in d["arms"].items():
        ag, sh = np.asarray(a["agreement"]), np.asarray(a["occupancy_shift"])
        lc = (f"{100 * ld['mean_disagreement']:.1f}\\%" if arm == "wt"
              else f"{100 * a['label_noise']:.0f}\\%")
        pa = f"{a['p_agreement_vs_spec']:.3f}" if "p_agreement_vs_spec" in a else "--"
        L.append(f"{dom if arm == 'wt' else ''} & {ARMNAME[arm]} & {lc} & "
                 f"{ag.mean():.3f} ({ag.std(ddof=1):.3f}) & {pa} & "
                 f"{sg(sh.mean())} ({sh.std(ddof=1):.3f}) & {a['p_occupancy_shift']:.3f} \\\\")
    b = np.asarray(d["baseline_agreement"])
    L.append(f" & baseline & -- & {b.mean():.3f} ({b.std(ddof=1):.3f}) & -- & -- & -- \\\\")
    if dom != list(dr)[-1]:
        L.append(r"\midrule")
L.append(r"\bottomrule\end{tabular}\end{table}" + "\n")
fr = dr["FourRooms"]
wt, n10 = np.asarray(fr["arms"]["wt"]["occupancy_shift"]), np.asarray(fr["arms"]["n10"]["occupancy_shift"])
L.append(
    "The label change is measured on one 2{,}048-step rollout of each seed's "
    "baseline, split into episodes as in training, and ranges over seeds from "
    f"{100 * fr['label_disagreement']['min']:.1f}\\% to "
    f"{100 * fr['label_disagreement']['max']:.1f}\\% in FourRooms and from "
    f"{100 * dr['CartPole']['label_disagreement']['min']:.1f}\\% to "
    f"{100 * dr['CartPole']['label_disagreement']['max']:.1f}\\% in CartPole. "
    "Compared directly with the 10\\% control, the other scope's occupancy "
    f"shift is larger by {np.mean(wt - n10):.3f} "
    f"($p={stats.ttest_rel(wt, n10).pvalue:.2f}$).\n")

GEN = RT / "paper/generated"
GEN.mkdir(exist_ok=True)
(GEN / "appendix.tex").write_text("\n".join(L[:SPLIT]) + "\n")
(GEN / "appendix_rl.tex").write_text("\n".join(L[SPLIT:]) + "\n")
print(f"wrote appendix.tex and appendix_rl.tex ({len(L)} lines)")
