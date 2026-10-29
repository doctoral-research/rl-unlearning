# Specifying What to Forget (ICLR 2026 submission)

This folder is the Overleaf project. It compiles as it stands with
`main.tex` as the main document. There is no separate export.

```
main.tex              the paper
references.bib        bibliography
iclr2026_conference.sty / .bst, fancyhdr.sty, natbib.sty
                      the official ICLR 2026 template files
generated/            appendix sections and the example figure, written by
                      data/make_*.py. Do not edit them by hand: the next
                      build overwrites them.
figures/              fig_*.pdf used by the paper, each next to the script
                      that draws it (style.py is shared)
data/                 measurements.json and the other recorded numbers,
                      with the scripts that compute them
build.sh              regenerate everything and compile
```

`\iclrfinalcopy` is commented out, so the PDF carries the anonymous
"Under review" header and line numbers.

## Building

```bash
./build.sh
```

It recomputes the numbers from the label files, the corpus and the training
histories, redraws the figures, regenerates `generated/`, compiles, and
deletes the LaTeX build leftovers. It stops if a reference is undefined and
lists any overfull box. It also writes `overleaf-kit.zip`: the 14 files the
compile reads (`main.tex`, the bibliography and style files, `figures/*.pdf`,
`generated/*.tex`) and nothing else, after checking that they compile on
their own.

Three kinds of file take minutes or a local model to produce, so they are
inputs to the build rather than part of it:

- `data/divergence_*.json` and `data/drift_label_disagreement.json` come
  from `scripts/drift_policy_divergence.py` (arms `ab_ms` against `ab_wt`,
  `ab_n1`, `ab_n10` in FourRooms and `cp_ms` against `cp_wt`, `cp_n1` in
  CartPole) and `scripts/drift_label_disagreement.py`.
- `data/authoring_eval.json` holds the validator stress test, run with
  `scripts/run_authoring_eval.py` against two Qwen2.5-Coder models served by
  ollama. The file records the commands.

Every experiment starts from the baseline that
`utils.checkpointing.baseline_checkpoint` names: each seed's final
checkpoint, except FourRooms seed 42, which uses its best.

## Where the numbers come from

Nothing in the text, the figures or the appendix is typed by hand.

| script | produces | from |
| --- | --- | --- |
| `data/make_fidelity_table.py` | fidelity per behaviour (kappa, Jaccard, precision, recall), and the person's agreement with the reference and the formalisations | `benchmark/labels/*.yaml`, `benchmark/labels/human/`, the corpus formalisations |
| `data/make_reader_tables.py` | reader agreement, the polarity table | the label files and the raw reader labels in `benchmark/labels/` |
| `data/make_drift_data.py` | label-scope experiment and its noise controls | `data/divergence_*.json`, training histories |
| `data/make_examples.py` | `generated/examples.tex`, `generated/appendix_language.tex` | corpus scenarios, resolved views, the validator |
| `data/make_appendix.py` | `generated/appendix.tex`, `generated/appendix_rl.tex` | all of the above, `benchmark/labels/prompts/`, `data/authoring_eval.json` |
| `figures/fig_*.py` | `figures/fig_*.pdf` | `data/measurements.json`, `data/probe_predictions.json`, training histories |

## Syncing with Overleaf

With Overleaf's Git integration, push this folder from the repository root
and pull edits made on Overleaf back the same way:

```bash
git remote add overleaf https://git.overleaf.com/<project-id>
git subtree push --prefix paper overleaf master
git subtree pull --prefix paper overleaf master --squash
```

Without it, upload `overleaf-kit.zip` ("New Project", then "Upload
Project"). After a later build, upload the files that changed, or start
from a fresh kit.

## Before submitting

- **Human labels.** In: one person labelled every step
  (`benchmark/labels/human/annotator_H1.json`). `data/make_fidelity_table.py`
  scores them, and Section 4.3, the limitations, the abstract and the
  fidelity figure and table report them. The numbers Section 4.3 quotes are
  checked by asserts in that script.
- **Release.** Confirmed: code, corpus, labels and prompts will be
  released. Add the repository link, or attach an anonymised copy as
  supplementary material, as the venue requires.
- **Bibliography.** The entries added for the related-work expansion were
  checked against the publisher records. The older entries have not been
  rechecked.
