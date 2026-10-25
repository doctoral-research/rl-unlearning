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
lists any overfull box.

The policy-divergence files in `data/divergence_*.json` and
`data/drift_label_disagreement.json` take minutes to produce, so they are
inputs to the build rather than part of it. They come from
`scripts/drift_policy_divergence.py` (arms `ab_ms` against `ab_wt`,
`ab_n1`, `ab_n10` in FourRooms and `cp_ms` against `cp_wt`, `cp_n1` in
CartPole) and `scripts/drift_label_disagreement.py`.

## Where the numbers come from

Nothing in the text, the figures or the appendix is typed by hand.

| script | produces | from |
| --- | --- | --- |
| `data/make_fidelity_table.py` | fidelity per behaviour (kappa, Jaccard, precision, recall) | `benchmark/labels/*.yaml`, the corpus formalisations |
| `data/make_reader_tables.py` | reader agreement, the polarity table | the label files and the raw reader labels in `benchmark/labels/` |
| `data/make_drift_data.py` | label-scope experiment and its noise controls | `data/divergence_*.json`, training histories |
| `data/make_examples.py` | `generated/examples.tex`, `generated/appendix_language.tex` | corpus scenarios, resolved views, the validator |
| `data/make_appendix.py` | `generated/appendix.tex`, `generated/appendix_rl.tex` | all of the above, `benchmark/labels/prompts/` |
| `figures/fig_*.py` | `figures/fig_*.pdf` | `data/measurements.json`, `data/probe_predictions.json`, training histories |

## Syncing with Overleaf

With Overleaf's Git integration, push this folder from the repository root
and pull edits made on Overleaf back the same way:

```bash
git remote add overleaf https://git.overleaf.com/<project-id>
git subtree push --prefix paper overleaf master
git subtree pull --prefix paper overleaf master --squash
```

Without it, upload the folder as a zip ("New Project", then "Upload
Project") and re-upload the files that change.

## Before submitting

- **Human subset.** Being collected. Once in, report human against reader
  and human against formalisation agreement, and replace the sentence in
  the limitations that says it has not been collected.
- **Release.** Confirmed: code, corpus, labels and prompts will be
  released. Add the repository link, or attach an anonymised copy as
  supplementary material, as the venue requires.
- **Bibliography.** The entries added for the related-work expansion were
  checked against the publisher records. The older entries have not been
  rechecked.
