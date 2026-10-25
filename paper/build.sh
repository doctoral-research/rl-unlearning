#!/bin/bash
# Rebuild everything the paper quotes, then compile it.
#
# This folder is the Overleaf project: main.tex, the style files, the
# bibliography, figures/*.pdf and generated/*.tex are all it needs to
# compile. The Python scripts in data/ and figures/ regenerate the numbers,
# figures and generated sections from the experiment results; Overleaf
# ignores them.
set -euo pipefail
cd "$(dirname "$0")"

echo "== data =="
# Everything the text and figures quote is recomputed from the label files,
# the corpus and the training histories. The policy-divergence JSONs are
# inputs here: they take minutes to produce and come from
# scripts/drift_policy_divergence.py and scripts/drift_label_disagreement.py.
python data/make_fidelity_table.py > /dev/null
python data/make_reader_tables.py > /dev/null
python data/make_drift_data.py > /dev/null
python data/make_examples.py > /dev/null

echo "== figures =="
for f in figures/fig*.py; do python "$f"; done

echo "== generated sections =="
python data/make_appendix.py

echo "== latex =="
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null || true
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex | grep -E "Output written|^!" || true

if grep -qiE "undefined (citation|reference)" main.log; then
    echo "!! undefined references -- see main.log"
    exit 1
fi
if grep -q "Overfull" main.log; then
    echo "!! overfull boxes:"; grep "Overfull" main.log
fi

# Keep the folder clean: only main.pdf stays. Overleaf runs its own build.
rm -f main.aux main.bbl main.blg main.log main.out
rm -rf figures/__pycache__ data/__pycache__
