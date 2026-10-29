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
pdflatex -interaction=nonstopmode -recorder main.tex | grep -E "Output written|^!" || true

if grep -qiE "undefined (citation|reference)" main.log; then
    echo "!! undefined references -- see main.log"
    exit 1
fi
if grep -q "Overfull" main.log; then
    echo "!! overfull boxes:"; grep "Overfull" main.log
fi

echo "== overleaf kit =="
# overleaf-kit.zip holds exactly the files this compile read from the folder
# (the recorder's list) and the two bibtex reads, and nothing that only
# regenerates them. It is compiled once more on its own before it is kept.
python3 - <<'EOF'
import re, subprocess, sys, tempfile, zipfile
from pathlib import Path

files = set()
for line in Path("main.fls").read_text().splitlines():
    if line.startswith("INPUT "):
        p = line[6:].removeprefix("./")
        if not p.startswith("/") and Path(p).suffix not in {".aux", ".out", ".bbl"}:
            files.add(p)
blg = Path("main.blg").read_text()
files |= set(re.findall(r"Database file #\d+: (\S+)", blg))
files |= {s if s.endswith(".bst") else s + ".bst"
          for s in re.findall(r"The style file: (\S+)", blg)}

with zipfile.ZipFile("overleaf-kit.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(files):
        z.write(f)

with tempfile.TemporaryDirectory() as d:
    zipfile.ZipFile("overleaf-kit.zip").extractall(d)
    run = lambda *c: subprocess.run(c, cwd=d, capture_output=True, text=True)
    for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):
        run(*cmd, *(["-interaction=nonstopmode", "main.tex"] if cmd[0] == "pdflatex" else ["main"]))
    log = Path(d, "main.log").read_text(errors="replace")
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    if not pages or re.search(r"undefined (citation|reference)", log, re.I):
        sys.exit("!! overleaf-kit.zip does not compile on its own")
print(f"overleaf-kit.zip: {len(files)} files, compiles alone to {pages.group(1)} pages")
EOF

# Keep the folder clean: only main.pdf and the kit stay. Overleaf runs its
# own build.
rm -f main.aux main.bbl main.blg main.log main.out main.fls
rm -rf figures/__pycache__ data/__pycache__
