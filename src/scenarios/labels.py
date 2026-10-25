"""Reference labels: what a corpus entry is measured against.

A corpus entry currently carries a verdict its own author assigned. That
makes expressibility a statement about the author's confidence, not about
the language, and it is the weakest part of the benchmark: the person who
wrote the grammar also decided whether the grammar captured each
behaviour.

Reference labels break that circularity. Someone who did not write the
scenario marks, step by step, where a behaviour occurs. The specification
is then scored against those marks with the same machinery that measures
specification drift, because "does this spec agree with a human" and
"do these two specs agree" are the same computation.

Nothing here generates labels. A label produced by the same process that
produced the specification measures nothing, so the file this module
loads has to be filled in by a person.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

UNLABELLED = None


class LabelError(ValueError):
    """Raised for a malformed or incomplete label file."""


@dataclass
class LabelSet:
    """Per-step boolean labels for the episodes of one corpus entry."""

    entry: str
    description: str
    env_id: str
    episodes: List[np.ndarray]
    annotator: Optional[str] = None

    @property
    def n_steps(self) -> int:
        return int(sum(len(e) for e in self.episodes))

    @property
    def positive_rate(self) -> float:
        if not self.n_steps:
            return 0.0
        return float(sum(int(e.sum()) for e in self.episodes) / self.n_steps)

    def as_masks(self) -> List[np.ndarray]:
        return list(self.episodes)


def load_label_file(path: Path | str) -> LabelSet:
    """Read a completed label file.

    Refuses a file with any step still unlabelled: a partially labelled
    set silently biases agreement towards whatever the blank default is.
    """
    path = Path(path)
    if not path.exists():
        raise LabelError(f"label file not found: {path}")
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict):
        raise LabelError(f"{path}: top level must be a mapping")
    for key in ("entry", "episodes"):
        if key not in doc:
            raise LabelError(f"{path}: missing required key {key!r}")

    episodes: List[np.ndarray] = []
    missing = 0
    for ep in doc["episodes"]:
        steps = ep.get("steps") or []
        vals = []
        for st in steps:
            lab = st.get("label", UNLABELLED)
            if lab is UNLABELLED:
                missing += 1
                vals.append(False)
            else:
                vals.append(bool(lab))
        episodes.append(np.asarray(vals, dtype=bool))

    if missing:
        raise LabelError(
            f"{path}: {missing} step(s) still have label: null. A partly "
            "labelled file biases agreement toward the blank default; "
            "finish it or drop those episodes.")

    if doc.get("description_changed_since_labelling"):
        # Not fatal: the labels are still a record of a real judgement. But
        # the sentence the annotator answered is no longer the sentence the
        # corpus asks, so fidelity computed from them measures agreement
        # with a description that has since been edited.
        warnings.warn(
            f"{path}: the entry description changed after these labels were "
            "written; they answer the text in 'description', not the current "
            "one in 'description_current'.", stacklevel=2)

    return LabelSet(
        entry=doc["entry"],
        description=doc.get("description", ""),
        env_id=doc.get("env", ""),
        episodes=episodes,
        annotator=doc.get("annotator"),
    )


def resolve_labels(entry, root: Path | str = ".") -> Optional[LabelSet]:
    """Load the label file an entry's `labels:` field points at."""
    ref = getattr(entry, "labels_ref", None)
    if not ref:
        return None
    return load_label_file(Path(root) / ref)


def agreement(labels: LabelSet, masks: Sequence[np.ndarray]) -> Dict[str, float]:
    """Per-step agreement between reference labels and a spec's masks."""
    if len(masks) != len(labels.episodes):
        raise LabelError(
            f"{len(masks)} predicted episodes but {len(labels.episodes)} "
            "labelled ones")
    tp = fp = fn = tn = 0
    for ref, pred in zip(labels.episodes, masks):
        if len(ref) != len(pred):
            raise LabelError(
                f"episode length mismatch: labelled {len(ref)}, "
                f"predicted {len(pred)}")
        tp += int(np.sum(ref & pred))
        fp += int(np.sum(~ref & pred))
        fn += int(np.sum(ref & ~pred))
        tn += int(np.sum(~ref & ~pred))
    n = tp + fp + fn + tn
    inter, union = tp, tp + fp + fn
    return {
        "n_steps": float(n),
        "accuracy": (tp + tn) / n if n else 0.0,
        "jaccard": inter / union if union else 1.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "label_rate": (tp + fn) / n if n else 0.0,
        "pred_rate": (tp + fp) / n if n else 0.0,
    }
