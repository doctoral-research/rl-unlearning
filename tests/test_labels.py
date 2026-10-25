"""Reference labels: the independent ground truth a corpus entry is
scored against.

The corpus's weakest point is that its verdicts were assigned by the
person who wrote the grammar. These tests cover the loader that lets
someone else's labels into the scoring path, and in particular its
refusal to accept a half-finished file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scenarios.labels import LabelError, agreement, load_label_file


def write(tmp_path: Path, doc) -> Path:
    p = tmp_path / "labels.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def complete_doc():
    return {
        "entry": "x.y", "env": "Env-v0", "description": "d",
        "annotator": "someone-else",
        "episodes": [
            {"id": 0, "steps": [{"t": 0, "label": True},
                                {"t": 1, "label": False}]},
            {"id": 1, "steps": [{"t": 0, "label": False}]},
        ],
    }


class TestLoading:
    def test_loads_episodes_as_bool_arrays(self, tmp_path):
        ls = load_label_file(write(tmp_path, complete_doc()))
        assert [list(e) for e in ls.episodes] == [[True, False], [False]]

    def test_keeps_the_annotator(self, tmp_path):
        assert load_label_file(write(tmp_path, complete_doc())).annotator \
            == "someone-else"

    def test_counts_steps_and_positive_rate(self, tmp_path):
        ls = load_label_file(write(tmp_path, complete_doc()))
        assert ls.n_steps == 3
        assert ls.positive_rate == pytest.approx(1 / 3)

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(LabelError, match="not found"):
            load_label_file(tmp_path / "absent.yaml")

    def test_missing_required_key_is_an_error(self, tmp_path):
        with pytest.raises(LabelError, match="episodes"):
            load_label_file(write(tmp_path, {"entry": "x"}))


class TestRefusesPartialWork:
    """A blank label silently reads as 'behaviour absent', which biases
    agreement toward whatever the default is. Better to refuse."""

    def test_rejects_unlabelled_steps(self, tmp_path):
        doc = complete_doc()
        doc["episodes"][0]["steps"][1]["label"] = None
        with pytest.raises(LabelError, match="still have label: null"):
            load_label_file(write(tmp_path, doc))

    def test_error_counts_the_blanks(self, tmp_path):
        doc = complete_doc()
        for ep in doc["episodes"]:
            for st in ep["steps"]:
                st["label"] = None
        with pytest.raises(LabelError, match="3 step"):
            load_label_file(write(tmp_path, doc))

    def test_missing_key_counts_as_unlabelled(self, tmp_path):
        doc = complete_doc()
        del doc["episodes"][1]["steps"][0]["label"]
        with pytest.raises(LabelError, match="1 step"):
            load_label_file(write(tmp_path, doc))


class TestAgreement:
    def test_perfect_agreement(self, tmp_path):
        ls = load_label_file(write(tmp_path, complete_doc()))
        got = agreement(ls, ls.as_masks())
        assert got["accuracy"] == 1.0 and got["jaccard"] == 1.0

    def test_disagreement_is_measured(self, tmp_path):
        ls = load_label_file(write(tmp_path, complete_doc()))
        flipped = [~e for e in ls.episodes]
        got = agreement(ls, flipped)
        assert got["accuracy"] == 0.0
        assert got["recall"] == 0.0

    def test_reports_both_rates(self, tmp_path):
        """Label rate and predicted rate separate 'wrong region' from
        'right region, wrong size'."""
        ls = load_label_file(write(tmp_path, complete_doc()))
        pred = [np.array([True, True]), np.array([False])]
        got = agreement(ls, pred)
        assert got["label_rate"] == pytest.approx(1 / 3)
        assert got["pred_rate"] == pytest.approx(2 / 3)

    def test_episode_count_mismatch_is_an_error(self, tmp_path):
        ls = load_label_file(write(tmp_path, complete_doc()))
        with pytest.raises(LabelError, match="predicted episodes"):
            agreement(ls, ls.as_masks()[:1])

    def test_episode_length_mismatch_is_an_error(self, tmp_path):
        ls = load_label_file(write(tmp_path, complete_doc()))
        bad = [np.array([True]), np.array([False])]
        with pytest.raises(LabelError, match="length mismatch"):
            agreement(ls, bad)


class TestStaleDescription:
    """Labels answer the sentence the annotator was shown.

    Editing a corpus description after labelling does not invalidate the
    judgement, but it does mean the labels no longer answer the question
    the corpus now asks. That has to be visible rather than silent.
    """

    def _doc(self):
        d = complete_doc()
        d["description"] = "Balance only while the cart is on the left."
        d["description_current"] = "Stop balancing while the cart is on the right."
        d["description_changed_since_labelling"] = True
        return d

    def test_warns_when_the_description_moved(self, tmp_path):
        with pytest.warns(UserWarning, match="description changed"):
            load_label_file(write(tmp_path, self._doc()))

    def test_labels_still_load(self, tmp_path):
        with pytest.warns(UserWarning):
            ls = load_label_file(write(tmp_path, self._doc()))
        assert ls.n_steps == 3

    def test_no_warning_when_unchanged(self, tmp_path):
        import warnings as w
        with w.catch_warnings():
            w.simplefilter("error")
            load_label_file(write(tmp_path, complete_doc()))
