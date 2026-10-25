"""Resumable training checkpoints.

A checkpoint that holds only `network` is enough to *evaluate* a policy but
not to *continue* training it. Resuming from one silently restarts Adam's
moment estimates, the return-normaliser, and every RNG stream, so the run
that comes out is not the run that would have happened — the training curve
carries a discontinuity at the resume point that is invisible in the saved
artefact.

This module saves everything that carries state across an update:

    network            policy + value parameters
    optimizer          Adam moments; dropping these is the big one
    global_step        so the schedule and step budget continue
    best_eval          so best-so-far tracking is not reset
    ret_rms            return normaliser (Welford state)
    obs_rms            observation normaliser, when enabled
    rng                python / numpy / torch (+ cuda) generator states
    sde_step           SDE noise resample counter

What is deliberately NOT restored
---------------------------------
The environment's internal state. Gym vector envs are not picklable in
general, and MiniGrid/MuJoCo state cannot be round-tripped portably. On
resume the envs are reset with a step-derived seed, so the in-flight
episodes at the crash point are lost and fresh ones begin.

For PPO this is a bounded perturbation — equivalent to truncating at most
`num_envs` episodes once — and it is recorded in the checkpoint as
`env_state_restored: False` so a resumed run is never mistaken for an
uninterrupted one. Everything on the learner side is exact.
"""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Bumped when the payload changes shape. Format 1 was network-only and
# cannot be resumed from; it is still loadable for evaluation.
CHECKPOINT_FORMAT = 2


def save_checkpoint_atomic(payload: Dict[str, Any], path: Path | str) -> Path:
    """Write a checkpoint so an interrupted save cannot corrupt it.

    torch.save straight to the destination leaves a truncated (often
    zero-byte) file if the process dies mid-write, and that file still has
    a perfectly valid name — so checkpoint discovery picks it and resume
    fails on the newest checkpoint rather than falling back. Writing to a
    temporary file in the same directory and renaming makes the swap
    atomic on POSIX: the destination is either the old checkpoint or the
    complete new one, never a partial.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with open(tmp_path, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def load_checkpoint(path: Path | str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load one of our checkpoints.

    Format-2 payloads embed numpy RNG state, which PyTorch 2.6+ refuses
    under its `weights_only=True` default. These files are produced by
    this codebase, so full unpickling is the correct call -- but it has to
    be explicit, and in one place rather than scattered across every
    consumer.
    """
    return torch.load(str(path), map_location=map_location, weights_only=False)


# Experiments start from a run's final checkpoint, except where that
# checkpoint is known to be unusable. FourRooms seed 42 collapsed in its last
# 400k steps: evaluation return fell from about 0.3 to 0, and the final
# policy reaches the goal in 2% of episodes against 66% for the run's best
# checkpoint.
BASELINE_EXCEPTIONS = {("fourrooms", 42): "best_model.pt"}


def baseline_checkpoint(env_name: str, seed: int,
                        root: Path | str = "experiments/checkpoints") -> Path:
    """The trained policy that experiments on `env_name` with `seed` start from."""
    name = BASELINE_EXCEPTIONS.get((env_name, int(seed)), "final_model.pt")
    return Path(root) / f"{env_name}_ppo_seed{seed}" / name


def _rms_state(rms: Any) -> Optional[Dict[str, Any]]:
    if rms is None:
        return None
    return {
        "mean": np.asarray(rms.mean).copy(),
        "var": np.asarray(rms.var).copy(),
        "count": float(rms.count),
    }


def _load_rms(rms: Any, state: Optional[Dict[str, Any]]) -> None:
    if rms is None or not state:
        return
    mean, var = state["mean"], state["var"]
    rms.mean = float(mean) if np.isscalar(rms.mean) or np.ndim(rms.mean) == 0 \
        else np.asarray(mean).copy()
    rms.var = float(var) if np.isscalar(rms.var) or np.ndim(rms.var) == 0 \
        else np.asarray(var).copy()
    rms.count = float(state["count"])


def _rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _load_rng(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
    cuda = state.get("torch_cuda")
    if cuda and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([
                torch.as_tensor(s, dtype=torch.uint8) for s in cuda
            ])
        except Exception:
            # Resuming a CUDA run on a machine with a different GPU count
            # is fine; the CUDA stream just starts fresh.
            pass


def build_checkpoint(
    agent,
    global_step: int,
    *,
    best_eval: float = -float("inf"),
    ret_rms: Any = None,
    obs_rms: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a fully resumable checkpoint payload."""
    payload: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "network": agent.network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "global_step": int(global_step),
        "best_eval": float(best_eval),
        "ret_rms": _rms_state(ret_rms),
        "obs_rms": _rms_state(obs_rms),
        "rng": _rng_state(),
        "sde_step": int(getattr(agent, "_sde_step_counter", 0)),
        "env_state_restored": False,
    }
    if extra:
        payload.update(extra)
    return payload


@dataclass
class ResumeState:
    """What a resumed run picks up from."""

    global_step: int
    best_eval: float
    path: Path
    restored_optimizer: bool
    restored_rng: bool


def is_resumable(payload: Dict[str, Any]) -> bool:
    return (
        isinstance(payload, dict)
        and int(payload.get("format", 1)) >= 2
        and "optimizer" in payload
        and "global_step" in payload
    )


def is_readable_checkpoint(path: Path) -> bool:
    """Whether a checkpoint file can actually be loaded and resumed from."""
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return False
        return is_resumable(
            torch.load(path, map_location="cpu", weights_only=False))
    except Exception:
        return False


def find_latest_checkpoint(
    ckpt_dir: Path, verify: bool = True,
) -> Optional[Path]:
    """Newest USABLE periodic checkpoint in `ckpt_dir`, by step number.

    Ordered by the step encoded in the filename rather than mtime, so a
    stray touch or an out-of-order write cannot pick the wrong one.

    With `verify` (the default) each candidate is opened newest-first and
    the first one that actually loads is returned. A machine that dies
    mid-save leaves a truncated file at the highest step number; without
    this, resume would fail on exactly the checkpoint it should skip.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    candidates: List[tuple[int, Path]] = []
    for p in ckpt_dir.glob("checkpoint_step*.pt"):
        digits = p.stem[len("checkpoint_step"):]
        if digits.isdigit():
            candidates.append((int(digits), p))
    if not candidates:
        return None
    for _step, path in sorted(candidates, reverse=True):
        if not verify or is_readable_checkpoint(path):
            return path
    return None


def load_for_resume(
    path: Path,
    agent,
    *,
    ret_rms: Any = None,
    obs_rms: Any = None,
    map_location: str = "cpu",
    strict_rng: bool = False,
) -> ResumeState:
    """Restore learner state from a checkpoint.

    Raises if the checkpoint predates the resumable format, rather than
    quietly resuming with fresh optimizer moments — that failure mode is
    exactly what this module exists to prevent.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not is_resumable(payload):
        raise ValueError(
            f"{path} is a format-{payload.get('format', 1)} checkpoint "
            "(network only) and cannot be resumed from. Restarting with it "
            "would reset the optimizer, normalisers and RNG streams while "
            "looking like a continuation."
        )

    agent.network.load_state_dict(payload["network"])
    agent.optimizer.load_state_dict(payload["optimizer"])
    _load_rms(ret_rms, payload.get("ret_rms"))
    _load_rms(obs_rms, payload.get("obs_rms"))

    restored_rng = True
    try:
        _load_rng(payload.get("rng"))
    except Exception:
        if strict_rng:
            raise
        restored_rng = False

    if hasattr(agent, "_sde_step_counter"):
        agent._sde_step_counter = int(payload.get("sde_step", 0))

    return ResumeState(
        global_step=int(payload["global_step"]),
        best_eval=float(payload.get("best_eval", -float("inf"))),
        path=path,
        restored_optimizer=True,
        restored_rng=restored_rng,
    )


def resolve_resume(
    resume: Any, ckpt_dir: Path,
) -> Optional[Path]:
    """Interpret the `resume` config option.

    ``false``/``never``/``None``  do not resume
    ``true``/``auto``             resume from the latest checkpoint if one exists
    ``<path>``                    resume from exactly that file
    """
    if resume in (None, False, "false", "never", "no", ""):
        return None
    if resume in (True, "true", "auto", "yes", "latest"):
        return find_latest_checkpoint(Path(ckpt_dir))
    path = Path(str(resume))
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path
