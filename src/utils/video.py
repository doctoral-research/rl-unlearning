"""
Video recording utility for visualizing agent behavior.

Records short MP4 clips of the agent acting in the environment.
Works headlessly (no display needed) via gymnasium's rgb_array rendering.
"""
import numpy as np
import gymnasium as gym
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def _expected_obs_dim(agent) -> Optional[int]:
    """Input width the agent's network expects, read off its first Linear
    layer. Returns None for conv encoders, where the first Linear is not
    the input layer."""
    import torch.nn as nn

    net = getattr(agent, "network", None)
    if net is None or getattr(net, "use_conv", False):
        return None
    for m in net.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    return None


def record_videos(
    agent,
    env_id: str,
    output_dir: str,
    label: str,
    num_videos: int = 3,
    max_steps: int = 500,
    fps: int = 30,
    seed: Optional[int] = None,
    deterministic: bool = True,
    include_pos: Optional[bool] = None,
) -> List[str]:
    """Record MP4 videos of the agent acting in the environment.

    Args:
        agent: Agent with a ``select_action(obs, deterministic)`` method.
        env_id: Gymnasium environment ID (e.g. "CartPole-v1").
        output_dir: Directory to save videos in.
        label: Prefix for filenames
            (e.g. "baseline", "after_trajectory_selective").
        num_videos: Number of episodes to record.
        max_steps: Maximum steps per episode (keeps videos short).
        fps: Frames per second in the output MP4.
        seed: Optional seed for reproducibility.
        deterministic: Whether to use deterministic action selection.

    Returns:
        List of saved file paths.
    """
    try:
        import cv2
    except ImportError:
        logger.warning(
            "opencv-python not installed — skipping video recording. "
            "Install with: pip install opencv-python"
        )
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from utils.env_wrappers import make_env
    # Default to symbolic encoding for MiniGrid envs (matches what
    # current MiniGrid policies are trained with, except legacy Empty-8x8
    # which uses image). Pin the FourRooms goal so visualizations match
    # the pinned-goal training setup. NOTE: only matters if the caller
    # doesn't already pass these in.
    if "FourRooms" in env_id:
        obs_enc = "symbolic"
        fgp = (13, 3)
    elif "MiniGrid-" in env_id and "Empty" not in env_id:
        obs_enc = "symbolic"
        fgp = None
    else:
        obs_enc = "image"
        fgp = None
    def _build(inc_pos):
        return make_env(env_id, render_mode="rgb_array",
                        obs_encoding=obs_enc, fixed_goal_pos=fgp,
                        include_pos=inc_pos)

    env = _build(True if include_pos is None else include_pos)
    # MiniGrid policies may be trained with or without the privileged
    # (x, y, direction) prefix (see MiniGridFlatPos.include_pos). When the
    # caller does not say which, infer it from the agent so the recorder
    # never mismatches the checkpoint it was handed.
    if include_pos is None:
        want = _expected_obs_dim(agent)
        got = int(env.observation_space.shape[0])
        if want is not None and want != got:
            env.close()
            env = _build(False)
    saved_paths = []

    for ep in range(num_videos):
        reset_kwargs = {"seed": seed + ep} if seed is not None else {}
        obs, _ = env.reset(**reset_kwargs)
        frames = [env.render()]

        total_reward = 0.0
        for _ in range(max_steps):
            action, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            total_reward += reward
            if terminated or truncated:
                break

        # Write MP4 via OpenCV then re-encode with ffmpeg for H.264
        filename = f"{label}_ep{ep}_r{total_reward:.0f}.mp4"
        filepath = out / filename
        tmp_path = out / f"_tmp_{filename}"
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))
        for frame in frames:
            # gymnasium renders RGB; OpenCV expects BGR
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        # Re-encode to H.264 for broad compatibility (VSCode, browsers)
        import shutil
        import subprocess
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_path), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-loglevel", "error", str(filepath)],
                check=True,
            )
            tmp_path.unlink()
        else:
            # No ffmpeg available — keep the mp4v version
            tmp_path.rename(filepath)

        saved_paths.append(str(filepath))
        logger.info(
            f"Saved video: {filepath} "
            f"({len(frames)} frames, return={total_reward:.1f})"
        )

    env.close()
    return saved_paths
