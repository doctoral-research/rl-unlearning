"""How many forget labels does each perturbation in the drift experiment flip?

The policy-level comparison asks whether two training runs end in different
places. Read on its own it cannot say whether the difference is large,
because training amplifies any change to the labels. This script measures
the size of each change at the level where it is applied: the fraction of
rollout steps whose forget label differs between the two arms.

For the two readings of `applies_to` that fraction depends on the episodes
the policy produces, so it is measured on rollouts of each seed's baseline,
with the per-episode segmentation the trainer uses. For label noise it is
the flip probability by construction, so it needs no measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_policy_divergence import SEEDS, build, load_cfg      # noqa: E402
from online_unlearn import transitions_in_forget_region        # noqa: E402
from scenarios import ForgetScenario, view_from_cfg            # noqa: E402
from utils.checkpointing import load_checkpoint                # noqa: E402
from utils.env_wrappers import make_env                        # noqa: E402

DOMAINS = {
    "FourRooms": ("env=fourrooms", "fourrooms/avoid_bottom_right",
                  "experiments/checkpoints/fourrooms_ppo_seed{seed}/final_model.pt"),
    "CartPole": ("env=cartpole", "cartpole/avoid_outer_third",
                 "experiments/checkpoints/cartpole_ppo_seed{seed}/final_model.pt"),
}


def rollout(agent, env, scenario, n, seed):
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    feats, dones = [], []
    for _ in range(n):
        feats.append(scenario.features(obs, env=env))
        a, _i = agent.select_action(obs, deterministic=False)
        obs, _r, term, trunc, _ = env.step(a)
        dones.append(term or trunc)
        if term or trunc:
            obs, _ = env.reset()
    return np.asarray(feats, np.float32), np.asarray(dones, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2048,
                    help="one rollout buffer, as in the trainer")
    ap.add_argument("--json", default="paper/data/drift_label_disagreement.json")
    args = ap.parse_args()
    out = {}
    for dom, (env_ov, scen, ck) in DOMAINS.items():
        cfg_ms = load_cfg(f"{env_ov},scenario={scen}")
        cfg_wt = load_cfg(f"{env_ov},scenario={scen}_wholetraj")
        env = make_env(cfg_ms.env_id, seed=0,
                       obs_encoding=cfg_ms.get("obs_encoding", "image"),
                       fixed_goal_pos=cfg_ms.get("fixed_goal_pos", None),
                       include_pos=cfg_ms.get("include_pos", True))
        view = view_from_cfg(cfg_ms, env=env)
        s_ms = ForgetScenario.from_config(cfg_ms, view=view)
        s_wt = ForgetScenario.from_config(cfg_wt, view=view)
        rows = []
        for seed in SEEDS:
            agent = build(cfg_ms)
            agent.network.load_state_dict(
                load_checkpoint(Path(ck.format(seed=seed)), map_location="cpu")["network"])
            f, d = rollout(agent, env, s_ms, args.steps, seed)
            m_ms = transitions_in_forget_region(f, s_ms, d)
            m_wt = transitions_in_forget_region(f, s_wt, d)
            rows.append({"seed": seed, "rate_ms": float(m_ms.mean()),
                         "rate_wt": float(m_wt.mean()),
                         "disagreement": float((m_ms != m_wt).mean())})
        dis = np.array([r["disagreement"] for r in rows])
        out[dom] = {"per_seed": rows, "mean_disagreement": float(dis.mean()),
                    "min": float(dis.min()), "max": float(dis.max()),
                    "mean_rate_ms": float(np.mean([r["rate_ms"] for r in rows])),
                    "mean_rate_wt": float(np.mean([r["rate_wt"] for r in rows]))}
        print(f"{dom:<10} labels differ on {100 * dis.mean():.1f}% of steps "
              f"(range {100 * dis.min():.1f}-{100 * dis.max():.1f}%), "
              f"forget rate {out[dom]['mean_rate_ms']:.3f} vs {out[dom]['mean_rate_wt']:.3f}")
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
