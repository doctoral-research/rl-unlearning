"""Does the reading of `applies_to` change the resulting POLICY?

Specification drift has so far been measured on labels: two defensible
readings of one English sentence disagree about which steps to forget.
That is a claim about a number in a table. The claim that matters for
anyone deploying this is stronger -- that the choice changes the policy
you end up with.

The comparison is clean here because the training pipeline is
deterministic given (config, seed): relaunching an identical run
reproduces its metrics bit for bit. Run-to-run noise is zero, so any
divergence between the two arms is caused by the one line that differs
between the two scenario files.

Both policies are evaluated on a COMMON set of states drawn from the
shared baseline, so the comparison is not contaminated by the two
policies visiting different places.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hydra import compose, initialize_config_dir        # noqa: E402

from agents import PPOAgent                              # noqa: E402
from utils.checkpointing import load_checkpoint          # noqa: E402
from utils.env_wrappers import make_env                  # noqa: E402

SEEDS = tuple(range(42, 57))


def load_cfg(overrides):
    """Same composition the trainer used, so dims and encoding match."""
    with initialize_config_dir(config_dir=str(Path("configs").resolve()),
                               version_base=None):
        return compose(config_name="config",
                       overrides=[o for o in overrides.split(",") if o])


def build(cfg):
    return PPOAgent(observation_dim=cfg.observation_dim,
                    action_dim=cfg.action_dim, device="cpu",
                    action_type=cfg.action_type)


def collect_states(agent, env, n, seed=0):
    """States from the shared baseline: neutral ground for both arms.

    Action sampling is seeded so the evaluation set is identical across
    invocations; otherwise the divergence number moves by a couple of
    points between runs purely from which states were visited.
    """
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=0)
    out = []
    for _ in range(n):
        out.append(obs.copy())
        a, _i = agent.select_action(obs, deterministic=False)
        obs, _r, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    return np.asarray(out, dtype=np.float32)


def logits(agent, states):
    with torch.no_grad():
        out = agent.network(torch.as_tensor(states))
    return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-overrides", default="env=fourrooms,scenario=fourrooms/avoid_bottom_right")
    ap.add_argument("--baseline-glob", default="experiments/checkpoints/fourrooms_ppo_seed{seed}/final_model.pt")
    ap.add_argument("--arms", nargs=2, default=["ab_ms", "ab_wt"],
                    help="output-dir suffixes for the two readings")
    ap.add_argument("--policy-file", default="unlearned_no_op.pt")
    ap.add_argument("--states", type=int, default=4000)
    ap.add_argument("--label", default="FourRooms")
    ap.add_argument("--json", default=None, help="write per-seed results here")
    args = ap.parse_args()
    N_STATES = args.states
    cfg = load_cfg(args.env_overrides)
    print(f"domain: {args.label}  ({cfg.env_id})")
    env = make_env(cfg.env_id, seed=0,
                   obs_encoding=cfg.get("obs_encoding", "image"),
                   fixed_goal_pos=cfg.get("fixed_goal_pos", None),
                   include_pos=cfg.get("include_pos", True))

    print(f"{'seed':<6}{'KL(ms||wt)':>12}{'KL(wt||ms)':>12}"
          f"{'argmax agree':>14}{'vs baseline':>13}")
    print("-" * 58)
    kls, agrees, base_agrees = [], [], []
    for seed in SEEDS:
        base_ck = Path(args.baseline_glob.format(seed=seed))
        base = build(cfg)
        base.network.load_state_dict(load_checkpoint(base_ck, map_location="cpu")["network"])
        states = collect_states(base, env, N_STATES, seed=seed)

        arms = {}
        for lab in args.arms:
            hits = list(Path("experiments/outputs").glob(
                f"*seed{seed}_*_{lab}/{args.policy_file}"))
            if not hits:
                raise SystemExit(f"missing policy for seed {seed} arm {lab}")
            ag = build(cfg)
            ag.network.load_state_dict(
                torch.load(hits[0], map_location="cpu")["network"])
            arms[lab] = ag

        lm = torch.log_softmax(logits(arms[args.arms[0]], states), dim=-1)
        lw = torch.log_softmax(logits(arms[args.arms[1]], states), dim=-1)
        lb = torch.log_softmax(logits(base, states), dim=-1)
        kl_mw = float((lm.exp() * (lm - lw)).sum(-1).mean())
        kl_wm = float((lw.exp() * (lw - lm)).sum(-1).mean())
        agree = float((lm.argmax(-1) == lw.argmax(-1)).float().mean())
        b_agree = float((lm.argmax(-1) == lb.argmax(-1)).float().mean())
        kls.append(kl_mw); agrees.append(agree); base_agrees.append(b_agree)
        print(f"{seed:<6}{kl_mw:12.4f}{kl_wm:12.4f}{agree:14.3f}{b_agree:13.3f}")

    print("-" * 58)
    print(f"{'mean':<6}{np.mean(kls):12.4f}{'':12}{np.mean(agrees):14.3f}"
          f"{np.mean(base_agrees):13.3f}")
    arm_arm = float(np.mean(agrees))
    arm_base = float(np.mean(base_agrees))
    print("\nTwo policies trained from the same baseline, same seed, same")
    print("hyperparameters, differing only in how one English sentence was")
    print(f"read, choose a different action on {100 * (1 - arm_arm):.1f}% "
          "of shared states.")
    print("\nRaw disagreement is NOT comparable across domains: it depends")
    print("on how many actions there are (7 in MiniGrid, 2 in CartPole).")
    print("The scale-free statement is the ratio against how far each arm")
    print("moved from the shared baseline.")
    print(f"  arm-vs-arm agreement      : {arm_arm:.3f}")
    print(f"  arm-vs-baseline agreement : {arm_base:.3f}")
    print(f"  ratio                     : {arm_arm / arm_base:.2f}")
    print("A ratio near 1.0 means the choice of reading moved the policy")
    print("about as much as the unlearning run itself did.")

    if args.json:
        import json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "domain": args.label, "env_id": cfg.env_id,
            "n_actions": int(cfg.action_dim), "n_states": N_STATES,
            "seeds": list(SEEDS), "kl_ms_wt": kls,
            "arm_arm_agreement": agrees,
            "arm_baseline_agreement": base_agrees,
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
