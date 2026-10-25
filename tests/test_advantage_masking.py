"""Disjoint-objective advantage masking.

Regression tests for a bug where zeroing advantages on forget transitions
did not actually remove their gradient: PPO normalises advantages
internally, so the zeros became -mean/std -- a small, systematic advantage
applied to every forget transition in every epoch, with a sign depending
on whether the forget region was above or below average value.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agents import PPOAgent


def _agent(obs_dim=4, n_actions=2):
    return PPOAgent(observation_dim=obs_dim, action_dim=n_actions,
                    action_type="discrete", device="cpu", n_epochs=1,
                    batch_size=64)


def _rollout(n=256, obs_dim=4, n_actions=2, forget_frac=0.2, mu_forget=0.5,
             seed=0):
    rng = np.random.default_rng(seed)
    adv = rng.normal(0.0, 1.0, n)
    mask = np.zeros(n, dtype=bool)
    mask[: int(forget_frac * n)] = True
    adv[mask] += mu_forget
    adv -= adv.mean()                     # GAE advantages are ~zero mean
    return {
        "observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "actions": rng.integers(0, n_actions, n),
        "log_probs": rng.standard_normal(n).astype(np.float32),
        "values": rng.standard_normal(n).astype(np.float32),
        "advantages": adv.astype(np.float32),
        "returns": rng.standard_normal(n).astype(np.float32),
    }, mask


def _normalised(advantages, mask=None):
    """Mirror of the normalisation inside PPOAgent.update."""
    t = torch.as_tensor(advantages, dtype=torch.float32)
    if mask is None:
        return (t - t.mean()) / (t.std() + 1e-8)
    fm = torch.as_tensor(mask, dtype=torch.bool)
    retain = ~fm
    t = (t - t[retain].mean()) / (t[retain].std() + 1e-8)
    return t.masked_fill(fm, 0.0)


class TestPreZeroingIsNotAblation:
    """The old approach: zero first, then let PPO normalise."""

    @pytest.mark.parametrize("mu_forget", [0.5, -0.5])
    def test_zeroed_entries_do_not_stay_zero(self, mu_forget):
        roll, mask = _rollout(mu_forget=mu_forget)
        adv = roll["advantages"].copy()
        adv[mask] = 0.0                           # what the loop used to do
        out = _normalised(adv)                    # PPO then normalises
        assert abs(float(out[mask].mean())) > 1e-3

    def test_residual_sign_follows_the_forget_region_value(self):
        """Why the artifact is not merely noise: its sign flips with the
        forget region's own advantage, so it reinforces the forbidden
        behaviour in some environments and suppresses it in others."""
        signs = []
        for mu in (0.5, -0.5):
            roll, mask = _rollout(mu_forget=mu)
            adv = roll["advantages"].copy()
            adv[mask] = 0.0
            signs.append(np.sign(float(_normalised(adv)[mask].mean())))
        assert signs[0] != signs[1]


class TestMaskedNormalisation:
    """The fix: normalise over retain only, then re-zero."""

    @pytest.mark.parametrize("mu_forget", [0.5, -0.5, 0.0])
    def test_forget_advantages_are_exactly_zero(self, mu_forget):
        roll, mask = _rollout(mu_forget=mu_forget)
        out = _normalised(roll["advantages"], mask)
        assert float(out[mask].abs().max()) == 0.0

    def test_retain_advantages_are_standardised(self):
        roll, mask = _rollout()
        out = _normalised(roll["advantages"], mask)[~torch.as_tensor(mask)]
        assert float(out.mean()) == pytest.approx(0.0, abs=1e-5)
        assert float(out.std()) == pytest.approx(1.0, abs=1e-3)

    def test_all_forget_mask_leaves_no_policy_gradient(self):
        roll, _ = _rollout()
        roll["forget_mask"] = np.ones(len(roll["advantages"]), dtype=bool)
        metrics = _agent().update(roll)
        assert metrics["policy_loss"] == pytest.approx(0.0, abs=1e-6)


class TestAgentIntegration:
    def test_update_accepts_a_forget_mask(self):
        agent = _agent()
        roll, mask = _rollout()
        roll["forget_mask"] = mask
        metrics = agent.update(roll)
        assert np.isfinite(metrics["policy_loss"])

    def test_update_without_mask_still_works(self):
        agent = _agent()
        roll, _ = _rollout()
        assert np.isfinite(agent.update(roll)["policy_loss"])

    def test_masked_update_changes_weights_differently(self):
        """Masked and unmasked updates are genuinely different operations,
        not a cosmetic relabelling."""
        roll, mask = _rollout()
        torch.manual_seed(0)
        a1 = _agent(); a1.update(dict(roll))
        torch.manual_seed(0)
        a2 = _agent(); a2.update({**roll, "forget_mask": mask})
        diffs = [float((p - q).abs().max())
                 for p, q in zip(a1.network.parameters(), a2.network.parameters())]
        assert max(diffs) > 1e-7
