"""Model backends for the authoring loop.

These tests never touch the network: they check construction, error
handling and the retry policy, which is where the logic lives.
"""

from __future__ import annotations

import pytest

from scenarios.backends import (
    BackendError, echo_backend, escalating_backend, make_backend,
)


class TestMakeBackend:
    def test_rejects_unknown_provider(self):
        with pytest.raises(BackendError, match="unknown provider"):
            make_backend("nope:model")

    def test_error_lists_valid_providers(self):
        with pytest.raises(BackendError, match="hf"):
            make_backend("nope")

    def test_model_may_contain_colons(self):
        """ollama names models like qwen2.5-coder:7b, so only the first
        separator is the provider boundary."""
        captured = {}

        def fake(model=None, **kw):
            captured["model"] = model
            return lambda p: ""

        from scenarios import backends
        backends.BACKENDS["fake"] = fake
        try:
            make_backend("fake:qwen2.5-coder:7b")
            assert captured["model"] == "qwen2.5-coder:7b"
        finally:
            del backends.BACKENDS["fake"]


class TestEchoBackend:
    def test_replays_in_order(self):
        b = echo_backend(["one", "two"])
        assert (b("p"), b("p")) == ("one", "two")

    def test_returns_empty_when_exhausted(self):
        b = echo_backend(["one"])
        b("p")
        assert b("p") == ""


class TestEscalatingBackend:
    def _factory(self, log):
        def make(temperature=0.0, **kw):
            def call(prompt):
                log.append(temperature)
                return "x"
            return call
        return make

    def test_first_attempt_stays_greedy(self):
        log = []
        b = escalating_backend(self._factory(log))
        b("p")
        assert log == [0.0]

    def test_temperature_rises_across_attempts(self):
        log = []
        b = escalating_backend(self._factory(log), temperatures=(0.0, 0.5, 1.0))
        for _ in range(3):
            b("p")
        assert log == [0.0, 0.5, 1.0]

    def test_saturates_at_the_last_temperature(self):
        log = []
        b = escalating_backend(self._factory(log), temperatures=(0.0, 0.7))
        for _ in range(4):
            b("p")
        assert log == [0.0, 0.7, 0.7, 0.7]

    def test_passes_through_extra_kwargs(self):
        seen = {}

        def make(temperature=0.0, model=None):
            seen["model"] = model
            return lambda p: "x"

        escalating_backend(make, model="m")("p")
        assert seen["model"] == "m"
