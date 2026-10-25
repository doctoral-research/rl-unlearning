"""Model backends for `authoring.author_scenario`.

`author_scenario` takes any `Callable[[str], str]` mapping a prompt to
candidate YAML, which keeps the repair loop free of provider code and
lets the tests drive it with scripted responses. This module supplies the
adapters for talking to a real model.

The loop is only worth reporting on when it has faced a real model. A
scripted backend proves the gates fire; it says nothing about how often a
model clears them, which is the number an authoring claim rests on.

Everything here prefers free and self-hosted, and the HTTP backends use
only the standard library so the package gains no dependency:

    hf:<repo>              a local transformers model on your own GPU
    ollama:<model>         a local ollama daemon
    openai-compat:<model>  any OpenAI-shaped endpoint, set --base-url:
                           llama.cpp, vLLM, LM Studio, Groq, OpenRouter,
                           Together, or anything else that speaks it

The paid first-party APIs are reachable through `openai-compat` by
pointing `base_url` at them, so they need no special case.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Optional, Sequence

Backend = Callable[[str], str]

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


class BackendError(RuntimeError):
    """Raised when a backend cannot be constructed or called."""


def _post_json(url: str, payload: dict, headers: Optional[dict] = None,
               timeout: float = 300.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:                 # pragma: no cover
        raise BackendError(
            f"{url} returned {exc.code}: {exc.read()[:300].decode(errors='replace')}"
        ) from exc
    except urllib.error.URLError as exc:                  # pragma: no cover
        raise BackendError(f"cannot reach {url}: {exc.reason}") from exc


def ollama_backend(model: str = "qwen2.5-coder:7b",
                   host: str = DEFAULT_OLLAMA_HOST,
                   temperature: float = 0.0,
                   num_predict: int = 1200) -> Backend:
    """A local ollama daemon. No key, no account, nothing leaves the machine."""
    url = f"{host.rstrip('/')}/api/chat"

    def call(prompt: str) -> str:
        out = _post_json(url, {
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "num_predict": num_predict},
        })
        return out.get("message", {}).get("content", "")

    return call


def openai_compatible_backend(model: str,
                              base_url: Optional[str] = None,
                              api_key: Optional[str] = None,
                              temperature: float = 0.0,
                              max_tokens: int = 1200) -> Backend:
    """Any endpoint speaking the OpenAI chat-completions shape.

    Covers llama.cpp's server, vLLM, LM Studio and the free tiers of the
    hosted providers. `api_key` is optional because a local server does
    not want one.
    """
    base = (base_url or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:8000/v1").rstrip("/")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    url = f"{base}/chat/completions"

    def call(prompt: str) -> str:
        out = _post_json(url, {
            "model": model, "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }, headers=headers)
        choices = out.get("choices") or []
        if not choices:
            raise BackendError(f"no choices in response: {str(out)[:200]}")
        return choices[0].get("message", {}).get("content") or ""

    return call


def hf_backend(model: str = DEFAULT_HF_MODEL,
               device: Optional[str] = None,
               temperature: float = 0.0,
               max_new_tokens: int = 1200,
               dtype: str = "auto") -> Backend:
    """A transformers model loaded in-process, on your own hardware.

    Greedy by default: a repair loop's convergence rate is not meaningful
    if resampling noise is doing the work.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise BackendError(
            "the 'transformers' package is not installed; "
            "pip install transformers accelerate") from exc

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(
        model, dtype=(torch.bfloat16 if dtype == "auto" and dev == "cuda"
                      else None),
        device_map=dev)
    net.eval()

    def call(prompt: str) -> str:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(net.device)
        with torch.no_grad():
            out = net.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    return call


def escalating_backend(factory: Callable[..., Backend],
                       temperatures: Sequence[float] = (0.0, 0.6, 0.9),
                       **kwargs) -> Backend:
    """Raise the sampling temperature on each successive attempt.

    A greedy backend handed a repair prompt tends to reproduce the
    mistake it just made: the feedback is in the context, but the most
    likely continuation has not changed much. Measured on
    qwen2.5-coder:7b, greedy retries recovered none of three rejected
    candidates and escalating ones recovered one, so determinism is part
    of the problem and model capacity is the rest.

    The first attempt stays greedy, so a backend that succeeds outright
    is unaffected and results stay reproducible.
    """
    state = {"n": 0}

    def call(prompt: str) -> str:
        temp = temperatures[min(state["n"], len(temperatures) - 1)]
        state["n"] += 1
        return factory(temperature=temp, **kwargs)(prompt)

    return call


def echo_backend(responses) -> Backend:
    """Replays a fixed list of responses, for tests and dry runs.

    Results obtained with this backend are not evidence about a model.
    """
    queue = list(responses)

    def call(_prompt: str) -> str:
        return queue.pop(0) if queue else ""

    return call


BACKENDS = {
    "hf": hf_backend,
    "ollama": ollama_backend,
    "openai-compat": openai_compatible_backend,
}


def make_backend(spec: str, **kwargs) -> Backend:
    """Build a backend from `provider[:model]`.

    The model may itself contain colons (`ollama:qwen2.5-coder:7b`), so
    only the first separator is significant.
    """
    provider, _, model = spec.partition(":")
    if provider not in BACKENDS:
        raise BackendError(
            f"unknown provider {provider!r}; valid: {sorted(BACKENDS)}")
    if model:
        kwargs["model"] = model
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return BACKENDS[provider](**kwargs)
