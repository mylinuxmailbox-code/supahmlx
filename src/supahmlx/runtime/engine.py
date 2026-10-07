"""Runtime model wrapper with generation APIs and streaming support."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

from supahmlx.config import ModeConfig
from supahmlx.generation.sampling import Sampler, SamplingConfig
from supahmlx.generation.speculative import ConservativeSpeculativeDecoder, SpeculativeStats
from supahmlx.runtime.interfaces import ExecutionPlan, ModelBackend
from supahmlx.tokenizer.base import Tokenizer


@dataclass(frozen=True)
class GenerationChunk:
    token_id: int
    text: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    generated_tokens: int
    latency_s: float
    speculative: SpeculativeStats


class RuntimeModel:
    """User-facing runtime object returned by supahmlx.load."""

    def __init__(
        self,
        backend: ModelBackend,
        tokenizer: Tokenizer,
        mode: ModeConfig,
        plan: ExecutionPlan,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.mode = mode
        self.plan = plan

    def _generate_token_ids(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        seed: int | None,
    ) -> tuple[list[int], SpeculativeStats]:
        prompt_tokens = self.tokenizer.encode(prompt)
        sampler = Sampler(seed=seed)

        def _target_step(context_tokens: list[int], limit: int) -> list[int]:
            output: list[int] = []
            state = list(context_tokens)
            for _ in range(limit):
                logits = self.backend.generate_token_logits(state)
                token = sampler.sample(
                    logits,
                    SamplingConfig(
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        greedy=temperature == 0,
                    ),
                )
                state.append(token)
                output.append(token)
            return output

        speculative = ConservativeSpeculativeDecoder(_target_step)
        return speculative.step(prompt_tokens, max_tokens)

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> Iterator[GenerationChunk]:
        """Yield generated tokens incrementally as text chunks."""
        temp = self.mode.temperature_default if temperature is None else temperature
        nucleus = self.mode.top_p_default if top_p is None else top_p
        k = self.mode.top_k_default if top_k is None else top_k
        token_ids, _ = self._generate_token_ids(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=nucleus,
            top_k=k,
            seed=seed,
        )
        for token_id in token_ids:
            yield GenerationChunk(token_id=token_id, text=self.tokenizer.decode([token_id]))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate text from prompt."""
        temp = self.mode.temperature_default if temperature is None else temperature
        nucleus = self.mode.top_p_default if top_p is None else top_p
        k = self.mode.top_k_default if top_k is None else top_k

        started = perf_counter()
        token_ids, speculative = self._generate_token_ids(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=nucleus,
            top_k=k,
            seed=seed,
        )
        text = self.tokenizer.decode(token_ids)
        latency = perf_counter() - started
        return GenerationResult(
            text=text,
            token_ids=token_ids,
            generated_tokens=len(token_ids),
            latency_s=latency,
            speculative=speculative,
        )
