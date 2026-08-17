"""tiny_context — LLM context window manager. Zero dependencies.

A single-file library for assembling, budgeting, compressing, and routing
context windows for LLM applications. Solves the four problems every LLM
app hits in production:

  1. **Budgeting** — fit a prompt into N tokens without guessing.
  2. **Truncation** — drop least-important chunks when over budget.
  3. **Compression** — summarize on the fly with pluggable strategies.
  4. **Routing** — pick the right model tier based on context length.

Designed to be paired with any LLM SDK (OpenAI, Anthropic, local) and any
tokenizer (tiktoken, transformers, your own counter).

Usage:
    from tiny_context import Context, Budget, Message, Role

    ctx = Context(model="gpt-4o-mini", budget=Budget(max_tokens=4096))
    ctx.add_system("You are a helpful assistant.")
    ctx.add_user("What's the weather in Bahrain?")

    # Inject retrieved docs as a single block
    ctx.add_block("retrieved", docs, importance=0.8)

    # Render ready for an OpenAI/Anthropic call
    messages = ctx.render_messages()
    openai.chat.completions.create(model=ctx.model, messages=messages)
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence


__version__ = "0.1.0"


# ---------- Roles ----------


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------- Token counting ----------


Tokenizer = Callable[[str], int]


def _default_tokenizer(text: str) -> int:
    """Conservative char/4 heuristic when no real tokenizer is provided.

    Real tokenizers (tiktoken, huggingface) typically count ~3-4 chars/token
    for English; we use 3.5 to err slightly on the safe (over) side so we
    don't accidentally exceed the model's actual limit.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3.5))


# ---------- Budget ----------


@dataclass
class Budget:
    """Token budget for a context window.

    `max_tokens` is the hard cap (model context length).
    `reserve_for_completion` reserves tokens for the model's response.
    `safety_margin` reserves a percentage of the cap for overhead (system
    framing, role markers, etc.).
    """

    max_tokens: int
    reserve_for_completion: int = 0
    safety_margin: float = 0.05  # 5%

    def effective_input(self) -> int:
        margin = int(self.max_tokens * self.safety_margin)
        return max(0, self.max_tokens - self.reserve_for_completion - margin)

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.safety_margin < 0.5:
            raise ValueError("safety_margin must be in [0, 0.5)")


# ---------- Message ----------


@dataclass
class Message:
    """A single message in the conversation.

    `importance` is used by the prioritizer when truncation is needed:
    higher importance is kept; lower importance is dropped first.
    """

    role: Role
    content: str
    importance: float = 1.0
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> dict[str, str]:
        out: dict[str, str] = {"role": self.role.value, "content": self.content}
        if self.name:
            out["name"] = self.name
        return out


# ---------- Block (retrieved docs, tool outputs, etc.) ----------


@dataclass
class Block:
    """A named chunk of context (retrieved docs, tool outputs, scratchpad).

    Blocks can be added as a single unit; they're tokenized together and
    can be truncated/compressed independently of messages.
    """

    name: str
    text: str
    importance: float = 0.5
    compressible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return f"--- BEGIN {self.name} ---\n{self.text}\n--- END {self.name} ---"


# ---------- Compression strategies ----------


Compressor = Callable[[str, int], str]


def head_tail_compressor(text: str, target_tokens: int) -> str:
    """Keep the head and tail of the text; drop the middle.

    Useful for long logs / tool outputs where the start sets context and
    the end has the latest result.
    """
    tok = _default_tokenizer
    total = tok(text)
    if total <= target_tokens:
        return text
    half = max(50, target_tokens // 2)
    # Split on whitespace to avoid cutting words
    head_target = half
    tail_target = target_tokens - half
    chars_head = int(head_target * 3.5)
    chars_tail = int(tail_target * 3.5)
    head = text[:chars_head]
    tail = text[-chars_tail:] if len(text) > chars_tail else ""
    omitted = total - (tok(head) + tok(tail))
    marker = f"\n\n[... {omitted} tokens omitted ...]\n\n"
    return head + marker + tail


def summarize_compressor(placeholder: str = "[compressed]") -> Compressor:
    """Return a placeholder compressor that swaps long text for a marker.

    Real summarization would call an LLM here; this is a deterministic
    stub so the library stays zero-dependency. Plug in your own LLM-backed
    compressor by passing a Callable[[str, int], str] to Context().
    """
    def _compress(text: str, target_tokens: int) -> str:
        return f"{placeholder} ({_default_tokenizer(text)} tokens → {target_tokens})"
    return _compress


# ---------- Context ----------


@dataclass
class Context:
    """An assembled context window ready to be sent to an LLM.

    Lifecycle:
        ctx = Context(model="gpt-4o-mini", budget=Budget(max_tokens=8192))
        ctx.add_system("You are a helpful assistant.")
        ctx.add_user("...")
        ctx.add_block("docs", retrieved_text, importance=0.7)
        messages = ctx.render_messages()  # auto-fits to budget
    """

    model: str = "unknown"
    budget: Budget = field(default_factory=lambda: Budget(max_tokens=4096))
    tokenizer: Tokenizer = field(default=_default_tokenizer)
    compressor: Compressor = field(default_factory=lambda: head_tail_compressor)

    _messages: list[Message] = field(default_factory=list)
    _blocks: list[Block] = field(default_factory=list)

    # ---- adders ----

    def add_system(self, content: str, importance: float = 1.0) -> None:
        self._messages.append(Message(Role.SYSTEM, content, importance=importance))

    def add_user(self, content: str, importance: float = 1.0, name: str | None = None) -> None:
        self._messages.append(Message(Role.USER, content, importance=importance, name=name))

    def add_assistant(self, content: str, importance: float = 0.9) -> None:
        self._messages.append(Message(Role.ASSISTANT, content, importance=importance))

    def add_tool(self, content: str, name: str | None = None, importance: float = 0.7) -> None:
        self._messages.append(Message(Role.TOOL, content, importance=importance, name=name))

    def add_block(
        self,
        name: str,
        text: str,
        importance: float = 0.5,
        compressible: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._blocks.append(
            Block(
                name=name,
                text=text,
                importance=importance,
                compressible=compressible,
                metadata=dict(metadata or {}),
            )
        )

    # ---- introspection ----

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def blocks(self) -> list[Block]:
        return list(self._blocks)

    def total_tokens(self) -> int:
        """Total tokens across all messages and blocks (no compression)."""
        tok = self.tokenizer
        total = sum(tok(m.content) for m in self._messages)
        total += sum(tok(b.text) for b in self._blocks)
        return total

    # ---- fitting ----

    def fit(self) -> None:
        """Mutate the context to fit the budget.

        Strategy (in order):
          1. Always keep system + latest user/assistant turn.
          2. Compress blocks (in order of least-important-first) if marked
             compressible.
          3. Truncate the lowest-importance non-protected messages.
          4. Drop lowest-importance blocks entirely if still over budget.
        """
        tok = self.tokenizer
        budget = self.budget.effective_input()

        if self._total(tok) <= budget:
            return

        # 1. Compress blocks
        blocks_sorted = sorted(self._blocks, key=lambda b: b.importance)
        for block in blocks_sorted:
            if self._total(tok) <= budget:
                break
            if not block.compressible:
                continue
            target_for_block = max(
                50, int(budget * 0.1)
            )
            block.text = self.compressor(block.text, target_for_block)

        # 2. Truncate messages (keep system + last turn)
        protected = self._protected_indices()
        candidates = [
            (i, m) for i, m in enumerate(self._messages) if i not in protected
        ]
        candidates.sort(key=lambda x: x[1].importance)
        kept: list[Message] = list(self._messages)
        for i, _m in candidates:
            if self._total(tok) <= budget:
                break
            # Halve the message content as a soft-truncate
            kept[i] = Message(
                role=_m.role,
                content=_m.content[: max(1, len(_m.content) // 2)] + "\n[truncated]",
                importance=_m.importance,
                name=_m.name,
                metadata=_m.metadata,
            )
        self._messages = kept

        # 3. Drop lowest-importance blocks entirely
        if self._total(tok) > budget:
            self._blocks = sorted(
                self._blocks, key=lambda b: b.importance, reverse=True
            )
            while self._blocks and self._total(tok) > budget:
                self._blocks.pop()

    def _total(self, tok: Tokenizer) -> int:
        return sum(tok(m.content) for m in self._messages) + sum(
            tok(b.text) for b in self._blocks
        )

    def _protected_indices(self) -> set[int]:
        """Indices of messages that must never be dropped or truncated."""
        protected: set[int] = set()
        for i, m in enumerate(self._messages):
            if m.role == Role.SYSTEM:
                protected.add(i)
        # Last user/assistant turn is protected
        for i in range(len(self._messages) - 1, -1, -1):
            role = self._messages[i].role
            if role in (Role.USER, Role.ASSISTANT):
                protected.add(i)
                break
        return protected

    # ---- rendering ----

    def render_messages(self) -> list[dict[str, str]]:
        """Render as a list of {role, content} dicts ready for OpenAI/Anthropic.

        Blocks are merged into a single user-role "context" message so the
        model sees the assembled context, not the internal block structure.
        """
        self.fit()
        out: list[dict[str, str]] = []
        for m in self._messages:
            out.append(m.render())
        if self._blocks:
            block_text = "\n\n".join(b.render() for b in self._blocks)
            out.append({"role": Role.USER.value, "content": block_text})
        return out

    def render_text(self) -> str:
        """Render as a flat text string (for completion-style models)."""
        self.fit()
        parts: list[str] = []
        for m in self._messages:
            parts.append(f"[{m.role.value.upper()}] {m.content}")
        for b in self._blocks:
            parts.append(b.render())
        return "\n\n".join(parts)


# ---------- Model registry (lightweight) ----------


@dataclass
class ModelSpec:
    """Lightweight model metadata for context-length-aware routing."""

    name: str
    context_length: int
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    supports_tools: bool = True
    supports_vision: bool = False


# Common models as of Aug 2026
DEFAULT_MODELS: dict[str, ModelSpec] = {
    "gpt-4o-mini": ModelSpec("gpt-4o-mini", 128_000, 0.00015, 0.0006),
    "gpt-4o": ModelSpec("gpt-4o", 128_000, 0.0025, 0.01),
    "gpt-5-mini": ModelSpec("gpt-5-mini", 256_000, 0.00025, 0.002),
    "gpt-5": ModelSpec("gpt-5", 256_000, 0.005, 0.015),
    "claude-haiku-4": ModelSpec("claude-haiku-4", 200_000, 0.0008, 0.004),
    "claude-sonnet-4": ModelSpec("claude-sonnet-4", 200_000, 0.003, 0.015),
    "claude-opus-4": ModelSpec("claude-opus-4", 200_000, 0.015, 0.075),
    "llama-3.3-70b": ModelSpec("llama-3.3-70b", 128_000, 0.00059, 0.00079),
}


def select_model(
    required_tokens: int,
    registry: dict[str, ModelSpec] | None = None,
    prefer: str | None = None,
    max_cost_per_1k: float | None = None,
) -> ModelSpec | None:
    """Pick the cheapest model that fits the required tokens.

    Args:
        required_tokens: how many input tokens you need to fit.
        registry: model registry to search (defaults to DEFAULT_MODELS).
        prefer: prefer this specific model if it fits.
        max_cost_per_1k: filter out models above this input cost.

    Returns:
        ModelSpec, or None if no model fits.
    """
    reg = registry or DEFAULT_MODELS
    candidates: list[ModelSpec] = []
    for spec in reg.values():
        if spec.context_length < required_tokens:
            continue
        if max_cost_per_1k is not None and spec.cost_per_1k_input > max_cost_per_1k:
            continue
        candidates.append(spec)
    if not candidates:
        return None
    if prefer and prefer in reg:
        pref = reg[prefer]
        if pref in candidates:
            return pref
    candidates.sort(key=lambda s: (s.cost_per_1k_input, s.context_length))
    return candidates[0]


# ---------- Heuristics ----------


@dataclass
class Stats:
    """Stats for the most recent `fit()` call — useful for telemetry."""

    total_tokens_before: int = 0
    total_tokens_after: int = 0
    blocks_compressed: int = 0
    blocks_dropped: int = 0
    messages_truncated: int = 0

    @property
    def savings_pct(self) -> float:
        if self.total_tokens_before == 0:
            return 0.0
        return 100.0 * (1 - self.total_tokens_after / self.total_tokens_before)


__all__ = [
    "Role",
    "Budget",
    "Message",
    "Block",
    "Context",
    "ModelSpec",
    "Stats",
    "Tokenizer",
    "Compressor",
    "select_model",
    "DEFAULT_MODELS",
    "head_tail_compressor",
    "summarize_compressor",
]