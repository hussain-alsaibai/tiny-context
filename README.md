# tiny-context — LLM Context Window Manager

> **Stop guessing how many tokens fit. Stop dropping the wrong chunk.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](tiny_context.py)
[![Part of the tiny-* ecosystem](https://img.shields.io/badge/tiny--*-ecosystem-purple.svg)](#ecosystem)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](tiny_context.py)

`tiny-context` is a single-file, zero-dependency library for assembling, budgeting, compressing, and routing context windows for LLM applications. Pair it with any LLM SDK (OpenAI, Anthropic, local) and any tokenizer (tiktoken, transformers, your own counter).

## ✨ Features

- **💰 Budget enforcement** — `Budget(max_tokens, reserve_for_completion, safety_margin)` so you never blow the context window.
- **🎯 Importance-aware fitting** — mark messages and blocks with importance scores; the fitter keeps high-importance content and drops low.
- **🗜️ Pluggable compression** — drop in `head_tail_compressor` (zero-dep default) or wire your own LLM-backed summarizer.
- **🧱 Block abstraction** — retrieved docs, tool outputs, scratchpad — treated as named, independently-truncatable chunks.
- **🚦 Model routing** — `select_model(required_tokens, prefer=..., max_cost_per_1k=...)` picks the cheapest model that fits.
- **🪶 Tiny** — ~14 KB single file, zero deps.

## 🚀 Quick Start

```python
from tiny_context import Context, Budget, select_model

ctx = Context(
    model=select_model(required_tokens=4000, prefer="gpt-4o-mini").name,
    budget=Budget(max_tokens=4096, reserve_for_completion=500),
)

ctx.add_system("You are a helpful assistant with access to company docs.")
ctx.add_user("What is the PTO policy for Bahrain employees?")

# Retrieved docs (RAG, tool outputs, anything)
ctx.add_block("retrieved", retrieved_text, importance=0.8)

messages = ctx.render_messages()  # auto-fits to budget

import openai
openai.chat.completions.create(model=ctx.model, messages=messages)
```

## 📦 Installation

```bash
pip install tiny-context
```

Or just copy `tiny_context.py` into your project — single file, zero dependencies.

## 💰 Budget

```python
from tiny_context import Budget

b = Budget(max_tokens=8192, reserve_for_completion=1000, safety_margin=0.05)
# b.effective_input() == 6784  (room for the model's response + 5% overhead)
```

## 🎯 Importance & Fitting

When the context exceeds budget, `Context.fit()` runs in order:

1. **Compress blocks** (least-important first, if marked `compressible=True`).
2. **Truncate messages** (lowest importance first), keeping `SYSTEM` and the latest user/assistant turn protected.
3. **Drop blocks** (lowest importance first) if still over.

```python
ctx.add_block("scratchpad", scratch, importance=0.3, compressible=True)
ctx.add_block("must-have", critical, importance=0.95, compressible=False)
```

## 🗜️ Compression

Built-in `head_tail_compressor` keeps the head and tail of long text and inserts an omission marker — useful for logs and tool outputs. Replace it with your LLM-backed summarizer:

```python
def my_llm_summarize(text: str, target_tokens: int) -> str:
    return openai_chat(
        model="gpt-4o-mini",
        system="Summarize concisely.",
        user=f"Summarize this in under {target_tokens} tokens:\n\n{text}",
    )

ctx = Context(budget=..., compressor=my_llm_summarize)
```

For tests and offline use, `summarize_compressor()` returns a deterministic placeholder.

## 🧱 Blocks

Blocks are named chunks that render as their own delimited region in the final prompt. They can be marked `compressible=True/False` and given `importance` independently of the conversation.

```python
ctx.add_block("docs", docs_text, importance=0.7)
ctx.add_block("tool-output", json.dumps(result), importance=0.4, compressible=True)
ctx.add_block("user-prefs", prefs, importance=0.95, compressible=False)
```

## 🚦 Model Routing

`select_model()` picks the cheapest model that fits, optionally preferring a specific model or filtering by cost:

```python
spec = select_model(
    required_tokens=12_000,
    prefer="gpt-4o-mini",     # prefer this if it fits
    max_cost_per_1k=0.001,    # exclude expensive models
)
ctx = Context(model=spec.name, budget=Budget(max_tokens=spec.context_length))
```

A default registry of common models (Aug 2026) ships in `DEFAULT_MODELS`. Pass your own dict to override:

```python
from tiny_context import select_model, ModelSpec

my_models = {
    "fast-local":  ModelSpec("fast-local",  8192,  cost_per_1k_input=0.0),
    "smart-cloud": ModelSpec("smart-cloud", 128_000, cost_per_1k_input=0.002),
}
spec = select_model(5000, registry=my_models)
```

## 🔌 Plug Your Own Tokenizer

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")

ctx = Context(
    model="gpt-4o-mini",
    budget=Budget(max_tokens=128_000),
    tokenizer=lambda text: len(enc.encode(text)),
)
```

## 🧪 Why?

Every LLM app hits the same four problems:

1. **You don't know how many tokens you're sending** → bills surprise you; responses get truncated mid-sentence.
2. **You don't know what to drop** → critical context gets cut; users see "I don't know."
3. **You can't summarize without a network round-trip** → tests are flaky; latency balloons.
4. **You don't pick the right model tier** → a 100-token query costs the same as a 100K-token one.

`tiny-context` solves all four with zero dependencies and one file you can read in 5 minutes.

## 🌐 Ecosystem

Part of the [tiny-* ecosystem](https://github.com/hussain-alsaibai) of zero-dependency Python libraries:

- **tiny-router** — zero-dep HTTP router with async handlers & DI (v0.3.0)
- **tiny-log** — structured logging
- **tiny-validator** — JSON Schema + JSON-Schema-like validation
- **tiny-cli** — zero-dep CLI framework
- **tiny-config** — config loader with env interpolation
- **tiny-cache** — multi-backend cache (LRU, TTL, file)
- **snapdb** — in-memory columnar DB with snapshots

## License

MIT.