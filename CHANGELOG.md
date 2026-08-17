# Changelog

All notable changes to tiny-context are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-08-17

### Added
- Initial release: zero-dependency LLM context window manager.
- `Budget` with `max_tokens`, `reserve_for_completion`, `safety_margin`, `effective_input()`.
- `Message` and `Block` dataclasses with importance + metadata.
- `Context` with `add_system`, `add_user`, `add_assistant`, `add_tool`, `add_block`.
- `Context.fit()` importance-aware fitting: compress blocks → truncate messages → drop blocks.
- `head_tail_compressor` (zero-dep default) and `summarize_compressor` (deterministic placeholder).
- Pluggable `Tokenizer` and `Compressor` (e.g. tiktoken, custom LLM summarizer).
- `render_messages()` returns OpenAI/Anthropic-ready list of `{role, content}` dicts.
- `render_text()` returns flat completion-style text.
- `select_model(required_tokens, prefer=, max_cost_per_1k=)` model routing with `DEFAULT_MODELS` registry.
- 24 tests covering Budget, Message, Context, fit, compressors, model routing, custom tokenizers.