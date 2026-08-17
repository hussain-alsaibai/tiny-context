"""Tests for tiny_context. Run with: python test_tiny_context.py -v"""

from __future__ import annotations

import unittest

from tiny_context import (
    Block,
    Budget,
    Context,
    Message,
    ModelSpec,
    Role,
    head_tail_compressor,
    select_model,
    summarize_compressor,
    DEFAULT_MODELS,
)


class TestBudget(unittest.TestCase):
    def test_effective_input(self):
        b = Budget(max_tokens=1000, reserve_for_completion=200, safety_margin=0.1)
        self.assertEqual(b.effective_input(), 700)

    def test_default_margin(self):
        b = Budget(max_tokens=4000)
        # 5% margin = 200, no reserve → 3800
        self.assertEqual(b.effective_input(), 3800)

    def test_invalid_max_tokens(self):
        with self.assertRaises(ValueError):
            Budget(max_tokens=0)

    def test_invalid_margin(self):
        with self.assertRaises(ValueError):
            Budget(max_tokens=1000, safety_margin=0.6)


class TestMessage(unittest.TestCase):
    def test_render_basic(self):
        m = Message(Role.USER, "hi")
        self.assertEqual(m.render(), {"role": "user", "content": "hi"})

    def test_render_with_name(self):
        m = Message(Role.TOOL, "result", name="calc")
        out = m.render()
        self.assertEqual(out["name"], "calc")


class TestContext(unittest.TestCase):
    def test_empty_total_tokens(self):
        ctx = Context()
        self.assertEqual(ctx.total_tokens(), 0)

    def test_add_messages_and_blocks(self):
        ctx = Context(budget=Budget(max_tokens=10_000))
        ctx.add_system("sys")
        ctx.add_user("hello")
        ctx.add_block("docs", "doc1 doc2", importance=0.7)
        self.assertEqual(len(ctx.messages), 2)
        self.assertEqual(len(ctx.blocks), 1)
        self.assertGreater(ctx.total_tokens(), 0)

    def test_render_messages_no_blocks(self):
        ctx = Context()
        ctx.add_system("be helpful")
        ctx.add_user("hi")
        self.assertEqual(
            ctx.render_messages(),
            [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_render_messages_with_blocks(self):
        ctx = Context()
        ctx.add_user("q")
        ctx.add_block("docs", "context here", importance=0.8)
        out = ctx.render_messages()
        # user message + block-as-user-message
        self.assertEqual(len(out), 2)
        self.assertIn("BEGIN docs", out[1]["content"])

    def test_fit_protects_system_and_last_turn(self):
        ctx = Context(budget=Budget(max_tokens=200))
        ctx.add_system("ALWAYS_KEEP_THIS_SYSTEM_PROMPT")
        # Add many filler messages
        for i in range(20):
            ctx.add_user(f"filler message number {i} " + "x" * 1000)
        ctx.fit()
        msgs = ctx.messages
        # System message preserved exactly
        self.assertEqual(msgs[0].role, Role.SYSTEM)
        self.assertEqual(msgs[0].content, "ALWAYS_KEEP_THIS_SYSTEM_PROMPT")
        # Last user turn preserved (somewhat)
        self.assertIn("filler message number", msgs[-1].content)

    def test_fit_compresses_blocks(self):
        ctx = Context(budget=Budget(max_tokens=500))
        long_text = "lorem ipsum " * 500  # ~6000 tokens
        ctx.add_user("q")
        ctx.add_block("docs", long_text, importance=0.5, compressible=True)
        before = ctx.total_tokens()
        ctx.fit()
        after = ctx.total_tokens()
        self.assertLess(after, before)

    def test_fit_drops_low_importance_blocks(self):
        ctx = Context(budget=Budget(max_tokens=100))
        ctx.add_user("q")
        ctx.add_block("low", "x" * 5000, importance=0.1)
        ctx.add_block("high", "y" * 50, importance=0.9)
        ctx.fit()
        # High-importance block survives
        names = [b.name for b in ctx.blocks]
        self.assertIn("high", names)
        self.assertNotIn("low", names)

    def test_custom_tokenizer(self):
        def word_counter(text: str) -> int:
            return len(text.split())

        ctx = Context(budget=Budget(max_tokens=10), tokenizer=word_counter)
        ctx.add_user("one two three four five six")
        ctx.add_user("seven eight nine ten eleven")
        ctx.fit()
        # Each message should be roughly halved
        for m in ctx.messages:
            self.assertLessEqual(len(m.content.split()), 10)

    def test_custom_compressor(self):
        def short_summary(text: str, target_tokens: int) -> str:
            return f"summary({len(text)})"

        ctx = Context(
            budget=Budget(max_tokens=300),
            compressor=short_summary,
        )
        ctx.add_user("q")
        ctx.add_block("docs", "x" * 5000, importance=0.5, compressible=True)
        ctx.fit()
        block = ctx.blocks[0]
        self.assertTrue(block.text.startswith("summary("))


class TestHeadTailCompressor(unittest.TestCase):
    def test_short_text_unchanged(self):
        text = "short text"
        self.assertEqual(head_tail_compressor(text, 100), text)

    def test_long_text_truncated(self):
        text = "word " * 1000
        out = head_tail_compressor(text, 50)
        self.assertIn("[... ", out)
        self.assertIn(" tokens omitted", out)


class TestSummarizeCompressor(unittest.TestCase):
    def test_marker_format(self):
        comp = summarize_compressor("[x]")
        out = comp("a" * 1000, 50)
        self.assertIn("[x]", out)
        self.assertIn("→ 50", out)


class TestSelectModel(unittest.TestCase):
    def test_default_registry(self):
        m = select_model(1000)
        self.assertIsNotNone(m)
        self.assertGreaterEqual(m.context_length, 1000)

    def test_required_tokens_filter(self):
        # No default model has > 500_000 tokens
        m = select_model(500_000)
        self.assertIsNone(m)

    def test_prefer(self):
        m = select_model(1000, prefer="claude-haiku-4")
        self.assertEqual(m.name, "claude-haiku-4")

    def test_prefer_overridden_if_doesnt_fit(self):
        m = select_model(500_000, prefer="gpt-4o-mini")
        # gpt-4o-mini has 128K, doesn't fit → prefer ignored, no candidate
        self.assertIsNone(m)

    def test_cost_filter(self):
        # claude-opus is most expensive; filter it out
        m = select_model(1000, max_cost_per_1k=0.001)
        self.assertIsNotNone(m)
        self.assertLessEqual(m.cost_per_1k_input, 0.001)

    def test_custom_registry(self):
        reg = {
            "tiny-1": ModelSpec("tiny-1", 100),
            "medium-1": ModelSpec("medium-1", 1000, cost_per_1k_input=0.01),
            "large-1": ModelSpec("large-1", 10_000, cost_per_1k_input=0.1),
        }
        m = select_model(500, registry=reg)
        self.assertEqual(m.name, "medium-1")
        m = select_model(5000, registry=reg)
        self.assertEqual(m.name, "large-1")
        m = select_model(20_000, registry=reg)
        self.assertIsNone(m)


if __name__ == "__main__":
    unittest.main(verbosity=2)