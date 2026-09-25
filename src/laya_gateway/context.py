"""Small, CPU-only preflight using Laya 0.3.4's exact input formatting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from tokenizers import Tokenizer


@dataclass(frozen=True)
class PreparedRequest:
    state_tokens: int
    # (instruction-token count, option-token counts), one row per question.
    questions: tuple[tuple[int, tuple[int, ...]], ...]

    def lengths(self, head_max_length: int) -> tuple[int, int]:
        """Return (longest untruncated row, sum of row lengths)."""
        lengths = []
        for instructions, option_tokens in self.questions:
            opt_budget = head_max_length - sum(option_tokens)
            if opt_budget < 16:
                per = max(4, (head_max_length - 16) // max(1, len(option_tokens)))
                option_tokens = tuple(min(size, per) for size in option_tokens)
                opt_budget = head_max_length - sum(option_tokens)
            # [CLS] instructions [SEP] options [SEP] state [SEP]
            lengths.append(
                4 + min(instructions, max(8, opt_budget)) + sum(option_tokens) + self.state_tokens
            )
        return max(lengths), sum(lengths)


class ContextSizer:
    """Count rows without loading Torch, Transformers, or model weights.

    Good: call prepare() once per valid-looking request, then lengths() for each
    worker's head budget. Bad: estimate from JSON bytes or trust a client token
    count: both can route a truncated request to a short-context worker.
    """

    def __init__(self, tokenizer_path: str | None = None):
        path = tokenizer_path or str(files("laya_gateway").joinpath("laya-tokenizer.json"))
        self.tokenizer = Tokenizer.from_file(path)
        self.fingerprint = hashlib.sha256(self.tokenizer.to_str().encode("utf-8")).hexdigest()
        self.mask_token = "[MASK]"
        if self.tokenizer.token_to_id(self.mask_token) is None:
            raise RuntimeError("Laya tokenizer is missing [MASK]")

    def _length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def prepare(self, payload: dict[str, Any]) -> PreparedRequest | None:
        """Return None for invalid shapes, leaving worker validation authoritative."""
        state, questions = payload.get("state"), payload.get("questions")
        if not isinstance(state, (str, dict, list)) or not isinstance(questions, dict):
            return None
        if not 1 <= len(questions) <= 32:
            return None
        rows = []
        for question in questions.values():
            if not isinstance(question, dict):
                return None
            kind, instructions, criteria = (
                question.get("type"), question.get("instructions"), question.get("criteria")
            )
            if (
                kind not in ("choice", "score", "noul")
                or not isinstance(instructions, str)
                or not 1 <= len(instructions) <= 4000
            ):
                return None
            if kind == "choice":
                if not isinstance(criteria, (dict, list)) or not 2 <= len(criteria) <= 20:
                    return None
                if isinstance(criteria, dict):
                    options = [
                        key if value is None or value == "" else f"{key}: {_render_criterion(value)}"
                        for key, value in criteria.items()
                    ]
                else:
                    if not all(isinstance(value, str) for value in criteria):
                        return None
                    # Agent._to_internal converts a list to a dict before rendering.
                    options = list(dict.fromkeys(criteria))
            elif kind == "score":
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 20:
                    return None
                options = [f"level {i}: {_render_criterion(value)}" for i, value in enumerate(criteria)]
            else:
                if criteria is not None and not isinstance(criteria, dict):
                    return None
                criteria = criteria or {}
                false_crit, true_crit = criteria.get("false"), criteria.get("true")
                options = [
                    "false: " + (
                        _render_criterion(false_crit)
                        if false_crit not in (None, "") else "no, the statement does not hold"
                    ),
                    "true: " + (
                        _render_criterion(true_crit)
                        if true_crit not in (None, "") else "yes, the statement holds"
                    ),
                ]
            instruction_len = self._length(
                f"{kind} question: {instructions.replace(self.mask_token, ' ')}"
            )
            option_lengths = tuple(
                1 + min(48, self._length(" " + option.replace(self.mask_token, " ")))
                for option in options
            )
            rows.append((instruction_len, option_lengths))
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return PreparedRequest(
            self._length(state_text.replace(self.mask_token, " ")),
            tuple(rows),
        )


def _render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)
