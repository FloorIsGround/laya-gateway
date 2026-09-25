from __future__ import annotations

from laya_gateway.context import ContextSizer


def test_preflight_matches_laya_034_sequence_lengths():
    # Golden lengths from laya.common.build_sequence with the pinned tokenizer.
    sizer = ContextSizer()
    payload = {
        "state": {"body": "Refund café [MASK] ☀", "tags": ["a", 2]},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Where? [MASK]",
                "criteria": {"billing": {"reason": "payments"}, "other": "everything else"},
            },
            "risk": {
                "type": "noul",
                "instructions": "Threat?",
                "criteria": {"true": {"signal": "cancel"}},
            },
        },
    }
    prepared = sizer.prepare(payload)
    assert prepared is not None
    assert prepared.lengths(192) == (48, 92)
    assert prepared.lengths(256) == (48, 92)
    assert len(sizer.fingerprint) == 64


def test_invalid_shape_is_left_to_worker():
    sizer = ContextSizer()
    assert sizer.prepare({"state": "x", "questions": {}}) is None
    assert sizer.prepare({"state": "x", "questions": {"a": {"type": "choice"}}}) is None
