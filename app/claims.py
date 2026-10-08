"""Claim-level detection: split an answer into atomic, self-contained factual claims."""
from __future__ import annotations

import logging
import re

from .generator import Completer, SLMError
from .verifier import make_hypothesis

logger = logging.getLogger("knowsure")

CLAIMS_PROMPT = """Split the answer below into short, self-contained factual claims.
Rules:
- Each claim must make sense on its own: repeat the subject instead of using pronouns.
- Keep only factual claims; drop opinions, hedges and filler.
- At most {max_claims} claims, one per line, with no numbering or bullets.
- If the answer contains no factual claim (for example "I don't know"), reply exactly: NONE

Question: {question}
Answer: {answer}"""

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_COMPOUND = re.compile(r",|;|\band\b|\bbut\b|\bwhile\b|\bwhereas\b", re.IGNORECASE)


def needs_splitting(answer: str) -> bool:
    """Short single-fact answers are already one claim; skip the extra model call for them."""
    return len(answer.split()) > 12 or bool(_COMPOUND.search(answer))


def split_claims(llm: Completer, question: str, answer: str, max_claims: int) -> list[str]:
    whole = make_hypothesis(question, answer)
    if max_claims <= 1 or not needs_splitting(answer):
        return [whole]
    try:
        reply = llm.complete(CLAIMS_PROMPT.format(max_claims=max_claims, question=question, answer=answer), 300)
    except SLMError as exc:  # claim splitting is an enhancement: fall back to checking the whole answer
        logger.warning("Claim splitting failed, verifying the whole answer: %s", exc)
        return [whole]
    if reply.strip().upper().startswith("NONE"):
        return [whole]
    claims = [_LIST_MARKER.sub("", line).strip() for line in reply.splitlines()]
    claims = list(dict.fromkeys(c for c in claims if len(c.split()) >= 3))[:max_claims]
    return claims or [whole]
