from __future__ import annotations

import re
from collections.abc import Sequence

import anthropic

from core import config
from core.bm25 import tokenize

NOT_FOUND = "Not found in the document."

SYSTEM = (
    "You format answers for a document search tool. You receive a question and numbered passages "
    "retrieved from the user's documents. Answer using only facts stated in those passages. "
    "After every sentence, cite the passage numbers it relies on, like [1] or [2][3]. "
    "Do not add outside knowledge, do not guess, and do not cite a number that was not given. "
    f"If the passages do not answer the question, reply exactly: {NOT_FOUND} "
    "Keep the answer under 150 words and use plain sentences or a short bullet list."
)

_CITE = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else anthropic.Anthropic()
    return _client


def build_prompt(question: str, passages: Sequence[str]) -> str:
    blocks = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"<passages>\n{blocks}\n</passages>\n\nQuestion: {question}"


def verify_citations(answer: str, n_passages: int) -> tuple[list[int], list[int]]:
    cited = sorted({int(m) for m in _CITE.findall(answer)})
    valid = [c for c in cited if 1 <= c <= n_passages]
    invalid = [c for c in cited if not 1 <= c <= n_passages]
    return valid, invalid


def best_sentence(question: str, passage: str) -> tuple[float, str]:
    terms = set(tokenize(question))
    sentences = [x.strip() for x in _SENTENCE.split(passage) if len(x.split()) >= 4]
    if not sentences:
        return 0.0, passage.strip()[:300]
    scored = [(sum(t in terms for t in set(tokenize(x))) / (1 + len(x) / 400), x) for x in sentences]
    return max(scored, key=lambda p: p[0])


def extractive(question: str, passages: Sequence[str], reason: str) -> dict:
    lines, cites = [], []
    for i, p in enumerate(passages[:5], start=1):
        score, sentence = best_sentence(question, p)
        if score > 0:
            lines.append(f"- {sentence[:400]} [{i}]")
            cites.append(i)
        if len(lines) == 3:
            break
    if not lines:
        return {"answer": NOT_FOUND, "mode": "extractive", "reason": reason, "citations": []}
    return {"answer": "\n".join(lines), "mode": "extractive", "reason": reason, "citations": cites}


def synthesize(question: str, passages: Sequence[str]) -> dict:
    if not passages:
        return extractive(question, passages, "no passages retrieved")
    try:
        response = _get_client().beta.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(question, passages)}],
        )
    except anthropic.AuthenticationError:
        return extractive(question, passages, "no valid Anthropic credentials")
    except anthropic.RateLimitError:
        return extractive(question, passages, "Claude API rate limited")
    except anthropic.APIStatusError as e:
        detail = e.body.get("error", {}).get("message", "") if isinstance(e.body, dict) else ""
        return extractive(question, passages, f"Claude API error {e.status_code}: {detail[:160]}".rstrip(": "))
    except anthropic.APIConnectionError:
        return extractive(question, passages, "Claude API unreachable")
    except Exception as e:
        return extractive(question, passages, f"Claude unavailable: {type(e).__name__}")

    if response.stop_reason == "refusal":
        return extractive(question, passages, "model declined")
    answer = "".join(b.text for b in response.content if b.type == "text").strip()
    valid, invalid = verify_citations(answer, len(passages))
    if invalid or (answer != NOT_FOUND and not valid):
        return extractive(question, passages, "answer failed citation check")
    return {"answer": answer, "mode": "claude", "model": response.model, "citations": valid, "reason": ""}


if __name__ == "__main__":
    assert verify_citations("A [1]. B [2][5].", 3) == ([1, 2], [5])
    assert "[1]" in build_prompt("q", ["p"])
    out = extractive("where do frogs live", ["Cats sleep a lot here. Frogs live in ponds and wet grass.", "Rates rose."], "test")
    assert out["answer"] == "- Frogs live in ponds and wet grass. [1]" and out["citations"] == [1]
    print("offline checks ok")
    result = synthesize("How long do cats sleep?",
                        ["Domestic cats sleep up to sixteen hours a day.", "Interest rates rose."])
    print(result)
