from __future__ import annotations

import json
import re
from collections.abc import Sequence

import requests

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


def _gemini_error_message(payload: dict | None, status_code: int) -> str:
    detail = ""
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            detail = str(error.get("message", "") or error.get("status", "")).strip()
    return f"Gemini API error {status_code}: {detail}".rstrip(": ")


def _gemini_answer(question: str, passages: Sequence[str]) -> dict:
    if not config.GEMINI_API_KEY:
        return extractive(question, passages, "no valid Gemini credentials")

    url = f"{config.GEMINI_BASE_URL}/models/{config.GEMINI_MODEL}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": build_prompt(question, passages)}]}],
        "generationConfig": {"maxOutputTokens": 16000, "temperature": 0, "topP": 0.95},
    }

    try:
        response = requests.post(url, params={"key": config.GEMINI_API_KEY}, json=payload, timeout=60)
        data = response.json() if response.content else {}
    except requests.Timeout:
        return extractive(question, passages, "Gemini API timeout")
    except requests.RequestException:
        return extractive(question, passages, "Gemini API unreachable")
    except json.JSONDecodeError:
        return extractive(question, passages, "Gemini API returned invalid JSON")

    if response.status_code in {401, 403}:
        return extractive(question, passages, "no valid Gemini credentials")
    if response.status_code == 429:
        return extractive(question, passages, "Gemini API rate limited")
    if response.status_code >= 500:
        return extractive(question, passages, _gemini_error_message(data, response.status_code))
    if not response.ok:
        return extractive(question, passages, _gemini_error_message(data, response.status_code))

    candidates = data.get("candidates") if isinstance(data, dict) else None
    candidate = candidates[0] if candidates else None
    if not isinstance(candidate, dict):
        return extractive(question, passages, "Gemini answer missing")

    finish_reason = str(candidate.get("finishReason", "")).lower()
    if finish_reason in {"recitation", "blocked"}:
        return extractive(question, passages, "model declined")

    content = candidate.get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    answer = "".join(part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")).strip()
    if not answer:
        return extractive(question, passages, "Gemini answer empty")

    valid, invalid = verify_citations(answer, len(passages))
    if invalid or (answer != NOT_FOUND and not valid):
        return extractive(question, passages, "answer failed citation check")
    return {"answer": answer, "mode": "gemini", "model": config.GEMINI_MODEL, "citations": valid, "reason": ""}


def synthesize(question: str, passages: Sequence[str]) -> dict:
    if not passages:
        return extractive(question, passages, "no passages retrieved")
    return _gemini_answer(question, passages)


if __name__ == "__main__":
    assert verify_citations("A [1]. B [2][5].", 3) == ([1, 2], [5])
    assert "[1]" in build_prompt("q", ["p"])
    out = extractive("where do frogs live", ["Cats sleep a lot here. Frogs live in ponds and wet grass.", "Rates rose."], "test")
    assert out["answer"] == "- Frogs live in ponds and wet grass. [1]" and out["citations"] == [1]
    print("offline checks ok")
    result = synthesize("How long do cats sleep?",
                        ["Domestic cats sleep up to sixteen hours a day.", "Interest rates rose."])
    print(result)
