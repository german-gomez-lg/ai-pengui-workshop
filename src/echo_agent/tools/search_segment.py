"""Semantic search over audience segments."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel
from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

class SegmentSearchArgs(BaseModel):
    """Segment search inputs."""

    query: str
    top_k: int | None = None


class SegmentSearchResult(BaseModel):
    """Segment search outputs."""

    query: str
    results: list[dict[str, str]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_segments() -> list[dict[str, str]]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "audience_segments.csv"
    segments: list[dict[str, str]] = []
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            segment_id = (row.get("segment_id") or "").strip()
            description = (row.get("description") or "").strip()
            if segment_id and description:
                segments.append({"segment_id": segment_id, "description": description})
    return segments


def _build_idf(docs: list[list[str]]) -> dict[str, float]:
    df: dict[str, int] = {}
    for tokens in docs:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    total = len(docs)
    return {term: math.log((total + 1) / (count + 1)) + 1 for term, count in df.items()}


def _tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = sum(counts.values())
    return {term: (count / total) * idf.get(term, 0.0) for term, count in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for term, weight in a.items():
        dot += weight * b.get(term, 0.0)
    norm_a = math.sqrt(sum(val * val for val in a.values()))
    norm_b = math.sqrt(sum(val * val for val in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@tool(desc="Semantic search over audience segments", side_effects="read", tags=["planner"])
async def search_segment(args: SegmentSearchArgs, ctx: ToolContext) -> SegmentSearchResult:
    query = args.query.strip()
    top_k = args.top_k or 3
    top_k = max(1, min(top_k, 5))

    segments = _load_segments()
    docs = [_tokenize(segment["description"]) for segment in segments]
    idf = _build_idf(docs)

    query_vec = _tfidf(_tokenize(query), idf)
    scored: list[tuple[float, dict[str, str]]] = []

    for segment, doc_tokens in zip(segments, docs):
        doc_vec = _tfidf(doc_tokens, idf)
        score = _cosine(query_vec, doc_vec)
        scored.append((score, segment))

    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, str]] = []
    for score, segment in scored[:top_k]:
        results.append(
            {
                "segment_id": segment["segment_id"],
                "description": segment["description"],
                "score": f"{score:.4f}",
            }
        )

    return SegmentSearchResult(query=query, results=results)
