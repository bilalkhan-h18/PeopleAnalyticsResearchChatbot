"""Question -> search queries -> hybrid retrieval -> cited, streamed answer."""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import anthropic

from .config import settings, supports_effort
from .llm import get_client, structured_call
from .metadata import reference_entry
from .prompts import DEFAULT_MODE, MODES, PLANNER_SCHEMA, PLANNER_SYSTEM, build_system_prompt
from .store import Hit, LiteratureStore

log = logging.getLogger(__name__)

# Models documented to accept `fallbacks: "default"` (server-side refusal fallback).
_FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5-1")


@dataclass
class Event:
    kind: str  # "status" | "delta"
    text: str


@dataclass
class AnswerResult:
    markdown: str
    plain_text: str
    sources_md: str = ""
    queries: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


def plan_queries(question: str, history: list[dict]) -> list[str]:
    recent = history[-6:]
    convo = "\n\n".join(f"{m['role'].upper()}: {m['content'][:1500]}" for m in recent)
    prompt = (f"<conversation>\n{convo}\n</conversation>\n\n" if convo else "") + f"<question>\n{question}\n</question>"
    try:
        result = structured_call(
            model=settings.utility_model,
            effort=settings.utility_effort,
            system=PLANNER_SYSTEM,
            prompt=prompt,
            schema=PLANNER_SCHEMA,
        )
    except anthropic.APIError as exc:
        log.warning("query planning failed, searching with the raw question: %s", exc)
        result = None
    queries = [q.strip() for q in (result or {}).get("queries", []) if q.strip()][:5]
    # Always search the user's own wording too; it catches exact terms the rewrite may drop.
    return [question] + [q for q in queries if q.lower() != question.lower()]


def search_result_blocks(hits: list[Hit]) -> list[dict]:
    return [
        {
            "type": "search_result",
            "source": h.id,
            "title": f"{h.metadata['label']} - {h.metadata['title']} ({h.pages})",
            "content": [{"type": "text", "text": h.text}],
            "citations": {"enabled": True},
        }
        for h in hits
    ]


def build_user_content(question: str, mode: str, hits: list[Hit]) -> list[dict]:
    instructions = MODES.get(mode, MODES[DEFAULT_MODE])
    note = "" if hits else (
        "\n\nNo excerpts were retrieved from the library for this question "
        "(the library may be empty or not cover it)."
    )
    return search_result_blocks(hits) + [
        {"type": "text", "text": f"<task>\n{instructions}\n</task>\n\n<question>\n{question}\n</question>{note}"}
    ]


def format_answer(content_blocks, hits: list[Hit]) -> tuple[str, str, str]:
    """Turn the model's cited text blocks into (answer_markdown, plain_text, sources_markdown).

    Sources are numbered per paper in order of first citation; the markers
    [1][2] are appended to the sentence/span the citation supports.
    """
    by_id = {h.id: h for h in hits}
    paper_number: dict[str, int] = {}
    cited_quotes: dict[str, list[tuple[str, str]]] = {}

    parts, plain = [], []
    for block in content_blocks:
        if getattr(block, "type", None) != "text":
            continue
        plain.append(block.text)
        markers = []
        for c in getattr(block, "citations", None) or []:
            hit = by_id.get(getattr(c, "source", None))
            if hit is None:
                continue
            pid = hit.paper_id
            if pid not in paper_number:
                paper_number[pid] = len(paper_number) + 1
            n = paper_number[pid]
            if n not in markers:
                markers.append(n)
            quote = " ".join(c.cited_text.split())
            if len(quote) > 400:
                quote = quote[:400] + "..."
            cited_quotes.setdefault(pid, []).append((hit.pages, quote))
        parts.append(block.text + "".join(f"<sup>[{n}]</sup>" for n in markers))

    papers = {h.paper_id: json.loads(h.metadata["paper_json"]) for h in hits}
    lines = []
    if paper_number:
        lines.append("### Cited in this answer")
        for pid, n in sorted(paper_number.items(), key=lambda kv: kv[1]):
            lines.append(f"**[{n}]** {reference_entry(papers[pid])}")
            seen = set()
            for pages, quote in cited_quotes.get(pid, []):
                if quote in seen:
                    continue
                seen.add(quote)
                lines.append(f"> {pages}: \"{quote}\"\n")
            lines.append("")
    uncited = [pid for pid in dict.fromkeys(h.paper_id for h in hits) if pid not in paper_number]
    if uncited:
        lines.append("### Also retrieved (not cited)")
        lines.extend(f"- {reference_entry(papers[pid])}" for pid in uncited)

    references = ""
    if paper_number:
        references = "\n\n---\n**References**\n\n" + "\n".join(
            f"[{n}] {reference_entry(papers[pid])}  " for pid, n in sorted(paper_number.items(), key=lambda kv: kv[1])
        )
    return "".join(parts) + references, "".join(plain), "\n".join(lines)


class ResearchAssistant:
    def __init__(self, store: LiteratureStore | None = None):
        self.store = store or LiteratureStore()

    def ask(
        self,
        question: str,
        history: list[dict],
        mode: str = DEFAULT_MODE,
        allow_general_knowledge: bool = True,
        year_min: int | None = None,
        n_sources: int = settings.default_sources,
    ) -> Iterator[Event | AnswerResult]:
        """Yields Events while working and an AnswerResult last.

        `history` is prior turns as plain {"role", "content"} strings. Retrieved
        excerpts are sent only with the current question, which keeps each
        follow-up request small.
        """
        yield Event("status", "Planning search queries...")
        queries = plan_queries(question, history)

        yield Event("status", f"Searching your library ({len(queries)} queries)...")
        hits = self.store.search(queries, k=n_sources, year_min=year_min)

        yield Event("status", f"Reading {len(hits)} excerpts from {len({h.paper_id for h in hits})} papers...")

        model = settings.answer_model
        kwargs: dict = {}
        if supports_effort(model):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": settings.answer_effort}
        if model in _FALLBACK_MODELS:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        messages = list(history) + [{"role": "user", "content": build_user_content(question, mode, hits)}]
        streamed = ""
        with get_client().beta.messages.stream(
            model=model,
            max_tokens=settings.answer_max_tokens,
            system=build_system_prompt(allow_general_knowledge),
            messages=messages,
            **kwargs,
        ) as stream:
            for event in stream:
                if event.type == "text":
                    streamed += event.text
                    yield Event("delta", streamed)
            final = stream.get_final_message()

        if final.stop_reason == "refusal":
            text = "The model declined to answer this request. Try rephrasing the question."
            yield AnswerResult(markdown=text, plain_text=text, queries=queries, hits=hits)
            return

        markdown, plain, sources_md = format_answer(final.content, hits)
        if final.stop_reason == "max_tokens":
            note = "\n\n*(Answer cut off at the output limit - raise ANSWER_MAX_TOKENS or ask a narrower question.)*"
            markdown += note
            plain += note
        queries_md = "### Search queries used\n" + "\n".join(f"- {q}" for q in queries)
        yield AnswerResult(
            markdown=markdown,
            plain_text=plain,
            sources_md=f"{sources_md}\n\n{queries_md}",
            queries=queries,
            hits=hits,
        )
