"""Bibliographic + people-analytics tagging for each paper.

Tags are extracted once per PDF (keyed by file hash) and cached in
data/paper_metadata.json, so re-running ingestion doesn't re-bill you.
"""

import hashlib
import json
import re
from pathlib import Path

from .chunking import Page
from .config import settings
from .llm import structured_call

PA_TOPICS = [
    "attrition & retention",
    "engagement & wellbeing",
    "performance & productivity",
    "recruitment & selection",
    "diversity, equity & inclusion",
    "compensation & rewards",
    "leadership & management",
    "learning & development",
    "careers & internal mobility",
    "workforce planning",
    "organisational network analysis",
    "hybrid & remote work",
    "culture & climate",
    "HR analytics adoption & maturity",
    "ethics, privacy & algorithmic management",
    "measurement & methods",
    "other",
]

_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "year": {"type": "string", "description": "Four-digit year, or empty if unknown"},
        "venue": {"type": "string", "description": "Journal, book or publisher"},
        "doi": {"type": "string"},
        "paper_type": {
            "type": "string",
            "enum": [
                "empirical-quantitative",
                "empirical-qualitative",
                "mixed-methods",
                "meta-analysis",
                "systematic-review",
                "narrative-review",
                "theory",
                "book-or-chapter",
                "practitioner-or-report",
                "other",
            ],
        },
        "topics": {"type": "array", "items": {"type": "string", "enum": PA_TOPICS}},
        "theories_frameworks": {"type": "array", "items": {"type": "string"}},
        "constructs": {"type": "array", "items": {"type": "string"}},
        "methods": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string", "description": "Sample, industry, country"},
        "summary": {"type": "string", "description": "2-3 sentences: question and main finding"},
    },
    "required": [
        "title", "authors", "year", "venue", "doi", "paper_type", "topics",
        "theories_frameworks", "constructs", "methods", "setting", "summary",
    ],
    "additionalProperties": False,
}

_SYSTEM = (
    "You catalogue academic and practitioner literature for a people analytics "
    "research library. Extract bibliographic details and tag the paper's "
    "theories/frameworks (e.g. Job Demands-Resources, Social Exchange Theory), "
    "constructs (e.g. turnover intention, work engagement), and methods "
    "(e.g. survival analysis, multilevel modelling, SEM, interviews). "
    "Use only what the text supports; leave fields empty rather than guessing."
)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def load_cache() -> dict:
    if settings.metadata_cache.exists():
        return json.loads(settings.metadata_cache.read_text())
    return {}


def save_cache(cache: dict) -> None:
    settings.metadata_cache.parent.mkdir(parents=True, exist_ok=True)
    settings.metadata_cache.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def fallback_metadata(path: Path, pages: list[Page]) -> dict:
    """Metadata without an API call: filename as title, year from the text."""
    first = pages[0].text[:3000] if pages else ""
    year = re.search(r"\b(19[89]\d|20[0-4]\d)\b", first)
    return {
        "title": path.stem.replace("_", " ").replace("-", " "),
        "authors": [],
        "year": year.group(1) if year else "",
        "venue": "",
        "doi": "",
        "paper_type": "other",
        "topics": [],
        "theories_frameworks": [],
        "constructs": [],
        "methods": [],
        "setting": "",
        "summary": "",
    }


def extract_metadata(path: Path, pages: list[Page]) -> dict:
    # Front matter + abstract + intro carry nearly all the bibliographic signal.
    excerpt = "\n\n".join(p.text for p in pages[:3])[:12000]
    if len(pages) > 3:
        # The discussion/conclusion is where frameworks and findings are summarised.
        excerpt += "\n\n[...]\n\n" + "\n\n".join(p.text for p in pages[-3:])[:6000]
    prompt = (
        f"File name: {path.name}\nTopic folder: {path.parent.name}\n\n"
        f"<paper_excerpt>\n{excerpt}\n</paper_excerpt>"
    )
    result = structured_call(
        model=settings.utility_model,
        effort=settings.utility_effort,
        system=_SYSTEM,
        prompt=prompt,
        schema=_SCHEMA,
    )
    if result is None:
        return fallback_metadata(path, pages)
    if not result.get("title"):
        result["title"] = path.stem
    return result


def cite_label(meta: dict) -> str:
    """Short in-text style label, e.g. 'Marler & Boudreau (2017)'."""
    surnames = [a.split(",")[0].split()[-1] for a in meta.get("authors", []) if a.strip()]
    if not surnames:
        who = meta.get("title", "Untitled")[:60]
    elif len(surnames) == 1:
        who = surnames[0]
    elif len(surnames) == 2:
        who = f"{surnames[0]} & {surnames[1]}"
    else:
        who = f"{surnames[0]} et al."
    year = meta.get("year") or "n.d."
    return f"{who} ({year})"


def reference_entry(meta: dict) -> str:
    """APA-ish reference line."""
    authors = meta.get("authors") or []
    if len(authors) > 6:
        authors_str = ", ".join(authors[:6]) + ", et al."
    else:
        authors_str = ", ".join(authors)
    year = f"({meta.get('year') or 'n.d.'})."
    title = f"*{meta.get('title') or 'Untitled'}*."
    lead = [authors_str, year, title] if authors_str else [title, year]
    parts = [p for p in lead + [meta.get("venue", "")] if p]
    entry = " ".join(parts)
    if meta.get("doi"):
        doi = meta["doi"]
        entry += f" https://doi.org/{doi}" if not doi.startswith("http") else f" {doi}"
    return entry
