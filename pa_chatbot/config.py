"""Settings, read from environment variables (or a local .env file)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    # Model that writes the final, cited answer.
    answer_model: str = os.getenv("ANSWER_MODEL", "claude-opus-5")
    # Model for small jobs: planning search queries and tagging papers at ingest time.
    utility_model: str = os.getenv("UTILITY_MODEL", "claude-opus-5")
    # low | medium | high | xhigh | max. Higher = deeper reasoning, more output tokens.
    answer_effort: str = os.getenv("ANSWER_EFFORT", "high")
    utility_effort: str = os.getenv("UTILITY_EFFORT", "low")
    answer_max_tokens: int = int(os.getenv("ANSWER_MAX_TOKENS", "32000"))

    corpus_dir: Path = field(default_factory=lambda: _path("CORPUS_DIR", "corpus"))
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", "data"))
    collection_name: str = os.getenv("COLLECTION_NAME", "pa_literature")

    # Chunking (characters; ~4 characters per token for English prose).
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "3600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "400"))

    # Retrieval.
    candidates_per_query: int = int(os.getenv("CANDIDATES_PER_QUERY", "30"))
    default_sources: int = int(os.getenv("DEFAULT_SOURCES", "12"))
    max_chunks_per_paper: int = int(os.getenv("MAX_CHUNKS_PER_PAPER", "3"))

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def metadata_cache(self) -> Path:
        return self.data_dir / "paper_metadata.json"


settings = Settings()


def supports_effort(model: str) -> bool:
    """Haiku 4.5 rejects `effort`; current Opus/Sonnet/Fable models accept it."""
    return not model.startswith("claude-haiku")
