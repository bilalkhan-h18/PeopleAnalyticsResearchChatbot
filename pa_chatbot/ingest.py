"""Build or update the literature index.

    python -m pa_chatbot.ingest                 # add new/changed PDFs in corpus/
    python -m pa_chatbot.ingest --rebuild       # wipe and re-index everything
    python -m pa_chatbot.ingest --no-llm        # skip Claude tagging (free, weaker metadata)

Sub-folders of corpus/ are kept as a "collection" label (e.g. corpus/attrition/);
folders starting with "_" are skipped.
Only new or changed files are processed, so re-running is cheap.
"""

import argparse
import json
import shutil
import sys

from .chunking import chunk_pages, read_pdf_pages, strip_reference_list
from .config import settings
from .metadata import cite_label, extract_metadata, fallback_metadata, file_hash, load_cache, save_cache
from .store import LiteratureStore


def paper_header(meta: dict) -> str:
    bits = [f"{meta.get('title', '')} ({meta.get('year') or 'n.d.'})"]
    for label, key in [("Frameworks", "theories_frameworks"), ("Constructs", "constructs"), ("Methods", "methods"), ("Topics", "topics")]:
        if meta.get(key):
            bits.append(f"{label}: {', '.join(meta[key])}")
    return "\n".join(bits)


def ingest(rebuild: bool = False, use_llm: bool = True) -> None:
    corpus = settings.corpus_dir
    # Folders starting with "_" (e.g. corpus/_removed/) are ignored.
    pdfs = sorted(
        p for p in corpus.rglob("*")
        if p.suffix.lower() == ".pdf" and not any(part.startswith("_") for part in p.relative_to(corpus).parts)
    )
    if not pdfs:
        sys.exit(f"No PDFs found under {corpus}. Add papers (sub-folders are fine) and re-run.")

    if rebuild and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    store = LiteratureStore()
    cache = load_cache()
    indexed = store.paper_ids()

    current_ids = set()
    added = skipped = failed = 0
    for n, path in enumerate(pdfs, start=1):
        pid = file_hash(path)
        current_ids.add(pid)
        rel = path.relative_to(corpus)
        if pid in indexed:
            skipped += 1
            continue
        print(f"[{n}/{len(pdfs)}] {rel}")
        try:
            pages = read_pdf_pages(path)
        except Exception as exc:  # corrupt or encrypted PDFs shouldn't stop the run
            print(f"    ! could not read: {exc}")
            failed += 1
            continue
        if not pages:
            print("    ! no extractable text (scanned PDF? run OCR first, e.g. `ocrmypdf`)")
            failed += 1
            continue

        meta = cache.get(pid)
        if meta is None:
            meta = extract_metadata(path, pages) if use_llm else fallback_metadata(path, pages)
            cache[pid] = meta
            save_cache(cache)
        print(f"    {cite_label(meta)} - {meta.get('title', '')[:80]}")

        body = strip_reference_list(pages)
        chunks = chunk_pages(body, settings.chunk_chars, settings.chunk_overlap)
        header = paper_header(meta)
        year = meta.get("year", "")
        base_meta = {
            "paper_id": pid,
            "file": str(rel),
            "collection": rel.parts[0] if len(rel.parts) > 1 else "",
            "title": meta.get("title", "")[:500],
            "label": cite_label(meta),
            "year_int": int(year) if year.isdigit() else 0,
            "paper_type": meta.get("paper_type", ""),
            "paper_json": json.dumps(meta, ensure_ascii=False),
        }
        store.add_chunks(
            ids=[f"{pid}-{c.index:04d}" for c in chunks],
            texts=[c.text for c in chunks],
            search_texts=[f"{header}\n\n{c.text}" for c in chunks],
            metadatas=[dict(base_meta, page_start=c.page_start, page_end=c.page_end) for c in chunks],
        )
        added += 1

    # Files that were deleted or changed on disk leave stale entries behind.
    removed = 0
    for stale in indexed - current_ids:
        store.delete_paper(stale)
        removed += 1

    print(
        f"\nDone. Added {added}, unchanged {skipped}, removed {removed}, failed {failed}. "
        f"Index now holds {len(store.paper_ids())} papers / {store.count()} chunks."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="delete the index and re-embed all PDFs")
    parser.add_argument("--no-llm", action="store_true", help="skip Claude metadata tagging")
    args = parser.parse_args()
    ingest(rebuild=args.rebuild, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
