"""Find and remove off-topic papers that find_papers.py downloaded.

Step 1 - review (changes nothing):
    python scripts/clean_corpus.py
  Screens every paper in corpus/_found_papers.csv and writes
  corpus/_cleanup_review.csv with a keep / remove / unsure decision and a reason.
  Open it in Excel and change any decision you disagree with, then close Excel.

Step 2 - apply:
    python scripts/clean_corpus.py --apply
  Moves every "remove" PDF to corpus/_removed/ (ingest ignores that folder).
  Delete corpus/_removed/ yourself once you're happy, or use --permanent to
  delete straight away. Then run `python -m pa_chatbot.ingest`, which also drops
  removed papers from an existing index.

Options:
    --no-llm         screen with keyword rules instead of Claude (free, cruder)
    --rescreen       ignore an existing review file and screen again
    --remove-unsure  treat "unsure" as "remove" when applying

Only files listed in the find_papers log are touched; PDFs you added yourself are never moved.
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pa_chatbot.config import settings  # noqa: E402
from pa_chatbot.screening import KEEP, REMOVE, UNSURE, screen  # noqa: E402

LOG = settings.corpus_dir / "_found_papers.csv"
REVIEW = settings.corpus_dir / "_cleanup_review.csv"
REMOVED_DIR = settings.corpus_dir / "_removed"
REVIEW_FIELDS = ["decision", "reason", "method", "on_disk", "topic", "file", "title", "venue", "year", "status", "openalex_id"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    try:
        # utf-8-sig so Excel shows accented author names correctly
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        sys.exit(f"\nCan't write {path.name} - it's probably open in Excel (or being synced by OneDrive). Close it and re-run.")


def pdf_path(row: dict) -> Path | None:
    if not row.get("file"):
        return None
    return settings.corpus_dir / row["topic"] / row["file"]


def build_review(use_llm: bool) -> list[dict]:
    if not LOG.exists():
        sys.exit(f"No log found at {LOG}. Run scripts/find_papers.py first.")
    rows, seen = [], set()
    for row in read_csv(LOG):
        key = row.get("openalex_id") or row.get("title")
        if key in seen or row.get("status") in ("removed", "off-topic"):
            continue
        seen.add(key)
        rows.append(row)

    how = "Claude" if use_llm else "keyword rules"
    print(f"Screening {len(rows)} papers with {how}...")
    verdicts = screen(rows, use_llm=use_llm)

    review = []
    for row, v in zip(rows, verdicts):
        path = pdf_path(row)
        review.append({
            **row,
            "decision": v.decision,
            "reason": v.reason,
            "method": v.method,
            "on_disk": "yes" if path and path.exists() else "no",
        })
    # Removals first so they're easy to eyeball in Excel.
    order = {REMOVE: 0, UNSURE: 1, KEEP: 2}
    review.sort(key=lambda r: (order.get(r["decision"], 3), r["topic"], r["title"]))
    return review


def print_summary(review: list[dict]) -> None:
    counts = Counter(r["decision"] for r in review)
    on_disk = Counter(r["decision"] for r in review if r["on_disk"] == "yes")
    print(f"\n{'decision':8}  {'papers':>6}  {'PDFs on disk':>12}")
    for d in (KEEP, UNSURE, REMOVE):
        print(f"{d:8}  {counts.get(d, 0):6}  {on_disk.get(d, 0):12}")

    for label, decision in (("UNSURE - check these", UNSURE), ("KEEP", KEEP)):
        items = [r for r in review if r["decision"] == decision and r["on_disk"] == "yes"]
        if items:
            print(f"\n{label}:")
            for r in items:
                print(f"  - {r['title'][:90]}  [{r['reason'][:60]}]")


def apply(review: list[dict], permanent: bool, remove_unsure: bool) -> None:
    targets = {REMOVE, UNSURE} if remove_unsure else {REMOVE}
    moved = missing = 0
    removed_ids = set()
    for r in review:
        if r["decision"].strip().lower() not in targets:
            continue
        path = pdf_path(r)
        removed_ids.add(r.get("openalex_id") or r.get("title"))
        if path is None or not path.exists():
            missing += 1
            continue
        if permanent:
            path.unlink()
        else:
            dest = REMOVED_DIR / r["topic"] / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
        moved += 1

    # Mark them in the find_papers log so a re-run of this script skips them.
    log_rows = read_csv(LOG)
    for row in log_rows:
        if (row.get("openalex_id") or row.get("title")) in removed_ids:
            row["status"] = "removed"
    write_csv(LOG, log_rows, list(log_rows[0].keys()) if log_rows else REVIEW_FIELDS)

    # Tidy empty topic folders.
    for folder in settings.corpus_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith("_") and not any(folder.iterdir()):
            folder.rmdir()

    verb = "Deleted" if permanent else f"Moved to {REMOVED_DIR}"
    print(f"{verb}: {moved} PDFs ({missing} marked for removal had no PDF on disk).")
    if not permanent and moved:
        print("Check that folder, then delete it when you're happy.")
    print("Next: python -m pa_chatbot.ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="act on the review file")
    parser.add_argument("--permanent", action="store_true", help="with --apply: delete instead of moving to _removed/")
    parser.add_argument("--remove-unsure", action="store_true", help="with --apply: also remove 'unsure' papers")
    parser.add_argument("--rescreen", action="store_true", help="screen again even if a review file exists")
    parser.add_argument("--no-llm", action="store_true", help="use keyword rules instead of Claude")
    args = parser.parse_args()

    if args.apply:
        if not REVIEW.exists():
            sys.exit("No review file yet. Run without --apply first and check corpus/_cleanup_review.csv.")
        review = read_csv(REVIEW)
        print(f"Using decisions from {REVIEW.name} (including any edits you made).")
        apply(review, permanent=args.permanent, remove_unsure=args.remove_unsure)
        return

    if REVIEW.exists() and not args.rescreen:
        review = read_csv(REVIEW)
        print(f"Loaded existing {REVIEW.name} (use --rescreen to screen again).")
    else:
        review = build_review(use_llm=not args.no_llm)
        write_csv(REVIEW, review, REVIEW_FIELDS)
        print(f"Wrote {REVIEW}")
    print_summary(review)
    print(
        f"\nNothing has been deleted. Review {REVIEW.name} (edit the 'decision' column if needed, then close Excel),\n"
        "then run:  python scripts/clean_corpus.py --apply"
    )


if __name__ == "__main__":
    main()
