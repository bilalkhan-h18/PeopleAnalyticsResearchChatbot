"""Find open-access people analytics papers on OpenAlex and download their PDFs.

    python scripts/find_papers.py                         # default topic list
    python scripts/find_papers.py --query "psychological safety teams" --topic teams
    python scripts/find_papers.py --list-only             # write the shortlist, don't download

Only open-access PDFs are downloaded. Every run appends to corpus/_found_papers.csv
so you can review what was found (including paywalled papers worth getting through
your library or company subscription).

OpenAlex is free and needs no key. Set OPENALEX_EMAIL in .env to use its faster
"polite pool".
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pa_chatbot.config import settings  # noqa: E402

API = "https://api.openalex.org/works"

DEFAULT_SEARCHES = {
    "hr-analytics": ["HR analytics", "people analytics", "workforce analytics", "evidence-based human resource management"],
    "attrition": ["voluntary turnover predictors", "turnover intention meta-analysis", "employee retention job embeddedness"],
    "engagement": ["work engagement job demands resources", "employee engagement performance meta-analysis"],
    "performance": ["individual performance measurement", "high performance work systems firm performance"],
    "recruitment": ["personnel selection validity", "recruitment source quality of hire"],
    "dei": ["gender pay gap organisations", "diversity climate outcomes", "algorithmic bias hiring"],
    "wellbeing": ["employee burnout antecedents", "employee wellbeing productivity"],
    "leadership": ["leader-member exchange outcomes", "span of control managers"],
    "networks": ["organizational network analysis", "social networks in organizations performance"],
    "hybrid-work": ["remote work productivity", "hybrid work employee outcomes"],
    "methods": ["survival analysis employee turnover", "multilevel modeling organizational research", "common method bias"],
}


def search(query: str, per_page: int, from_year: int | None) -> list[dict]:
    filters = ["type:article|review|book-chapter"]
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    params = {
        "search": query,
        "filter": ",".join(filters),
        "sort": "cited_by_count:desc",
        "per-page": per_page,
    }
    if os.getenv("OPENALEX_EMAIL"):
        params["mailto"] = os.getenv("OPENALEX_EMAIL")
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def describe(work: dict) -> dict:
    oa = work.get("best_oa_location") or {}
    source = ((work.get("primary_location") or {}).get("source") or {})
    authors = [a["author"]["display_name"] for a in work.get("authorships", []) if a.get("author")]
    return {
        "title": work.get("title") or "",
        "year": work.get("publication_year") or "",
        "authors": "; ".join(authors[:6]),
        "venue": source.get("display_name") or "",
        "doi": work.get("doi") or "",
        "cited_by": work.get("cited_by_count", 0),
        "pdf_url": oa.get("pdf_url") or "",
        "openalex_id": work.get("id", ""),
    }


def safe_name(info: dict) -> str:
    first_author = (info["authors"].split(";")[0].split() or ["unknown"])[-1]
    title = re.sub(r"[^A-Za-z0-9]+", "_", info["title"])[:60].strip("_")
    return f"{first_author}_{info['year']}_{title}.pdf"


def download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "pa-research-assistant/1.0"})
    except requests.RequestException:
        return False
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        return False  # many "pdf_url"s are landing pages; skip those
    dest.write_bytes(r.content)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", help="a single search query (default: built-in topic list)")
    parser.add_argument("--topic", default="misc", help="corpus sub-folder for --query results")
    parser.add_argument("--per-query", type=int, default=15)
    parser.add_argument("--from-year", type=int, default=None)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    searches = {args.topic: [args.query]} if args.query else DEFAULT_SEARCHES
    log_path = settings.corpus_dir / "_found_papers.csv"
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    seen: set[str] = set()

    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["topic", "status", "file", "title", "year", "authors", "venue", "doi", "cited_by", "pdf_url", "openalex_id"])
        if new_log:
            writer.writeheader()
        for topic, queries in searches.items():
            folder = settings.corpus_dir / topic
            for q in queries:
                print(f"\n[{topic}] {q}")
                try:
                    works = search(q, args.per_query, args.from_year)
                except requests.RequestException as exc:
                    print(f"  ! search failed: {exc}")
                    continue
                for work in works:
                    info = describe(work)
                    if not info["title"] or info["openalex_id"] in seen:
                        continue
                    seen.add(info["openalex_id"])
                    status, fname = "paywalled", ""
                    if info["pdf_url"]:
                        fname = safe_name(info)
                        dest = folder / fname
                        if dest.exists():
                            status = "already-have"
                        elif args.list_only:
                            status = "open-access"
                        else:
                            folder.mkdir(parents=True, exist_ok=True)
                            status = "downloaded" if download(info["pdf_url"], dest) else "download-failed"
                            time.sleep(0.5)
                    print(f"  {status:15} {info['year']} {info['title'][:80]}")
                    writer.writerow({"topic": topic, "status": status, "file": fname, **info})

    print(f"\nLog written to {log_path}. Next: python -m pa_chatbot.ingest")


if __name__ == "__main__":
    main()
