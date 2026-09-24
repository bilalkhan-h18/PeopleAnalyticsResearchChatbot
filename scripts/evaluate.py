"""Check retrieval quality on a fixed question set, optionally with full answers.

    python scripts/evaluate.py              # retrieval only - free, no API calls
    python scripts/evaluate.py --plan       # also use Claude to plan queries (small cost)
    python scripts/evaluate.py --answer     # full answers written to eval/results.md (costs ~ one chat turn per question)

Re-run after changing chunking, the embedding model or prompts, and compare.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pa_chatbot.assistant import AnswerResult, ResearchAssistant, plan_queries  # noqa: E402


def load_questions(path: Path) -> list[tuple[str, list[str]]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        question, _, expected = line.partition("=>")
        out.append((question.strip(), [e.strip().lower() for e in expected.split(";") if e.strip()]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.txt")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--answer", action="store_true")
    parser.add_argument("-k", type=int, default=12)
    args = parser.parse_args()

    assistant = ResearchAssistant()
    report = ["# Evaluation run\n"]
    found_total = expected_total = 0
    for question, expected in load_questions(args.questions):
        queries = plan_queries(question, []) if (args.plan or args.answer) else [question]
        hits = assistant.store.search(queries, k=args.k)
        titles = list(dict.fromkeys(h.metadata["title"] for h in hits))
        print(f"\nQ: {question}")
        for t in titles:
            print(f"   - {t[:100]}")
        if expected:
            found = [e for e in expected if any(e in t.lower() for t in titles)]
            found_total += len(found)
            expected_total += len(expected)
            print(f"   expected papers found: {len(found)}/{len(expected)}")

        if args.answer:
            result = None
            for item in assistant.ask(question, [], n_sources=args.k):
                if isinstance(item, AnswerResult):
                    result = item
            report.append(f"## {question}\n\n{result.markdown}\n\n{result.sources_md}\n")

    if expected_total:
        print(f"\nRecall of expected papers: {found_total}/{expected_total}")
    if args.answer:
        out = ROOT / "eval" / "results.md"
        out.write_text("\n".join(report))
        print(f"\nAnswers written to {out}")


if __name__ == "__main__":
    main()
