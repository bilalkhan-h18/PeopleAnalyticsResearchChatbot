"""Decide whether a paper belongs in a people analytics library, from its title and venue.

Claude does the judging (a batch of 40 titles costs well under a cent on
Sonnet-class models). Without an API key or with use_llm=False, a keyword
rule set is used instead: cruder, but free and good at catching the obvious
medical and natural-science papers.
"""

import re
from dataclasses import dataclass

from .config import settings
from .llm import structured_call

KEEP, REMOVE, UNSURE = "keep", "remove", "unsure"

_SYSTEM = """\
You screen papers for a people analytics research library. A people analytics \
practitioner uses the library to find literature, theories, frameworks and methods \
for questions about a company's workforce.

KEEP papers on: human resource management, organisational behaviour, industrial-\
organisational and work psychology, employee attitudes and behaviour (engagement, \
commitment, turnover, performance, wellbeing and burnout at work, motivation), \
leadership and management, teams, careers, recruitment and selection, pay and \
rewards, diversity and discrimination in employment, labour and personnel economics, \
future of work and remote work, workplace technology and algorithmic management, \
organisational networks, HR/people/workforce analytics. Also KEEP research-methods \
papers widely used in social and organisational research (measurement and validity, \
scale development, SEM, mediation, multilevel models, survival analysis, missing \
data, statistical power, common method bias, literature review methods).

REMOVE papers from unrelated fields: clinical medicine, epidemiology and public \
health (unless about employees at work), biology and genetics, chemistry, physics, \
materials, climate and earth science, pure computer science and machine learning, \
finance and accounting (unless about employees or executives as people), \
education (unless about teachers as employees), tourism, and anything else with no \
clear link to people at work.

Use UNSURE only when a title could genuinely go either way; explain why in a few words."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "decision": {"type": "string", "enum": [KEEP, REMOVE, UNSURE]},
                    "reason": {"type": "string"},
                },
                "required": ["id", "decision", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}

# ---- keyword fallback -------------------------------------------------------

_OFF_TOPIC = re.compile(
    r"\b(cancer|tumou?r|oncolog|clinical|patients?|disease|diagnos|therap|drug|"
    r"sepsis|cardi|heart|hypertens|diabet|alzheimer|dementia|surg|thyroid|nutrition|"
    r"mortality|epidemiolog|pandemic|covid|vaccin|mrna|protein|genom|gene\b|genetic|sequenc|"
    r"microbio|immun|neuro|brain|crystallograph|molecul|chemi|catalys|quantum|physic|"
    r"climate|planetary|ecolog|environmental|carbon|dengue|bacteria|imagej|"
    r"image analysis|deep convolutional|object classes|document recognition|"
    r"mutual fund|security market|earnings management|audit|esg ratings|investor|"
    r"prisma|strobe|cochrane|risk of bias|randomi[sz]ed trials?|tourism|breastfeeding)",
    re.IGNORECASE,
)
_ON_TOPIC = re.compile(
    r"\b(employee|employment|worker|workforce|workplace|personnel|human resource|hrm?\b|"
    r"job|work engagement|burnout|turnover|retention|commitment|organi[sz]ational|"
    r"leader|manager|team|career|recruit|selection|hiring|talent|compensation|pay\b|wage|"
    r"diversity|discriminat|gender gap|remote work|telework|hybrid work|"
    r"people analytics|hr analytics|workforce analytics|performance appraisal|"
    r"common method|structural equation|multilevel|mediation|indirect effects|"
    r"construct validity|discriminant validity|survival analysis|power analys|"
    r"literature review|meta-analy|voice|silence)",
    re.IGNORECASE,
)

_MEDICAL_VENUES = re.compile(
    r"(lancet|bmj|jama|new england journal|plos medicine|circulation|heart journal|"
    r"cancer|clinic|nature (biotech|methods|reviews)|nucleic acids|bioinformatics|"
    r"neurolog|neuroimage|alzheimer|dementia|diabetes|thyroid|intensive care|"
    r"physical review|geoscientific|climatic|crystallograph|molecular biology)",
    re.IGNORECASE,
)


@dataclass
class Verdict:
    decision: str
    reason: str
    method: str  # "claude" | "keywords"


def keyword_verdict(title: str, venue: str) -> Verdict:
    text = f"{title} {venue}"
    on = _ON_TOPIC.search(text)
    off = _OFF_TOPIC.search(text) or _MEDICAL_VENUES.search(venue or "")
    if on and not off:
        return Verdict(KEEP, f"matches '{on.group(0)}'", "keywords")
    if off and not on:
        return Verdict(REMOVE, f"off-topic term '{off.group(0)}'", "keywords")
    if on and off:
        return Verdict(UNSURE, f"mixed: '{on.group(0)}' vs '{off.group(0)}'", "keywords")
    return Verdict(UNSURE, "no clear signal in title/venue", "keywords")


def screen(papers: list[dict], use_llm: bool = True, batch_size: int = 40) -> list[Verdict]:
    """papers: dicts with 'title', 'venue' and optionally 'year'. Returns one Verdict per paper."""
    if not use_llm:
        return [keyword_verdict(p.get("title", ""), p.get("venue", "")) for p in papers]

    verdicts: list[Verdict | None] = [None] * len(papers)
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start + batch_size]
        listing = "\n".join(
            f"{i}. {p.get('title', '').strip()} | {p.get('venue', '').strip() or 'unknown venue'} | {p.get('year', '')}"
            for i, p in enumerate(batch)
        )
        result = structured_call(
            model=settings.utility_model,
            effort=settings.utility_effort,
            system=_SYSTEM,
            prompt=f"Screen each paper (id. title | venue | year):\n\n{listing}",
            schema=_SCHEMA,
            max_tokens=16000,
        )
        for d in (result or {}).get("decisions", []):
            i = d.get("id")
            if isinstance(i, int) and 0 <= i < len(batch):
                verdicts[start + i] = Verdict(d["decision"], d.get("reason", ""), "claude")

    # Anything Claude skipped (or a failed batch) falls back to keywords.
    return [
        v or keyword_verdict(p.get("title", ""), p.get("venue", ""))
        for v, p in zip(verdicts, papers)
    ]
