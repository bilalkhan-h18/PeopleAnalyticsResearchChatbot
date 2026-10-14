"""People Analytics Advisor - Gradio 5 UI.

    python app.py            # http://127.0.0.1:7860
"""

import os

import anthropic
import gradio as gr

from pa_chatbot.assistant import AnswerResult, ResearchAssistant
from pa_chatbot.config import settings
from pa_chatbot.prompts import AUDIENCES, DEFAULT_AUDIENCE, DEFAULT_MODE, MODES

assistant = ResearchAssistant()

EXAMPLES = [
    "What evidence links staff turnover and engagement to store sales and financial performance?",
    "Help me build a case for linking our HR data with sales data at store level.",
    "How do I convince a sceptical CFO that people analytics is worth investing in?",
    "Early leaver rates are high in our first 90 days. What should I look at first?",
    "What people metrics should a retail business put on an executive dashboard?",
    "How could I show the cost of turnover to the business in a way leaders will act on?",
]

PRIVACY_NOTE = (
    "Questions and retrieved excerpts are sent to the Claude API using your personal key. "
    "**Don't paste employee-level data, names, or confidential company information** - describe the problem instead."
)


def library_summary() -> str:
    papers = assistant.store.papers()
    if not papers:
        return "**Library is empty.** Add PDFs to `corpus/` and run `python -m pa_chatbot.ingest`."
    years = sorted(int(p["year"]) for p in papers if str(p.get("year", "")).isdigit())
    span = f", {years[0]}-{years[-1]}" if years else ""
    return f"**{len(papers)} papers** / {assistant.store.count()} passages{span}"


def respond(message, chat, api_history, mode, audience, allow_gk, year_min, n_sources):
    message = (message or "").strip()
    if not message:
        yield chat, api_history, gr.update(), ""
        return
    chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    yield chat, api_history, gr.update(), ""

    try:
        for item in assistant.ask(
            message,
            api_history,
            mode=mode,
            allow_general_knowledge=allow_gk,
            year_min=int(year_min) if year_min else None,
            n_sources=int(n_sources),
            audience=audience,
        ):
            if isinstance(item, AnswerResult):
                chat[-1]["content"] = item.markdown
                api_history = api_history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": item.plain_text},
                ]
                yield chat, api_history, item.sources_md, ""
            elif item.kind == "status":
                chat[-1]["content"] = f"*{item.text}*"
                yield chat, api_history, gr.update(), ""
            else:
                chat[-1]["content"] = item.text
                yield chat, api_history, gr.update(), ""
    except anthropic.AuthenticationError:
        chat[-1]["content"] = "**API key rejected.** Check `ANTHROPIC_API_KEY` in your `.env` file."
        yield chat, api_history, gr.update(), ""
    except anthropic.RateLimitError:
        chat[-1]["content"] = "**Rate limited by the API.** Wait a minute and try again."
        yield chat, api_history, gr.update(), ""
    except anthropic.APIStatusError as exc:
        chat[-1]["content"] = f"**API error {exc.status_code}:** {exc.message}"
        yield chat, api_history, gr.update(), ""
    except anthropic.APIConnectionError:
        chat[-1]["content"] = "**Couldn't reach the Claude API.** Check your internet connection."
        yield chat, api_history, gr.update(), ""


def clear():
    return [], [], "", ""


with gr.Blocks(title="People Analytics Advisor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## People Analytics Advisor\nBusiness-focused answers, business cases and analysis plans, backed by your research library.")
    api_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                height=620,
                show_copy_button=True,
                render_markdown=True,
                sanitize_html=True,
                placeholder="Ask about a workforce problem, a business case, a stakeholder, or an analysis you're planning.",
            )
            box = gr.Textbox(
                placeholder="e.g. How do I show the Head of Retail that staff turnover is costing us sales?",
                show_label=False,
                lines=2,
                submit_btn=True,
            )
            gr.Examples(EXAMPLES, inputs=box)
            gr.Markdown(PRIVACY_NOTE)
        with gr.Column(scale=1):
            mode = gr.Dropdown(list(MODES), value=DEFAULT_MODE, label="Answer mode")
            audience = gr.Dropdown(list(AUDIENCES), value=DEFAULT_AUDIENCE, label="Audience",
                                   info="Who you're preparing this for.")
            allow_gk = gr.Checkbox(
                value=True,
                label="Allow general knowledge beyond my library",
                info="Marked in answers as '(general knowledge)'.",
            )
            year_min = gr.Number(value=0, label="Only papers published from (0 = any year)", precision=0)
            n_sources = gr.Slider(4, 24, value=settings.default_sources, step=1, label="Excerpts to retrieve")
            clear_btn = gr.Button("New conversation")
            library = gr.Markdown(library_summary())
            with gr.Accordion("Sources for the last answer", open=True):
                sources = gr.Markdown()

    inputs = [box, chatbot, api_history, mode, audience, allow_gk, year_min, n_sources]
    outputs = [chatbot, api_history, sources, box]
    box.submit(respond, inputs, outputs)
    clear_btn.click(clear, None, [chatbot, api_history, sources, box])
    demo.load(library_summary, None, library)


if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
