"""People Analytics Research Assistant - Gradio 5 UI.

    python app.py            # http://127.0.0.1:7860
"""

import os

import anthropic
import gradio as gr

from pa_chatbot.assistant import AnswerResult, ResearchAssistant
from pa_chatbot.config import settings
from pa_chatbot.prompts import DEFAULT_MODE, MODES

assistant = ResearchAssistant()

EXAMPLES = [
    "Our engineering attrition jumped after the return-to-office mandate. How should I frame and test whether the mandate caused it?",
    "I think managers' span of control drives engagement scores down. What theory supports or challenges this, and what hypotheses could I test?",
    "What's the best way to analyse time-to-exit for new hires when many are still employed?",
    "Does the evidence support pay transparency reducing gender pay gaps?",
    "Which frameworks describe how organisations mature in their use of HR analytics?",
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


def respond(message, chat, api_history, mode, allow_gk, year_min, n_sources):
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


with gr.Blocks(title="People Analytics Research Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## People Analytics Research Assistant\nLiterature, frameworks and approaches for your workforce questions, grounded in your own library.")
    api_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                height=620,
                show_copy_button=True,
                render_markdown=True,
                sanitize_html=True,
                placeholder="Describe a workforce problem, a working thesis, or a research question.",
            )
            box = gr.Textbox(
                placeholder="e.g. Why might high performers be leaving faster than others, and how could I test it?",
                show_label=False,
                lines=2,
                submit_btn=True,
            )
            gr.Examples(EXAMPLES, inputs=box)
            gr.Markdown(PRIVACY_NOTE)
        with gr.Column(scale=1):
            mode = gr.Dropdown(list(MODES), value=DEFAULT_MODE, label="Answer mode")
            allow_gk = gr.Checkbox(
                value=True,
                label="Allow general knowledge beyond my library",
                info="Marked in answers as '(general knowledge - not from your library)'.",
            )
            year_min = gr.Number(value=0, label="Only papers published from (0 = any year)", precision=0)
            n_sources = gr.Slider(4, 24, value=settings.default_sources, step=1, label="Excerpts to retrieve")
            clear_btn = gr.Button("New conversation")
            library = gr.Markdown(library_summary())
            with gr.Accordion("Sources for the last answer", open=True):
                sources = gr.Markdown()

    inputs = [box, chatbot, api_history, mode, allow_gk, year_min, n_sources]
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
