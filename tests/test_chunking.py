from pa_chatbot.chunking import Page, chunk_pages, clean_text, strip_reference_list


def test_clean_text_joins_hyphenated_words():
    assert clean_text("turn-\nover  intention") == "turnover intention"


def test_chunks_respect_size_and_track_pages():
    pages = [Page(1, "\n\n".join(["Alpha sentence here."] * 40)), Page(2, "\n\n".join(["Beta sentence here."] * 40))]
    chunks = chunk_pages(pages, chunk_chars=300, overlap=50)
    assert len(chunks) > 4
    assert all(len(c.text) <= 300 + 50 for c in chunks)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)


def test_chunks_overlap():
    pages = [Page(1, "\n\n".join(f"Paragraph {i}." for i in range(50)))]
    chunks = chunk_pages(pages, chunk_chars=100, overlap=30)
    last_of_first = chunks[0].text.split("\n\n")[-1]
    assert last_of_first in chunks[1].text.split("\n\n")
    assert chunks[1].text.split("\n\n")[0] in chunks[0].text


def test_long_paragraph_is_split():
    text = " ".join(["This is a sentence."] * 200)
    chunks = chunk_pages([Page(1, text)], chunk_chars=500, overlap=0)
    assert len(chunks) > 5
    assert all(len(c.text) <= 500 for c in chunks)


def test_strip_reference_list_drops_bibliography():
    pages = [Page(i, f"Body text {i}") for i in range(1, 9)]
    pages.append(Page(9, "Conclusion text.\n\nReferences\n\nSmith, J. (2019). Title."))
    pages.append(Page(10, "Jones, K. (2020). Another."))
    kept = strip_reference_list(pages)
    assert [p.number for p in kept] == list(range(1, 10))
    assert kept[-1].text == "Conclusion text."


def test_strip_reference_list_ignores_early_heading():
    pages = [Page(1, "Contents\nReferences\n")] + [Page(i, "Body") for i in range(2, 11)]
    assert len(strip_reference_list(pages)) == 10
