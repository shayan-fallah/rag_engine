from knowledge.chunking import split_text


def test_short_text_is_one_chunk():
    assert split_text("  A small product.\n\n\n\nWith two lines.  ", size=100) == [
        "A small product.\n\nWith two lines."
    ]


def test_empty_text_has_no_chunks():
    assert split_text(" \n ") == []


def test_long_text_is_split_at_lines_with_overlap():
    lines = [f"Line {i}: " + "word " * 15 for i in range(20)]
    chunks = split_text("\n".join(lines), size=300, overlap=100)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    # every line is in some chunk, and never cut in the middle
    for line in lines:
        assert any(line.strip() in c for c in chunks)
    # consecutive chunks share the overlap
    assert chunks[0].split("\n")[-1] == chunks[1].split("\n")[0]


def test_long_line_is_split_at_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = split_text(text, size=200, overlap=0)
    assert all(len(c) <= 200 for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_persian_sentences():
    text = "این گوشی دوربین خوبی دارد. " * 30 + "آیا ضد آب است؟ " * 30
    chunks = split_text(text, size=250, overlap=50)
    assert all(len(c) <= 250 for c in chunks)
    assert chunks[-1].endswith("؟")


def test_word_longer_than_chunk_is_cut():
    chunks = split_text("x" * 1000, size=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks) and "".join(chunks) == "x" * 1000
