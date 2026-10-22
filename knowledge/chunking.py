"""Split text into chunks of at most `size` characters.

Most product descriptions fit in one chunk. Longer text is split at paragraph
boundaries first, then sentences, and only cut mid-sentence as a last resort.
Each chunk starts with the end of the previous one (`overlap` characters), so a
fact on a boundary is complete in at least one chunk.
"""

import re

SENTENCE_END = re.compile(r"(?<=[.!?;。؟؛])\s+")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int = 1500, overlap: int = 150) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    size = max(size, 100)
    if len(text) <= size:
        return [text]
    overlap = min(overlap, size // 2)

    pieces = []
    for line in text.split("\n"):
        pieces.extend(_split_piece(line.strip(), size))

    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if current and len("\n".join(current + [piece])) > size:
            chunks.append("\n".join(current))
            current = _tail(current, overlap, room=size - len(piece) - 1)
        current.append(piece)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_piece(piece: str, size: int) -> list[str]:
    """A line longer than `size` -> sentences -> hard cuts at whitespace."""
    if len(piece) <= size:
        return [piece]
    out = []
    for sentence in SENTENCE_END.split(piece):
        while len(sentence) > size:
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            out.append(sentence)
    return out


def _tail(pieces: list[str], overlap: int, room: int) -> list[str]:
    """The last pieces of a chunk, up to `overlap` characters, to start the next chunk with."""
    tail: list[str] = []
    length = 0
    for piece in reversed(pieces):
        length += len(piece) + 1
        if length > min(overlap, room):
            break
        tail.insert(0, piece)
    return tail
