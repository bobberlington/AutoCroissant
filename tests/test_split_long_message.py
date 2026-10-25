"""split_long_message: Discord-sized chunks that keep code fences balanced."""
import pytest

from commands.utils import BREAK_LEN, split_long_message

FENCE = "```"


def test_splits_on_newlines():
    lines = [f"line {i:04d} " + "x" * 40 for i in range(200)]
    chunks = split_long_message("\n".join(lines))
    assert len(chunks) > 1 and all(len(c) <= BREAK_LEN for c in chunks)
    assert [l for c in chunks for l in c.split("\n")] == lines


def test_code_block_reopened_and_closed():
    body = "\n".join(f"{i:05d} " + "y" * 30 for i in range(150))
    chunks = split_long_message(f"Problems:\n{FENCE}\n{body}\n{FENCE}\nafter")
    assert len(chunks) >= 3
    assert all(c.count(FENCE) % 2 == 0 for c in chunks)
    assert all(len(c) <= 2000 for c in chunks)
    assert chunks[0].startswith("Problems:") and chunks[0].endswith(f"\n{FENCE}")
    for c in chunks[1:len(chunks) - 1]:
        assert c.startswith(f"{FENCE}\n") and c.endswith(f"\n{FENCE}")
    assert chunks[len(chunks) - 1].endswith("after")
    inner = [l for c in chunks for l in c.split("\n") if l != FENCE]
    assert inner == ["Problems:"] + body.split("\n") + ["after"]


@pytest.mark.parametrize("text,size", [("word " * 30, 20)])
def test_no_newline_falls_back_to_spaces_then_hard_cut(text, size):
    chunks = split_long_message(text, size)
    assert all(len(c) <= size for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

