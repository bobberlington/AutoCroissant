"""split_long_message: Discord-sized chunks that keep code fences balanced."""
import pytest

from commands.utils import BREAK_LEN, split_long_message

FENCE = "```"


def test_short_message_unchanged():
    assert split_long_message("hello") == ["hello"]
    assert split_long_message("") == []


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


def test_several_blocks_like_update_stats():
    """update_stats sends "path\\n```\\nproblems\\n```" bundles joined together."""
    bundle = "".join(f"Creatures/Card_{i}.psd\n{FENCE}\nHP NOT FOUND\nSPD NOT FOUND\n{FENCE}" for i in range(120))
    chunks = split_long_message(bundle)
    assert len(chunks) > 1 and all(c.count(FENCE) % 2 == 0 for c in chunks)


@pytest.mark.parametrize("text,size", [("word " * 30, 20), ("z" * 50, 20)])
def test_no_newline_falls_back_to_spaces_then_hard_cut(text, size):
    chunks = split_long_message(text, size)
    assert all(len(c) <= size for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


@pytest.mark.parametrize("text", ["\n" + "z" * 3000, " " + "z" * 3000, "\n" + "z" * 1000 + " " + "z" * 2000])
def test_leading_break_gives_no_empty_chunk(text):
    chunks = split_long_message(text)
    assert all(chunks) and all(len(c) <= BREAK_LEN for c in chunks)
