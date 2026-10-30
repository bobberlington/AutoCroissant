"""
The rulebook's chapters (the card repo's site/rulebook/NN-slug.md, written on the website; their syntax is its
site/assets/rulebook.js) as plain text: db/rulebook.json, which the card data build writes and /query_rulebook and
/export_rulebook read. No network, no Discord. Uncompiled, like card_data.
"""
from json import dumps, loads
from re import IGNORECASE, compile as regex, sub
from typing import Optional

from global_config import RULEBOOK_SRC

# ========================
# Configuration
# ========================
SCHEMA = 1
CHAPTER_FILE = regex(r"^(\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)\.md$")    # build_site.RULE_FILE
SKIP = regex(r"^(?::{3,}.*|=+\s*page\s*=+|-{3,}|\*{3,}|\{[^}]*\}|\|[\s|:-]*-[\s|:-]*\|)$", IGNORECASE)   # fences, breaks, rules, a table's
INLINE = (                                          # markup -> words, in order (build_site.rulebook_text, rulebook.js plainText)
    (regex(r"^#{1,3}\s+"), ""), (regex(r"^>\s?"), ""), (regex(r"^\[![\w-]+(?:\s+[\w-]+)*\]\s*"), ""),
    (regex(r"!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]"), r"\1"), (regex(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]"), lambda m: m.group(2) or m.group(1)),
    (regex(r"!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?"), ""), (regex(r"\[([^\[\]]+)\]\{[^}]*\}"), r"\1"),
    (regex(r"\[([^\[\]]+)\]\((?:https?:|#|mailto:)[^)]*\)"), r"\1"), (regex(r"\*\*\*(.+?)\*\*\*"), r"\1"),
    (regex(r"\*\*(.+?)\*\*"), r"\1"), (regex(r"\*([^*\s](?:[^*]*[^*\s])?)\*"), r"\1"), (regex(r"\\([\\*\[\]{}()!_`|])"), r"\1"))
ATTRS = regex(r"\s*\{[^{}]*\}$")                    # a heading's or a picture's {.class key=value}


# ========================
# Helper Functions
# ========================
def plain_text(source: str) -> str:
    """A chapter's words without markup: pictures and fences dropped, card pictures and links as their names, list
    markers as typed, a table's cells joined by |."""
    out = []
    for line in str(source or "").replace("\r\n", "\n").split("\n"):
        text = line.strip()
        if SKIP.match(text):
            continue
        if not regex(r"\]\{[^{}]*\}$").search(text):  # [text]{.red} at the end is a colour, not attributes
            text = ATTRS.sub("", text)
        for pattern, repl in INLINE:
            text = pattern.sub(repl, text)
        if text.startswith("|"):
            text = " | ".join(cell.strip() for cell in text.strip("|").split("|"))
        out.append(text.strip())
    return sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def chapter(path: str, source: str) -> Optional[dict]:
    """A chapter file as {id, num, title, file, text}; None for a file not named NN-slug.md (README.md...)."""
    name = path.rsplit("/", 1)[-1]
    m = CHAPTER_FILE.match(name)
    if not m:
        return None
    heads = [line[2:] for line in source.replace("\r\n", "\n").split("\n") if line.startswith("# ")]
    title = ATTRS.sub("", heads[0]).strip() if heads else m.group(2).replace("-", " ").title()
    return {"id": m.group(2), "num": m.group(1), "title": title, "file": name, "text": plain_text(source)}


def label(entry: dict) -> str:
    """How a chapter is named in replies: "7. Stats" (the contents, number 0: its title alone)."""
    num = int(entry.get("num") or 0)
    return f"{num}. {entry['title']}" if num else entry["title"]


# ========================
# Public API Functions
# ========================
def chapters(files: dict) -> list:
    """The chapters of {path: text} (the build's files under RULEBOOK_SRC), in file order (the book's)."""
    found = [chapter(path, text) for path, text in sorted(files.items()) if path.startswith(RULEBOOK_SRC + "/")]
    return [c for c in found if c]


def rulebook_text(entries: list) -> str:
    """db/rulebook.json: {"schema": 1, "chapters": [...]}, one chapter per line."""
    lines = ",\n".join(dumps(c, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for c in entries)
    return '{"schema":%d,"chapters":[\n%s\n]}\n' % (SCHEMA, lines) if lines else '{"schema":%d,"chapters":[]}\n' % SCHEMA


def parse_rulebook(text: str, where: str = "rulebook.json") -> list:
    """db/rulebook.json's chapters; ValueError for a bad file."""
    data = loads(text)
    found = data.get("chapters") if isinstance(data, dict) else None
    if not isinstance(found, list) or not all(isinstance(c, dict) and isinstance(c.get("title"), str)
                                              and isinstance(c.get("text"), str) for c in found):
        raise ValueError(f"{where}: not a rulebook file")
    return found


# ========================
# Module Exports
# ========================
__all__ = [
    'chapters',
    'label',
    'parse_rulebook',
    'plain_text',
    'rulebook_text',
]
