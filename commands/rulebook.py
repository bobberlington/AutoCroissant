"""
The rulebook's chapters (the card repo's site/rulebook/NN-slug.md, written on the website; their syntax is its
site/assets/rulebook.js) as plain text and their sections: db/rulebook.json, which the card data build writes and
/query_rulebook and /export_rulebook read. No network, no Discord.
"""
from bisect import bisect_left
from json import dumps, loads
from re import IGNORECASE, compile as regex, sub
from typing import Optional
from unicodedata import normalize

from global_config import RULEBOOK_SRC

# ========================
# Configuration
# ========================
SCHEMA = 1
CHAPTER_FILE = regex(r"^(\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)\.md$")    # build_site.RULE_FILE
TITLED = regex(r'^:{3,}\s*[a-z][\w-]*\s*["“](.*)["”]\s*(?:\{[^{}]*\})?$', IGNORECASE)   # ::: panel "Title": its title stays
SKIP = regex(r"^(?::{3,}.*|=+\s*page\s*=+|-{3,}|\*{3,}|\{[^}]*\}|\|[\s|:-]*-[\s|:-]*\|)$", IGNORECASE)   # fences, breaks, rules, a table's
INLINE = (                                          # markup -> words, in order (build_site.rulebook_text, rulebook.js plainText)
    (regex(r"^#{1,3}\s+"), ""), (regex(r"^>\s?"), ""), (regex(r"^\[![\w-]+(?:\s+[\w-]+)*\]\s*"), ""),
    (regex(r"!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]"), r"\1"), (regex(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]"), lambda m: m.group(2) or m.group(1)),
    (regex(r"!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?"), ""), (regex(r"\[([^\[\]]+)\]\{[^}]*\}"), r"\1"),
    (regex(r"\[([^\[\]]+)\]\((?:https?:|#|mailto:)[^)]*\)"), r"\1"), (regex(r"\*\*\*(.+?)\*\*\*"), r"\1"),
    (regex(r"\*\*(.+?)\*\*"), r"\1"), (regex(r"\*([^*\s](?:[^*]*[^*\s])?)\*"), r"\1"), (regex(r"\\([\\*\[\]{}()!_`|])"), r"\1"))
ATTRS = regex(r"\s*\{[^{}]*\}$")                    # a heading's or a picture's {.class key=value}
WEB_ONLY = regex(r"\{[^{}]*\.web\b[^{}]*\}\s*$")     # a chapter's "# Title {.web}": on the website only (owner 2026-11-07)
# rulebook.js's blocks, for its sections: FENCE, CLOSE, HEAD; BREAK ends one; in a row, so does a column
FENCE = regex(r"^\s*:{3,}\s*([a-z][\w-]*)\s*(.*)$", IGNORECASE)
CLOSE = regex(r"^\s*:{3,}\s*$")
HEAD = regex(r"^(#{1,3})\s+(.*)$")
BREAK = regex(r"^\s{0,3}(?:-{3,}|\*{3,})\s*$|^\s*=+\s*page\s*=+\s*$", IGNORECASE)
PICTURE = regex(r"^!\[[^\]]*\]\([^)\s]+\)\s*(?:\{([^}]*)\})?\s*$")
CARDS = regex(r"^(?:\s*(?:!\[\[[^\]\n]+\]\]|->|\u2192))+\s*(\{[^}]*\})?\s*$")
ICON = regex(r"\[[^\[\]\n]{1,52}\](?![({])")          # titleText: an icon token


# ========================
# Helper Functions
# ========================
def _inline(text: str) -> str:
    """A line's markup as words (INLINE)."""
    for pattern, repl in INLINE:
        text = pattern.sub(repl, text)
    return text


def _plain_lines(lines: list) -> list:
    """plain_text's lines as (source line number, text): no blank line at either end, one between texts."""
    out = []
    for n, line in enumerate(lines):
        text = line.strip()
        titled = TITLED.match(text)
        if titled and titled.group(1).strip():
            text = titled.group(1)
        elif SKIP.match(text):
            continue
        if not regex(r"\]\{[^{}]*\}$").search(text):  # [text]{.red} at the end is a colour, not attributes
            text = ATTRS.sub("", text)
        text = _inline(text)
        if text.startswith("|"):
            text = " | ".join(cell.strip() for cell in text.strip("|").split("|"))
        text = text.strip()
        if text or (out and out[-1][1]):
            out.append((n, text))
    while out and not out[-1][1]:
        out.pop()
    return out


def plain_text(source: str) -> str:
    """A chapter's words without markup: pictures and fences dropped (a fence's "Title" kept), card pictures and links as
    their names, list markers as typed, a table's cells joined by |."""
    return "\n".join(text for _, text in _plain_lines(str(source or "").replace("\r\n", "\n").split("\n")))


def slugify(text: str) -> str:
    """rulebook.js slugify: a section's anchor (#/rules/<chapter id>/<slug>)."""
    text = sub(r"['\u2019]", "", sub(r"[\u0300-\u036f]", "", normalize("NFKD", text.lower())))
    return sub(r"^-+|-+$", "", sub(r"[^a-z0-9]+", "-", text)) or "section"


def title_text(heading: str) -> str:
    """A section's title as words (rulebook.js titleText): attributes, icon tokens and markup dropped."""
    return " ".join(_inline(ICON.sub("", ATTRS.sub("", heading.strip()))).split())


def _column(line: str) -> bool:
    """A row's column block (rulebook.js column): a callout, card pictures or a picture that doesn't float."""
    pic = PICTURE.match(line.strip())
    words = {w.lstrip(".") for w in (pic.group(1) or "").split() if "=" not in w} if pic else set()
    return line.startswith(">") or bool(CARDS.match(line)) or bool(pic and not words & {"free", "left", "right", "end"})


def _holds(lines: list, n: int) -> bool:
    """The fence opened on line n holds a '## ' section (rulebook.js holdsSection)."""
    depth = 1
    for line in lines[n + 1:]:
        if FENCE.match(line):
            depth += 1
        elif CLOSE.match(line):
            depth -= 1
            if not depth:
                return False
        elif (head := HEAD.match(line)) and len(head.group(1)) == 2:
            return True
    return False


def _sections(lines: list, at) -> list:
    """[[start, end, title, slug]] of each '## ' section as rulebook.js groups them (in fences too) and each titled
    ::: panel (a section on the website); start and end are at(source line): text lines."""
    found, frames = [], [[False, None, None]]           # per fence: [a row, its open section, its own (a panel's)]

    def close(frame: list, n: int, k: int = 1) -> None:
        if frame[k] is not None:
            found[frame[k]][1] = at(n)
            frame[k] = None

    def add(n: int, title: str) -> int:
        found.append([at(n), None, title, slugify(title)])
        return len(found) - 1

    for n, line in enumerate(lines):
        frame = frames[-1]
        fence = FENCE.match(line)
        if fence:
            if frame[0] or _holds(lines, n):
                close(frame, n)
            titled = TITLED.match(line.strip()) if fence.group(1).lower() == "panel" else None
            title = title_text(titled.group(1)) if titled else ""
            frames.append([fence.group(1).lower() == "row", None, add(n, title) if title else None])
            continue
        if CLOSE.match(line):
            if len(frames) > 1:
                close(frame, n)
                close(frames.pop(), n, 2)
            continue
        head = HEAD.match(line)
        if (head and len(head.group(1)) < 3) or BREAK.match(line) or (frame[0] and _column(line)):
            close(frame, n)
        if head and len(head.group(1)) == 2 and title_text(head.group(2)):    # an empty title: no section
            frame[1] = add(n, title_text(head.group(2)))
    for frame in frames:
        close(frame, len(lines))
        close(frame, len(lines), 2)
    return found


def chapter(path: str, source: str) -> Optional[dict]:
    """A chapter file as {id, num, title, file, text, sections}; None for a file not named NN-slug.md (README.md...) or a
    {.web} one."""
    name = path.rsplit("/", 1)[-1]
    m = CHAPTER_FILE.match(name)
    if not m:
        return None
    lines = source.replace("\r\n", "\n").split("\n")
    heads = [line[2:] for line in lines if line.startswith("# ")]
    if heads and WEB_ONLY.search(heads[0]):
        return None
    title = ATTRS.sub("", heads[0]).strip() if heads else m.group(2).replace("-", " ").title()
    plain = _plain_lines(lines)
    starts = [n for n, _ in plain]
    return {"id": m.group(2), "num": m.group(1), "title": title, "file": name, "text": "\n".join(t for _, t in plain),
            "sections": _sections(lines, lambda n: bisect_left(starts, n))}


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
    'slugify',
    'title_text',
]
