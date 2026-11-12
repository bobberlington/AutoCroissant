"""
The standards channel's one post per push (owner 2026-11-11): the guards' lines (card_guards), the card check's notes
(card_check) and the font check's missing fonts (font_check) in one message, one line per card with its commit link,
a note two checks tell (TWINS) once; the Development role pinged only when a font is missing. Each check reports only
what the push made new on a card (the old notes stay in its commit comment and job summary). Stdlib only:
TTSCardMaker's card-guards.yml runs it in its tell job on the runner's own python3 (no venv there).

    python3 -m commands.standards_post --repo <owner/name> --after <sha> [--plan plan.json [--landed landed.json]]
                                       [--card-check report.json] [--font-check report.json] --out <payload out>
    payload(repository, after, plan, landed, card_check, font_check) -> the webhook body ({} when nothing to tell)
"""
from argparse import ArgumentParser
from json import dump, load
from os.path import basename, isfile, splitext
from typing import Optional

# ========================
# Configuration
# ========================
LINE = "{login} pushed {card}: {notes}. [{short}]({url})"       # what to do, as short as it goes (owner 2026-11-06)
SHORT = (                                   # a card check note's start -> what to do (owner 2026-11-06)
    ("NO STATS FOUND", "no stat bars"), ("NO STARS FOUND", 'no stars (group them in "Stars")'),
    ("UNKNOWN TYPE", "unknown card kind (check its folder)"), ("STATS TOO HIGH", "a stat over 20"),
    ("HP NOT FOUND", "no HP bar"), ("DEF NOT FOUND", "no DEF bar"), ("ATK NOT FOUND", "no ATK bar"),
    ("SPD NOT FOUND", "no SPD bar"), ("UNKNOWN ICON TOKEN: ", "unknown icon "),
    ("UNKNOWN ICON CHARACTER: ", "unknown icon character "), ("ICON TOKEN NOT IN STOOGEDINGS FONT: ", "not in stoogedings: "),
    ("ICON TOKEN PARTLY IN STOOGEDINGS FONT: ", "partly in stoogedings: "), ("DUPLICATE LAYER IDS", "duplicate layer IDs"),
    ("EFFECT LAYER WITHOUT TEXT LAYER: ", "FX layer without its text: "), ("EFFECT LAYER WORDS NOT IN TEXT: ", "FX words not in the text: "),
    ("EFFECT LAYER OUT OF DATE: ", "FX layer out of date: "))
# a card check note and the guards issue telling the same thing: (note start, issue code, the wording kept)
TWINS = (("NO STARS FOUND", "R6", "check"), ("DUPLICATE LAYER IDS", "R13", "guards"))
ROLE_ID = "546427268751294475"              # the card makers' role, pinged for a missing font (owner 2026-10-23)
FONT_NOTE = "the {font} font is not in MDW/Fonts ({layers}), bite this user"    # the owner's wording, per card
DISCORD_LIMIT = 2000
SUPPRESS_EMBEDS = 4                         # Discord message flag: no link previews
URL_SHA = 12                                # hex digits of a commit link
NOBODY = "Someone"


# ========================
# Helper Functions
# ========================
def card_name(path: str) -> str:
    """A card's name from its PSD path (card_models.card_name)."""
    return splitext(basename(path))[0].replace("_", " ")


def short(note: str) -> str:
    """A card check note as what to do (SHORT); a variant's "A Snack2: " before it kept, a named one's name after it."""
    split = note.find(": ")
    for prefix, rest in (("", note), *([(note[:split + 2], note[split + 2:])] if split > 0 else [])):
        for start, phrase in SHORT:
            if rest.startswith(start):
                tail = rest[len(start):].removesuffix(" (run Effect Layers > Update all)") if start.endswith(": ") else ""
                return prefix + phrase + tail
    return note


def _twin(note: str, start: str) -> bool:
    """A card check note (or a variant's "A Snack2: " one) starting so."""
    split = note.find(": ")
    return note.startswith(start) or (split > 0 and note[split + 2:].startswith(start))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ========================
# The post
# ========================
def payload(repository: str, after: str, plan: Optional[dict] = None, landed: Optional[dict] = None,
            card_check: Optional[dict] = None, font_check: Optional[dict] = None) -> dict:
    """
    The webhook body: one LINE per card with something to tell, in path order; the role mention alone on the first
    line when a font is missing; whole card lines go from the end to fit DISCORD_LIMIT; {} when nothing to tell.

    Args:
        repository: owner/name (the commit links)
        after: The push's head (a card's link when no commit of its own is known; the overflow's link)
        plan: The guards' plan.json (a card's "told" lines, its issues' codes, login and commit)
        landed: land's landed.json: its "told" replaces the plan's (what stood after the landing)
        card_check: card_check's --json report (a card's "new" notes, else all of its notes)
        font_check: font_check's --json report (a card's "new" fonts, else all of its missing ones)
    """
    url = f"https://github.com/{repository}/commit/"
    cards: dict = {}                            # psd path -> {login, commit, card, guards: [(text, code)], check, fonts}

    def card(path: str, name: str = "", login: str = "", commit: str = "") -> dict:
        one = cards.setdefault(path, {"login": "", "commit": "", "card": name or card_name(path),
                                      "guards": [], "check": [], "fonts": []})
        one["login"], one["commit"] = one["login"] or login, one["commit"] or commit
        return one

    told = (landed or {}).get("told", {})
    for entry in (plan or {}).get("cards", []):     # the card's commit and login even with nothing to tell
        path = entry.get("psd") or entry.get("card", "")
        if not path:
            continue
        codes = {issue.get("text"): issue.get("code") for issue in entry.get("issues", [])}
        card(path, entry.get("card", ""), entry.get("login", ""), entry.get("commit", ""))["guards"] += [
            (line, codes.get(line, "")) for line in told.get(path, entry.get("told", []))]
    for one in (card_check or {}).get("cards", []):
        notes = one.get("new", one.get("notes", []))
        if notes:
            card(one["path"], login=one.get("author", ""))["check"] += notes
    for one in (font_check or {}).get("cards", []):
        missing = one.get("missing", {})
        fonts = [font for font in one.get("new", list(missing)) if font in missing]
        if fonts:
            card(one["path"], login=one.get("author", ""))["fonts"] += [(font, missing[font]) for font in fonts]
    lines, ping = [], False
    for path in sorted(cards):
        one = cards[path]
        guards, check = list(one["guards"]), list(one["check"])
        for start, code, kept in TWINS:         # told by both: once
            if any(_twin(note, start) for note in check) and any(c == code for _, c in guards):
                if kept == "check":
                    guards = [(text, c) for text, c in guards if c != code]
                else:
                    check = [note for note in check if not _twin(note, start)]
        notes = [text for text, _ in guards] + [short(note) for note in check] + [
            FONT_NOTE.format(font=font, layers=", ".join(layers)) for font, layers in one["fonts"]]
        notes = [note.rstrip(".") for note in dict.fromkeys(notes) if note]
        if not notes:
            continue
        ping = ping or bool(one["fonts"])
        commit = one["commit"] or after         # none (local files): no link
        line = LINE.format(login=one["login"] or NOBODY, card=one["card"], notes="; ".join(notes), short=commit[:8],
                           url=url + commit[:URL_SHA])
        lines.append(line if commit else line[:line.rindex(" [")])
    if not lines:
        return {}
    head = [f"<@&{ROLE_ID}>"] if ping else []

    def assemble(kept: int) -> str:
        rest = len(lines) - kept
        more = [f"... and {_plural(rest, 'more card')}" + (f": [details]({url}{after})" if after else "")] if rest else []
        return "\n".join(head + lines[:kept] + more)

    kept = len(lines)
    content = assemble(kept)
    while len(content) > DISCORD_LIMIT and kept > 0:
        kept -= 1
        content = assemble(kept)
    return {"content": content, "allowed_mentions": {"roles": [ROLE_ID]} if ping else {"parse": []},
            "flags": SUPPRESS_EMBEDS}


# ========================
# Command line
# ========================
def _read(path: str) -> Optional[dict]:
    """A JSON file, or None when absent (a check that did not run or broke)."""
    if not path or not isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return load(fh)


def main(args: Optional[list] = None) -> int:
    """Writes the post to --out; prints which inputs it read."""
    cli = ArgumentParser(prog="python3 -m commands.standards_post",
                         description="The standards channel's one post per push from the guards' and the two checks' files.")
    cli.add_argument("--repo", required=True, metavar="OWNER/NAME", help="the card repo (the commit links)")
    cli.add_argument("--after", required=True, metavar="SHA", help="the push's head")
    cli.add_argument("--plan", default="", metavar="FILE", help="the guards' plan.json")
    cli.add_argument("--landed", default="", metavar="FILE", help="land's landed.json (what stood after the landing)")
    cli.add_argument("--card-check", default="", metavar="FILE", help="card_check's --json report")
    cli.add_argument("--font-check", default="", metavar="FILE", help="font_check's --json report")
    cli.add_argument("--out", required=True, metavar="FILE", help="the webhook body ({} when nothing to tell)")
    options = cli.parse_args(args)
    read = {name: _read(path) for name, path in (("plan", options.plan), ("landed", options.landed),
                                                 ("card_check", options.card_check), ("font_check", options.font_check))}
    post = payload(options.repo, options.after, **read)
    with open(options.out, "w", encoding="utf-8") as fh:
        dump(post, fh)
    print(f"read: {', '.join(name for name, data in read.items() if data is not None) or 'nothing'}; "
          f"{'a post of ' + str(len(post['content'])) + ' characters' if post else 'nothing to tell'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'FONT_NOTE',
    'LINE',
    'ROLE_ID',
    'SHORT',
    'TWINS',
    'card_name',
    'main',
    'payload',
    'short',
]
