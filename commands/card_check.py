"""
Card check for the card PSDs: what the bot's parser and validator note about a card, the lines /update_stats prints (the
parser's own notes among them are also the website's Parser notes; the validator's are printed only). TTSCardMaker's
card-guards.yml runs it on every push to main that touches a PSD (card-check.yml until 2026-10-25): job summary, commit
comment, and its notes the push made new (a pushed PSD's before the push lacked; owner 2026-11-11) in the standards
channel's one post (standards_post), no ping.

    python -m commands.card_check [--psd <path>...]
                                  [--repo <owner/name> --after <sha> [--before <sha>] [--author <login>] [--whole-tree]
                                   [--clone <dir>]] [--json <out>] [--discord <payload out>]
    card_notes(parser, edits, psd, rel) -> one card's notes; check_files / check_blobs -> Report
    markdown(report), discord_payload(report) -> the job summary / commit comment and this check's part of the post

The notes equal /update_stats' for the same PSD, Types/ names and site/Card Edits.json: the parser's own (NO STARS FOUND,
NO STATS FOUND ...) and CardValidator's (HP NOT FOUND, STATS TOO HIGH ...), after
the card's website edits (problems_ok, allow_high_stats, creatures ...); a PSD with variant tags: each picture's ("A Snack2:
..."). A notification, never a
gate: findings exit 0, an unreadable PSD or a failed API call is one error line. The repo path classifies a card, so
--psd files belong to the checkout of their nearest ancestor holding Types/ (its Types/ and edits file apply); --repo
reads the pushed PSDs by blob sha (Types/, the manifest and the edits file from the tree at --after; --clone: the
guards' clone first, so each PSD is fetched once for the three checks; with
--before, each card's notes its PSD before the push lacked are its "new" ones, font_check.old_blobs). No Discord
or config imports (the token: config through github_api, else GITHUB_TOKEN); the guards' checkout
(card_guards.SPARSE_FILES) holds its import closure.
"""
from argparse import ArgumentParser
from dataclasses import dataclass, field
from json import dump, loads
from os import remove, walk
from os.path import abspath, basename, dirname, isdir, isfile, join, relpath, splitext
from tempfile import mkstemp
from typing import Optional

from psd_tools import PSDImage

from commands.card_edits import CardAuthors, CardEdits
from commands.card_models import StatsDatabase, card_name
from commands.font_check import API_TIMEOUT, NOBODY, is_card_psd, old_blobs, pushed_files, tree_blobs
from commands.github_api import blob_text, fetch_verified, token_from_env
from commands.psd_parser import CardValidator, IconVocabulary, PSDParser, picture_numbers, variants_of
from commands.standards_post import payload, short
from global_config import EDITS_PATH, ICON_MANIFEST_PATH, STARS_DIR, TYPES_DIR

# ========================
# Configuration
# ========================
UNREADABLE = "could not be read"


# ========================
# The parser's vocabulary: Types/, Types/Stars/, the icon manifest
# ========================
@dataclass
class Vocabulary:
    """What the parser needs from the card repo besides the PSD."""
    all_types: list = field(default_factory=list)   # Types/ stems, lowercased
    stars: list = field(default_factory=list)       # Types/Stars/ PNG stems, lowercased, spaces as underscores
    manifest: Optional[dict] = None                 # the stoogedings manifest, when the repo has one

    def parser(self) -> PSDParser:
        """A parser on an empty database (the stored card lends only author and series, which the notes never read)."""
        db = StatsDatabase()
        db.all_types = list(self.all_types)
        return PSDParser(db.all_types, db, IconVocabulary.build(db.all_types, self.stars, self.manifest))


def vocabulary(paths, manifest: Optional[bytes] = None) -> Vocabulary:
    """card_data's reading of a tree: Types/ entries with a dot outside Stars, Types/Stars/ PNGs, the manifest's bytes."""
    vocab = Vocabulary(manifest=loads(manifest.decode("utf-8")) if manifest else None)
    for path in paths:
        if path.startswith(TYPES_DIR) and "Stars" not in path and "." in path:
            vocab.all_types.append(splitext(basename(path))[0].lower())
        elif path.startswith(STARS_DIR + "/") and path.lower().endswith(".png"):
            vocab.stars.append(splitext(basename(path))[0].lower().replace(" ", "_"))
    return vocab


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def clone_vocabulary(root: str) -> Vocabulary:
    """The vocabulary of a checkout: every file under Types/ (dotfiles skipped) through the tree filter, the manifest."""
    paths = []
    for folder, subdirs, files in walk(join(root, TYPES_DIR)):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
        rel = relpath(folder, root).replace("\\", "/")
        paths += [f"{rel}/{name}" for name in sorted(files) if not name.startswith(".")]
    manifest = join(root, ICON_MANIFEST_PATH)
    return vocabulary(paths, _read(manifest) if isfile(manifest) else None)


def _parse_edits(path: str, text: str, errors: list) -> Optional[CardEdits]:
    try:
        return CardEdits.parse(text)
    except ValueError as e:                     # not JSON: the notes stand without the website's edits
        errors.append(f"{path}: {UNREADABLE} ({e}); the notes stand without the website's edits")
        return None


def clone_edits(root: str, errors: list) -> Optional[CardEdits]:
    """The checkout's edits file (EDITS_PATH), or None."""
    path = join(root, EDITS_PATH)
    return _parse_edits(EDITS_PATH, _read(path).decode("utf-8"), errors) if isfile(path) else None


def repo_context(repository: str, blobs: dict, errors: list) -> tuple:
    """
    (parser, edits) from a tree listing.

    Args:
        repository: owner/name
        blobs: {path: blob sha} of the tree (tree_blobs)
        errors: Gains a line when the manifest or the edits file can't be read

    Returns:
        (PSDParser, CardEdits or None)
    """
    manifest = None
    if ICON_MANIFEST_PATH in blobs:
        try:
            manifest = fetch_verified(repository, blobs[ICON_MANIFEST_PATH], timeout=API_TIMEOUT)
        except Exception as e:                  # the parser runs without the icon manifest
            errors.append(f"{ICON_MANIFEST_PATH}: {e}")
    edits = None
    path = EDITS_PATH if EDITS_PATH in blobs else None
    if path is not None:
        try:
            edits = _parse_edits(path, blob_text(repository, blobs[path], timeout=API_TIMEOUT), errors)
        except Exception as e:                  # a failed download
            errors.append(f"{path}: {e}; the notes stand without the website's edits")
    return vocabulary(blobs, manifest).parser(), edits


# ========================
# One card
# ========================
def parse_card(parser: PSDParser, psd, rel: str, variant: Optional[int] = None):
    """The parser's CardInfo for a PSD file (or the PSD open) at its repo path (the seam the tests fake)."""
    return parser.parse(psd, rel, variant=variant)


def card_notes(parser: PSDParser, edits: Optional[CardEdits], psd, rel: str) -> list:
    """
    One card's notes, as /update_stats prints them: parse, apply the card's website edits, validate.

    Args:
        parser: PSDParser with the repo's vocabulary
        edits: The repo's site/Card Edits.json, or None
        psd: The PSD file, or the PSD open (a PSDImage)
        rel: Its repo path (classifies the card; names it)

    Returns:
        The notes, [] for a clean card; a PSD with variant tags: each picture's, "A Snack2: " before variant 2's (a
        drawn duplicate's own name: "Gust of Trees: ")
    """
    opened = psd
    if not isinstance(psd, PSDImage):
        try:
            opened = PSDImage.open(psd)
        except Exception:                       # parse_card says why
            opened = psd
    pictures = variants_of(opened) if isinstance(opened, PSDImage) else [1]
    if len(pictures) < 2:
        card = parse_card(parser, opened, rel)
        if edits is not None:
            edits.apply(card.name, card)
        return CardValidator.validate(card)
    notes, drawn = [], drawn_names(opened, rel, edits)
    for n in pictures:
        card = parse_card(parser, opened, rel, n)
        name = card.name if n == 1 else f"{card.name}{n}" if n not in drawn else card.display_name \
            if CardAuthors.key(card.display_name or "") == drawn[n] else drawn[n].title()    # its printed title's case
        if edits is not None:
            edits.apply(name, card)
        notes += [note if n == 1 else f"{name}: {note}" for note in CardValidator.validate(card)]
    return notes


def drawn_names(psd: PSDImage, rel: str, edits: Optional[CardEdits]) -> dict:
    """{n: card key} of the tagged PSD's pictures that picture-only duplicates are (their Card Edits.json dup_of names
    the card: gust of trees, stubborn queen), as picture_numbers matches their PNGs beside it (a tag first); a numbered
    PNG's (A Snack2) stays out."""
    own = CardAuthors.key(card_name(rel))
    keys = [key for key, fields in (edits.cards if edits is not None else {}).items()
            if isinstance(fields.get("dup_of"), str) and CardAuthors.key(fields["dup_of"]) == own and key != own]
    if not keys:
        return {}
    folder = rel[:rel.rfind("/") + 1]
    found = picture_numbers(psd, [f"{folder}{key.replace(' ', '_')}.png" for key in keys])
    return {n: CardAuthors.key(card_name(path)) for path, n in found.items() if CardAuthors.key(card_name(path)) != f"{own}{n}"}


# ========================
# Reports
# ========================
@dataclass
class Card:
    """One card PSD with notes."""
    path: str
    author: str
    notes: list
    new: Optional[list] = None                  # the notes its PSD before the push lacked (None: not compared, all)


@dataclass
class Report:
    """One check's findings: how many PSDs were read, the cards with notes, the errors."""
    checked: int = 0                            # PSDs read
    cards: list = field(default_factory=list)   # Card, in path order
    errors: list = field(default_factory=list)  # "path: why" (a bad blob, an unreadable PSD or edits file)

    def as_dict(self) -> dict:
        """The --json report."""
        return {"checked": self.checked, "cards": [{"path": c.path, "author": c.author, "notes": c.notes,
                                                    **({"new": c.new} if c.new is not None else {})} for c in self.cards],
                "errors": self.errors}


def markdown(report: Report) -> str:
    """The job summary and commit comment: one line per card with notes, else "Nothing to report."; then the errors."""
    out = ["## Card check", "", f"{report.checked} card PSD{'s' if report.checked != 1 else ''} read.", ""]
    if report.cards:
        out += [f"- `{card.path}` ({card.author}): " + "; ".join(map(short, card.notes)) for card in report.cards]
    else:
        out.append("Nothing to report.")
    if report.errors:
        out += ["", "Errors:"] + [f"- {e}" for e in report.errors]
    return "\n".join(out) + "\n"


def discord_payload(report: Report, repository: str = "", after: str = "") -> dict:
    """This check's part of the standards post (standards_post.payload): a line per card with notes told (its new ones
    when compared), a commit link with after; {} when nothing to tell. Nobody is pinged (owner, 2026-10-23)."""
    return payload(repository, after, card_check=report.as_dict())


# ========================
# Sources: files, the GitHub API
# ========================
def _record(report: Report, rel: str, author: str, notes: list) -> None:
    report.checked += 1
    if notes:
        report.cards.append(Card(rel, author, notes))


def _check_one(report: Report, parser: PSDParser, edits: Optional[CardEdits], rel: str, author: str, psd: str) -> None:
    """Read one PSD into the report; an unreadable one is an error line, never a stop."""
    try:
        notes = card_notes(parser, edits, psd, rel)
    except Exception as e:                      # one bad PSD never stops the check
        report.errors.append(f"{rel}: {UNREADABLE} ({e})")
        return
    _record(report, rel, author, notes)


def checkout_root(path: str) -> str:
    """The nearest ancestor of a file holding Types/ (the checkout it is in); "" when none."""
    folder = dirname(abspath(path))
    while not isdir(join(folder, TYPES_DIR)):
        parent = dirname(folder)
        if parent == folder:
            return ""
        folder = parent
    return folder


def check_files(paths: list, author: str = NOBODY) -> Report:
    """
    PSD files given on the command line.

    Each belongs to the checkout of its nearest ancestor holding Types/: its path from there classifies it and the
    checkout's Types/ and edits file apply. A file outside any checkout is an unknown card with no edits. A PSD under
    MDW/ or Markers/ is no card (/update_stats never parses one): an error line.

    Args:
        paths: The files
        author: Who to name for them

    Returns:
        Report
    """
    report = Report()
    contexts: dict = {}
    for path in paths:
        root = checkout_root(path)
        if root not in contexts:
            contexts[root] = ((clone_vocabulary(root).parser(), clone_edits(root, report.errors)) if root
                              else (Vocabulary().parser(), None))
        parser, edits = contexts[root]
        rel = relpath(abspath(path), root).replace("\\", "/") if root else basename(path)
        if not is_card_psd(rel):
            report.errors.append(f"{rel}: not a card PSD (MDW/ and Markers/ hold none; /update_stats never parses it)")
            continue
        _check_one(report, parser, edits, rel, author, path)
    return report


def check_blobs(repository: str, psds: dict, parser: PSDParser, edits: Optional[CardEdits], clone: str = "",
                olds: Optional[dict] = None) -> Report:
    """
    PSDs read by blob sha through the API, in path order (each blob goes to a temp file for the parser). With clone:
    its object store first and kept filled (the guards' clone: they read each pushed PSD whole anyway).

    Args:
        repository: owner/name
        psds: {path: (blob sha, author)}
        parser: PSDParser with the tree's vocabulary
        edits: The tree's edits file, or None
        clone: A clone read first and kept filled (github_api.fetch_verified)
        olds: {path: its blob before the push, "" for a new card} (font_check.old_blobs): each card's new notes

    Returns:
        Report
    """
    report = Report()
    for path in sorted(psds):
        sha, author = psds[path]
        cards = len(report.cards)
        try:
            data = fetch_verified(repository, sha, timeout=API_TIMEOUT, clone=clone)
        except Exception as e:                  # a failed download is an error line
            report.errors.append(f"{path}: {e}")
            continue
        handle, temp = mkstemp(suffix=".psd")
        try:
            with open(handle, "wb") as fh:
                fh.write(data)
            _check_one(report, parser, edits, path, author, temp)
        finally:
            remove(temp)
        _mark_new(report, cards, repository, parser, edits, sha, olds, clone)
    return report


def _mark_new(report: Report, cards: int, repository: str, parser: PSDParser, edits: Optional[CardEdits], sha: str,
              olds: Optional[dict], clone: str) -> None:
    """The card just recorded (past cards): its notes the PSD before the push lacked (the same blob: none; a new card or
    an unreadable old blob: all)."""
    if olds is None or len(report.cards) == cards:
        return
    card = report.cards[-1]
    old = olds.get(card.path, "")
    if old == sha:
        card.new = []
        return
    before = []
    if old:
        handle, temp = mkstemp(suffix=".psd")
        try:
            with open(handle, "wb") as fh:
                fh.write(fetch_verified(repository, old, timeout=API_TIMEOUT, clone=clone))
            before = card_notes(parser, edits, temp, card.path)
        except Exception:                       # the old blob unread: every note is told
            before = []
        finally:
            remove(temp)
    card.new = [note for note in card.notes if note not in before]


# ========================
# Command line
# ========================
def main(args: Optional[list] = None) -> int:
    """The command line: files or the API; the markdown on stdout, --json and --discord files when asked."""
    cli = ArgumentParser(prog="python -m commands.card_check",
                         description="What the bot's parser and validator note about card PSDs (the lines /update_stats prints). "
                                     "Reports, never rejects: findings exit 0.")
    cli.add_argument("--psd", nargs="*", default=[], metavar="PATH", help="PSD files to check (each in the checkout of its nearest ancestor holding Types/)")
    cli.add_argument("--repo", default="", metavar="OWNER/NAME", help="read the pushed PSDs by blob sha through the GitHub API")
    cli.add_argument("--before", default="", metavar="SHA", help="the push's previous head (github.event.before); empty: --after's own files")
    cli.add_argument("--after", default="", metavar="SHA", help="the push's head (github.sha)")
    cli.add_argument("--author", default=NOBODY, metavar="LOGIN", help="who to name when a commit's author has no login (github.actor)")
    cli.add_argument("--whole-tree", action="store_true", help="with --repo: every card PSD of the tree at --after (~813 blob downloads, 16 GB)")
    cli.add_argument("--clone", default="", metavar="DIR", help="with --repo: a clone whose blobs are read first; fetched ones are kept in it")
    cli.add_argument("--json", default="", metavar="OUT", help="write the report as JSON here")
    cli.add_argument("--discord", default="", metavar="OUT", help="write this check's part of the standards post here ({} when nothing to tell)")
    options = cli.parse_args(args)
    token_from_env()
    if options.repo:
        errors: list = []
        try:
            blobs = tree_blobs(options.repo, options.after)
            parser, edits = repo_context(options.repo, blobs, errors)
            olds = None
            if options.whole_tree:
                psds, notes = {p: (s, options.author) for p, s in blobs.items() if is_card_psd(p)}, []
            else:
                psds, _, notes = pushed_files(options.repo, options.before, options.after, options.author, options.clone)
                olds = old_blobs(options.repo, options.before, psds, options.clone, notes)
            report = check_blobs(options.repo, psds, parser, edits, options.clone, olds)
        except Exception as e:                  # the API failed (a 5xx after the retries, no network, a truncated tree): one line
            report, notes = Report(), [f"GitHub API: {e}"]
        report.errors = errors + report.errors + notes
    else:
        report = check_files(options.psd, options.author)
    print(markdown(report), end="")
    if options.json:
        with open(options.json, "w", encoding="utf-8") as fh:
            dump(report.as_dict(), fh, indent=1)
    if options.discord:
        with open(options.discord, "w", encoding="utf-8") as fh:
            dump(discord_payload(report, options.repo, options.after if options.repo else ""), fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'Card',
    'Report',
    'UNREADABLE',
    'Vocabulary',
    'card_notes',
    'check_blobs',
    'check_files',
    'checkout_root',
    'clone_edits',
    'clone_vocabulary',
    'discord_payload',
    'main',
    'markdown',
    'parse_card',
    'repo_context',
    'vocabulary',
]
