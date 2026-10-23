"""
Card check for the card PSDs: what the bot's parser and validator note about a card, the lines /update_stats prints (the
parser's own notes among them are also the website's Parser notes; the validator's are printed only). TTSCardMaker's
card-check.yml Action runs it on every push to main that touches a PSD, Types/ or site/Card Edits.json (job summary, commit
comment, a Discord message pinging the card makers' role).

    python -m commands.card_check [--psd <path>...] [--all <checkout root>]
                                  [--repo <owner/name> --after <sha> [--before <sha>] [--author <login>] [--whole-tree]]
                                  [--json <out>] [--discord <payload out>]
    card_notes(parser, edits, psd, rel) -> one card's notes; check_files / check_checkout / check_blobs -> Report
    markdown(report), discord_payload(report) -> the job summary / commit comment and the webhook body

The notes equal /update_stats' for the same PSD, Types/ names and site/Card Edits.json: the parser's own (NO STARS FOUND,
MISSPELT TYPE, SEVERAL CREATURES ...) and CardValidator's (ABILITY TEXT NOT FOUND, HP NOT FOUND, STATS TOO HIGH ...), after
the card's website edits (problems_ok, allow_high_stats, creatures ...) and ABILITY_EXCLUSIONS. A notification, never a
gate: findings exit 0, an unreadable PSD or a failed API call is one error line. The repo path classifies a card, so
--psd files belong to the checkout of their nearest ancestor holding Types/ (its Types/ and edits file apply); --all
parses a checkout in a process pool; --repo reads the pushed PSDs by blob sha (Types/, the manifest and the edits file
from the tree at --after). No Discord or config imports (the token: config through github_api, else GITHUB_TOKEN);
SPARSE_FILES is the import closure card-check.yml checks out.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from json import dump, loads
from os import cpu_count, remove, walk
from os.path import abspath, basename, dirname, isdir, isfile, join, relpath, splitext
from tempfile import mkstemp
from typing import Optional

from commands.card_edits import CardEdits
from commands.card_models import StatsDatabase, card_name
from commands.font_check import API_TIMEOUT, DISCORD_LIMIT, EXCLUDE_FOLDERS, NOBODY, POOL_CHUNK, POOL_MIN, ROLE_ID, is_card_psd, \
    pushed_files, tree_blobs
from commands.github_api import RemotePSD, blob_text, fetch_verified, range_reads, token_from_env
from commands.psd_parser import CardValidator, IconVocabulary, PSDParser
from global_config import EDITS_PATHS, ICON_MANIFEST_PATH, STARS_DIR, TYPES_DIR

# ========================
# Configuration
# ========================
SPARSE_FILES = ("commands/card_check.py", "commands/font_check.py", "commands/github_api.py", "commands/psd_parser.py",
                "commands/card_models.py", "commands/card_edits.py", "global_config.py")   # the import closure card-check.yml checks out
LINE = "{name} has pushed {card} with: {notes}."                                  # the Discord line per card
CLOSING = ("The bot's next /update_stats prints these too; the parser's own notes (no stars found, misspelt type, several "
           "creatures) also show on the website as Parser notes.")                  # the Discord message's last line
UNREADABLE = "could not be read"
_worker: tuple = ()                         # (parser, edits) of a pool worker, built once by _init_worker


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
    """repo_traverser's reading of a tree: Types/ entries with a dot outside Stars, Types/Stars/ PNGs, the manifest's bytes."""
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
    """The checkout's edits file (the first of EDITS_PATHS present), or None."""
    for rel in EDITS_PATHS:
        path = join(root, rel)
        if isfile(path):
            return _parse_edits(rel, _read(path).decode("utf-8"), errors)
    return None


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
    path = next((p for p in EDITS_PATHS if p in blobs), None)
    if path is not None:
        try:
            edits = _parse_edits(path, blob_text(repository, blobs[path], timeout=API_TIMEOUT), errors)
        except Exception as e:                  # a failed download
            errors.append(f"{path}: {e}; the notes stand without the website's edits")
    return vocabulary(blobs, manifest).parser(), edits


# ========================
# One card
# ========================
def parse_card(parser: PSDParser, psd, rel: str):
    """The parser's CardInfo for a PSD file (or the PSD open) at its repo path (the seam the tests fake)."""
    return parser.parse(psd, rel)


def card_notes(parser: PSDParser, edits: Optional[CardEdits], psd, rel: str) -> list:
    """
    One card's notes, as /update_stats prints them: parse, apply the card's website edits, validate.

    Args:
        parser: PSDParser with the repo's vocabulary
        edits: The repo's site/Card Edits.json, or None
        psd: The PSD file, or the PSD open (a PSDImage)
        rel: Its repo path (classifies the card; names it)

    Returns:
        The notes, [] for a clean card
    """
    card = parse_card(parser, psd, rel)
    if edits is not None:
        edits.apply(card.name, card)
    return CardValidator.validate(card)


# ========================
# Reports
# ========================
@dataclass
class Card:
    """One card PSD with notes."""
    path: str
    author: str
    notes: list


@dataclass
class Report:
    """One check's findings: how many PSDs were read, the cards with notes, the errors."""
    checked: int = 0                            # PSDs read
    cards: list = field(default_factory=list)   # Card, in path order
    errors: list = field(default_factory=list)  # "path: why" (a bad blob, an unreadable PSD or edits file)

    def as_dict(self) -> dict:
        """The --json report."""
        return {"checked": self.checked, "cards": [{"path": c.path, "author": c.author, "notes": c.notes} for c in self.cards],
                "errors": self.errors}


def markdown(report: Report) -> str:
    """The job summary and commit comment: one line per card with notes, else "Nothing to report."; then the errors."""
    out = ["## Card check", "", f"{report.checked} card PSD{'s' if report.checked != 1 else ''} read.", ""]
    if report.cards:
        out.append("Notes (what /update_stats prints; the parser's own ones also show on the website as Parser notes):")
        out += [f"- `{card.path}` ({card.author}): " + "; ".join(card.notes) for card in report.cards]
    else:
        out.append("Nothing to report.")
    if report.errors:
        out += ["", "Errors:"] + [f"- {e}" for e in report.errors]
    return "\n".join(out) + "\n"


def discord_payload(report: Report) -> dict:
    """
    The webhook body: the role mention once, one LINE per card, CLOSING last; {} when nothing to report. Whole lines are
    dropped from the end to fit DISCORD_LIMIT ("... and N more cards"). Only the role may be pinged.
    """
    if not report.cards:
        return {}
    lines = [LINE.format(name=c.author, card=card_name(c.path), notes="; ".join(c.notes)) for c in report.cards]

    def assemble(kept: int) -> str:
        rest = len(lines) - kept
        more = [f"... and {rest} more card{'s' if rest != 1 else ''}"] if rest else []
        return "\n".join([f"<@&{ROLE_ID}>"] + lines[:kept] + more + [CLOSING])

    kept = len(lines)
    content = assemble(kept)
    while len(content) > DISCORD_LIMIT and kept > 0:
        kept -= 1
        content = assemble(kept)
    return {"content": content, "allowed_mentions": {"roles": [ROLE_ID]}}


# ========================
# Sources: files, a checkout, the GitHub API
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


def card_psds(root: str) -> list:
    """(repo path, file path) of every card PSD under a checkout, in path order; MDW/ and Markers/ left out like the traverser."""
    found = []
    for folder, subdirs, files in walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in EXCLUDE_FOLDERS and d != ".git")
        for name in sorted(files):
            rel = relpath(join(folder, name), root).replace("\\", "/")
            if is_card_psd(rel):
                found.append((rel, join(folder, name)))
    return found


def _init_worker(vocab: Vocabulary, edits: Optional[CardEdits]) -> None:
    global _worker
    _worker = (vocab.parser(), edits)


def _notes_of(item: tuple) -> tuple:
    """(rel, file path) -> (notes, None) or (None, the error text) in a pool worker."""
    rel, path = item
    try:
        return card_notes(_worker[0], _worker[1], path, rel), None
    except Exception as e:                      # one bad PSD never stops the check
        return None, str(e)


def check_checkout(root: str) -> Report:
    """
    Every card PSD under a checkout, with its Types/ and edits file.

    Args:
        root: The checkout

    Returns:
        Report; from POOL_MIN PSDs the parses run in a process pool (half the cores), merged in path order
    """
    report = Report()
    vocab, edits = clone_vocabulary(root), clone_edits(root, report.errors)
    found = card_psds(root)
    if len(found) < POOL_MIN:
        parser = vocab.parser()
        for rel, path in found:
            _check_one(report, parser, edits, rel, NOBODY, path)
        return report
    workers = max(1, min((cpu_count() or 2) // 2, -(-len(found) // POOL_CHUNK)))
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(vocab, edits)) as pool:
        for (rel, _), (notes, error) in zip(found, pool.map(_notes_of, found, chunksize=POOL_CHUNK)):
            if error is not None:
                report.errors.append(f"{rel}: {UNREADABLE} ({error})")
            else:
                _record(report, rel, NOBODY, notes)
    return report


def check_blobs(repository: str, psds: dict, parser: PSDParser, edits: Optional[CardEdits], ref: str = "") -> Report:
    """
    PSDs read by blob sha through the API, in path order (each blob goes to a temp file for the parser). With ref (the
    pushed commit) and a psd-tools that reads lazily, a PSD is read by range requests at ref instead: only what the
    parser reads is fetched; one that fails that way is read whole.

    Args:
        repository: owner/name
        psds: {path: (blob sha, author)}
        parser: PSDParser with the tree's vocabulary
        edits: The tree's edits file, or None
        ref: The commit the PSDs are at ("": whole blobs)

    Returns:
        Report
    """
    report = Report()
    ranges = bool(ref) and range_reads()
    for path in sorted(psds):
        sha, author = psds[path]
        if ranges:
            notes = _ranged_notes(repository, ref, path, parser, edits)
            if notes is not None:
                _record(report, path, author, notes)
                continue
        try:
            data = fetch_verified(repository, sha, timeout=API_TIMEOUT)
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
    return report


def _ranged_notes(repository: str, ref: str, path: str, parser: PSDParser, edits: Optional[CardEdits]) -> Optional[list]:
    """One PSD's notes from range reads at ref; None when they failed or the PSD raised (the whole blob decides)."""
    remote = None
    try:
        remote = RemotePSD(repository, ref, path)
        notes = card_notes(parser, edits, remote.psd, path)
        return None if remote.failed else notes
    except Exception as e:
        print(f"{path}: range reads failed ({e}); reading it whole")
        return None
    finally:
        if remote is not None:
            remote.close()


# ========================
# Command line
# ========================
def main(args: Optional[list] = None) -> int:
    """The command line: files, a checkout or the API; the markdown on stdout, --json and --discord files when asked."""
    cli = ArgumentParser(prog="python -m commands.card_check",
                         description="What the bot's parser and validator note about card PSDs (the lines /update_stats prints). "
                                     "Reports, never rejects: findings exit 0.")
    cli.add_argument("--psd", nargs="*", default=[], metavar="PATH", help="PSD files to check (each in the checkout of its nearest ancestor holding Types/)")
    cli.add_argument("--all", default="", metavar="ROOT", help="every card PSD under a checkout (MDW/ and Markers/ excluded)")
    cli.add_argument("--repo", default="", metavar="OWNER/NAME", help="read the pushed PSDs by blob sha through the GitHub API")
    cli.add_argument("--before", default="", metavar="SHA", help="the push's previous head (github.event.before); empty: --after's own files")
    cli.add_argument("--after", default="", metavar="SHA", help="the push's head (github.sha)")
    cli.add_argument("--author", default=NOBODY, metavar="LOGIN", help="who to name when a commit's author has no login (github.actor)")
    cli.add_argument("--whole-tree", action="store_true", help="with --repo: every card PSD of the tree at --after (~813 blob downloads, 16 GB)")
    cli.add_argument("--json", default="", metavar="OUT", help="write the report as JSON here")
    cli.add_argument("--discord", default="", metavar="OUT", help="write the Discord webhook payload here ({} when nothing to report)")
    options = cli.parse_args(args)
    token_from_env()
    if options.repo:
        errors: list = []
        try:
            blobs = tree_blobs(options.repo, options.after)
            parser, edits = repo_context(options.repo, blobs, errors)
            if options.whole_tree:
                psds, notes = {p: (s, options.author) for p, s in blobs.items() if is_card_psd(p)}, []
            else:
                psds, _, notes = pushed_files(options.repo, options.before, options.after, options.author)
            report = check_blobs(options.repo, psds, parser, edits, options.after)
        except Exception as e:                  # the API failed (a 5xx after the retries, no network, a truncated tree): one line
            report, notes = Report(), [f"GitHub API: {e}"]
        report.errors = errors + report.errors + notes
    elif options.all:
        report = check_checkout(options.all)
    else:
        report = check_files(options.psd, options.author)
    print(markdown(report), end="")
    if options.json:
        with open(options.json, "w", encoding="utf-8") as fh:
            dump(report.as_dict(), fh, indent=1)
    if options.discord:
        with open(options.discord, "w", encoding="utf-8") as fh:
            dump(discord_payload(report), fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'CLOSING',
    'Card',
    'LINE',
    'Report',
    'SPARSE_FILES',
    'UNREADABLE',
    'Vocabulary',
    'card_notes',
    'card_psds',
    'check_blobs',
    'check_checkout',
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
