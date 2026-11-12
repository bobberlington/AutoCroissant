"""Parser golden cards: the PSDs, the frozen icon vocabulary and the parse that both the test and make_goldens.py use.
Also the corpus: every card PSD at one card repo commit, replayed from the bytes a lazy parse read of it."""
import json
import zlib
from bisect import bisect_right
from pathlib import Path

import pytest

# ========================
# Configuration
# ========================
GOLDENS = Path(__file__).resolve().parent / "goldens"
VOCAB = GOLDENS / "_vocab.json"          # Types/, Types/Stars/ and the stoogedings manifest when the goldens were made
CORPUS = GOLDENS / "_corpus.json"        # every card's parse at its "rev" (make_goldens.py --corpus)
RECORDS = Path.home() / ".cache" / "autocroissant-tests" / "corpus"   # per PSD blob: the bytes a parse read
CERTIFIED = ("Auxiliary/Minions/Mini_Doomer.psd",                       # validation-and-qa section 3
             "Field/1 Stars/The_Freezer.psd",
             "Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd")
CARDS = CERTIFIED + (
    "Creatures/Kirby/3 Stars/Pix.psd",                                  # 3 creatures
    "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd",           # atk 10 + 2; title in 3 layers
    "Items/Heal/2 Stars/A_Snack.psd",                                   # hidden titles of 2 duplicates
    "Auxiliary/Minions/Anubisath_Guardian.psd",                         # empty ability; card_data's edited Mini Doomer
    "Auxiliary/Minions/God_Tamer.psd",                                  # 5 stars in a Stars group (none until 2026-10-22), empty ability
    "Auxiliary/Status Effects/Debuffs/1 Stars/Poison.psd",              # status-effect folder
    "N.M.E/Termination_Shock.psd",                                      # star count (STAR_CORE_ALPHA)
)


# ========================
# Helpers
# ========================
def golden_file(rel: str) -> Path:
    return GOLDENS / (rel[:-4].replace("/", "__").replace(" ", "_") + ".json")


def parser():
    """A PSDParser on an empty database with the frozen vocabulary (fresh parse: no author or series preservation)."""
    from commands.card_models import StatsDatabase
    from commands.psd_parser import IconVocabulary, PSDParser
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    db = StatsDatabase()
    db.all_types = list(vocab["all_types"])
    return PSDParser(db.all_types, db, IconVocabulary.build(db.all_types, vocab["stars"], vocab["manifest"]))


@pytest.fixture(scope="module")
def psd_parser(clone):
    return parser()


def parse(psd_parser, clone: Path, rel: str, variant=None) -> dict:
    """Everything the golden pins: to_dict() and the validator's problems. clone: a folder, or the PSD open."""
    from commands.psd_parser import CardValidator
    card = psd_parser.parse(clone if not isinstance(clone, Path) else str(clone / rel), rel, variant=variant)
    out = {"card": card.to_dict()}
    out["validator"] = CardValidator.validate(card)
    return out


# ========================
# Corpus
# ========================
class Ranges:
    """A PSD file as a lazy open reads it (seek, tell, read). Over its bytes: notes each range read. Over a record's
    segments: a byte not recorded sets missed and raises OSError."""
    def __init__(self, size: int, segments=(), data: bytes = None):
        self.size, self.data, self.pos, self.missed, self.reads = size, data, 0, False, []
        self.segments = list(segments)
        self.starts = [start for start, _ in self.segments]

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset + (0, self.pos, self.size)[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        start = self.pos
        stop = self.size if n is None or n < 0 else min(start + n, self.size)
        if stop <= start:
            return b""
        self.pos = stop
        if self.data is not None:
            self.reads.append((start, stop))
            return self.data[start:stop]
        i = bisect_right(self.starts, start) - 1
        if i >= 0 and stop <= self.starts[i] + len(self.segments[i][1]):
            first, data = self.segments[i]
            return data[start - first:stop - first]
        self.missed = True
        raise OSError(f"bytes {start}-{stop} not in the record")


def parse_all(psd_parser, fp: Ranges, rel: str) -> dict:
    """{key: parse} as the card data build parses the PSD: once, or once per picture (rel#n for n > 1); a parse that
    raises: {"error": ...}."""
    from psd_tools import PSDImage
    from commands.psd_parser import variants_of
    try:
        psd = PSDImage.open(fp, lazy=True)
        pictures = variants_of(psd)
        if len(pictures) < 2:
            return {rel: parse(psd_parser, psd, rel)}
        return {rel if n == 1 else f"{rel}#{n}": parse(psd_parser, psd, rel, n) for n in pictures}
    except Exception as error:
        return {rel: {"error": f"{type(error).__name__}: {error}"}}


def load_record(blob: str):
    """(size, [(start, bytes)]) of a PSD blob's record; None when it has none."""
    path = RECORDS / f"{blob}.z"
    if not path.is_file():
        return None
    head, _, body = zlib.decompress(path.read_bytes()).partition(b"\n")
    meta, segments, at = json.loads(head), [], 0
    for start, stop in meta["spans"]:
        segments.append((start, body[at:at + stop - start]))
        at += stop - start
    return meta["size"], segments


def save_record(blob: str, data: bytes, reads: list) -> None:
    """Add the ranges a parse read to the blob's record (what earlier parsers read stays)."""
    old = load_record(blob)
    spans = sorted(reads + [(start, start + len(part)) for start, part in (old[1] if old else ())])
    merged = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    head = json.dumps({"size": len(data), "spans": merged}).encode()
    RECORDS.mkdir(parents=True, exist_ok=True)
    temp = RECORDS / f".{blob}.z"
    temp.write_bytes(zlib.compress(head + b"\n" + b"".join(data[a:b] for a, b in merged), 6))
    temp.replace(RECORDS / f"{blob}.z")
