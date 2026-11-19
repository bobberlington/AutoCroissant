"""Parser corpus: every card PSD at _corpus.json's rev, replayed from the bytes a lazy parse read of it (the records
in ~/.cache), parses as tests/goldens/_corpus.json says. A failure lists every card a parser change moves. A PSD with no
record (a fresh cache) or one the parser reads past is parsed from its blob in the clone and recorded, as
make_goldens.py --records does."""
import json

import pytest

from conftest import card_blob
from golden_cards import CORPUS, Ranges, load_record, parse_all, parser, save_record

EXPECTED = json.loads(CORPUS.read_text(encoding="utf-8"))["cards"]
FOLDERS = sorted({key.split("/")[0] for key in EXPECTED})
SHOWN = 8                                   # changed parses shown field by field


@pytest.fixture(scope="module")
def psd_parser():
    return parser()


def _fields(want: dict, got: dict) -> list:
    """(field, old, new) of each difference: the card's fields, the validator, an error."""
    out = [(k, want.get(k), got.get(k)) for k in ("error", "validator") if want.get(k) != got.get(k)]
    old, new = want.get("card") or {}, got.get("card") or {}
    return out + [(k, old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def _parse(psd_parser, clone, rel: str, blob: str) -> dict:
    """parse_all from the blob's record; with none, or a read past it: from the blob, then recorded."""
    record = load_record(blob)
    if record is not None:
        fp = Ranges(*record)
        got = parse_all(psd_parser, fp, rel)
        if not fp.missed:
            return got
    data = card_blob(clone, blob)
    fp = Ranges(len(data), data=data)
    got = parse_all(psd_parser, fp, rel)
    save_record(blob, data, fp.reads)
    return got


@pytest.mark.parametrize("folder", FOLDERS)
def test_corpus(clone, psd_parser, folder):
    cards = {key: want for key, want in EXPECTED.items() if key.startswith(folder + "/")}
    blobs = {key.split("#")[0]: want["blob"] for key, want in cards.items()}
    changed = []
    for rel, blob in sorted(blobs.items()):
        got = _parse(psd_parser, clone, rel, blob)
        for key in sorted({k for k in cards if k.split("#")[0] == rel} | set(got)):
            want = {k: v for k, v in cards.get(key, {}).items() if k != "blob"}
            if got.get(key, {}) != want:
                changed.append((key, _fields(want, got.get(key, {}))))
    lines = [f"{len(changed)} of {len(cards)} parses in {folder}/ changed (once accepted: make_goldens.py --corpus)"]
    for i, (key, fields) in enumerate(changed):
        lines.append(f"  {key}: {', '.join(name for name, _, _ in fields)}")
        lines += [f"    {name}: {old!r:.150} -> {new!r:.150}" for name, old, new in fields] if i < SHOWN else []
    if changed:
        pytest.fail("\n".join(lines), pytrace=False)
