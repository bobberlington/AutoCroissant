"""Parser corpus: every card PSD at _corpus.json's rev, replayed from the bytes a lazy parse read of it (the records
in ~/.cache, no clone, no PSD), parses as tests/goldens/_corpus.json says. A failure lists every card a parser change
moves; a read outside a record is no card change: rebuild the records (python tests/make_goldens.py --records)."""
import json

import pytest

from golden_cards import CORPUS, RECORDS, Ranges, load_record, parse_all, parser

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


@pytest.mark.parametrize("folder", FOLDERS)
def test_corpus(psd_parser, folder):
    cards = {key: want for key, want in EXPECTED.items() if key.startswith(folder + "/")}
    blobs = {key.split("#")[0]: want["blob"] for key, want in cards.items()}
    changed, stale, seen = [], [], 0
    for rel, blob in sorted(blobs.items()):
        record = load_record(blob)
        if record is None:
            stale.append(rel)
            continue
        fp = Ranges(*record)
        got = parse_all(psd_parser, fp, rel)
        seen += 1
        if fp.missed:
            stale.append(rel)
            continue
        for key in sorted({k for k in cards if k.split("#")[0] == rel} | set(got)):
            want = {k: v for k, v in cards.get(key, {}).items() if k != "blob"}
            if got.get(key, {}) != want:
                changed.append((key, _fields(want, got.get(key, {}))))
    if not seen:
        pytest.skip(f"no corpus records in {RECORDS}: python tests/make_goldens.py --records")
    lines = [f"{len(changed)} of {len(cards)} parses in {folder}/ changed (once accepted: make_goldens.py --corpus)"]
    for i, (key, fields) in enumerate(changed):
        lines.append(f"  {key}: {', '.join(name for name, _, _ in fields)}")
        lines += [f"    {name}: {old!r:.150} -> {new!r:.150}" for name, old, new in fields] if i < SHOWN else []
    if stale:
        lines.append(f"{len(stale)} PSDs read bytes their records lack, not compared (python tests/make_goldens.py "
                     f"--records): {', '.join(stale[:10])}")
    if changed or stale:
        pytest.fail("\n".join(lines), pytrace=False)
