"""
Write tests/goldens/ from the CURRENT parser and the cards at conftest.CARDS_REV. By default only the goldens whose PSD
differs at the pin (or that have none yet); name card paths to redo just those; --all only after a parser change was
accepted (validation-and-qa section 4):
    python tests/make_goldens.py [--vocab] [--all | <card path> ...]
(--vocab also re-freezes Types/, Types/Stars/ and the manifest). The corpus (tests/test_parser_corpus.py):
    python tests/make_goldens.py --records          the records only, for a parser that reads new ranges
    python tests/make_goldens.py --corpus [<rev>]  also _corpus.json, at its rev or a new one (--all does this too)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import _blob, blob_sha, no_lazy, pinned_clone  # noqa: E402  (also puts the repo on sys.path, with config)
from golden_cards import CARDS, CORPUS, RECORDS, VOCAB, Ranges, golden_file, load_record, parse, parse_all, parser, \
    save_record  # noqa: E402

CLONE = pinned_clone()                    # the card repo at CARDS_REV


def freeze_vocab() -> None:
    from commands.card_check import clone_vocabulary
    vocab = clone_vocabulary(str(CLONE))
    VOCAB.parent.mkdir(exist_ok=True)
    VOCAB.write_text(json.dumps({"all_types": vocab.all_types, "stars": vocab.stars, "manifest": vocab.manifest},
                                indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _changed(rel: str) -> bool:
    golden = golden_file(rel)
    return not golden.exists() or json.loads(golden.read_text(encoding="utf-8"))["blob"] != blob_sha(CLONE / rel)


def make_corpus(rev: str, expected: bool) -> None:
    """Record what a lazy parse reads of every card PSD at rev; with expected, write CORPUS from the same parses."""
    from commands.card_data import EXCLUDE_FOLDERS
    from global_config import RULEBOOK_DIR
    tree = subprocess.run(["git", "-C", str(CLONE), "ls-tree", "-r", "-z", rev], capture_output=True, check=True,
                          env=no_lazy()).stdout.decode().split("\0")
    skip = set(EXCLUDE_FOLDERS) | {RULEBOOK_DIR}
    psds = {path: line.split()[2] for line, path in (entry.split("\t", 1) for entry in filter(None, tree))
            if path.endswith(".psd") and not skip & set(path.split("/"))}
    have = subprocess.run(["git", "-C", str(CLONE), "cat-file", "--batch-check"], capture_output=True, check=True,
                          input="".join(f"{sha}\n" for sha in psds.values()).encode(), env=no_lazy()).stdout.decode()
    missing = [line.split()[0] for line in have.splitlines() if line.endswith("missing")]
    if missing:
        sys.exit(f"{len(missing)} of {len(psds)} PSD blobs at {rev[:8]} are not in the clone: pick a rev it has")
    p, cards, wrong, start = parser(), {}, [], time.time()
    for rel, sha in sorted(psds.items()):
        _, data = _blob(CLONE, sha)
        fp = Ranges(len(data), data=data)
        got = parse_all(p, fp, rel)
        save_record(sha, data, fp.reads)
        if parse_all(p, Ranges(*load_record(sha)), rel) != got:
            wrong.append(rel)
        cards.update({key: {"blob": sha, **parsed} for key, parsed in got.items()})
    kept = sum((RECORDS / f"{sha}.z").stat().st_size for sha in set(psds.values()))
    print(f"{len(psds)} records ({kept / 1e6:.1f} MB) in {time.time() - start:.0f} s: {RECORDS}")
    if wrong:
        sys.exit(f"replay differs from the parse (nothing written to {CORPUS.name}): {', '.join(wrong)}")
    if expected:
        dump = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)  # noqa: E731
        lines = [f"  {dump(key)}: {dump(cards[key])}" for key in sorted(cards)]
        CORPUS.write_text(f'{{"rev": "{rev}",\n "cards": {{\n' + ",\n".join(lines) + "\n}}\n", encoding="utf-8")
        print(f"{CORPUS}: {len(cards)} parses")


def main() -> None:
    if "--records" in sys.argv or "--corpus" in sys.argv:
        named = [a for a in sys.argv[1:] if not a.startswith("--")] if "--corpus" in sys.argv else []
        rev = named[0] if named else json.loads(CORPUS.read_text(encoding="utf-8"))["rev"]
        rev = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "--verify", f"{rev}^{{commit}}"], check=True,
                             capture_output=True, text=True, env=no_lazy()).stdout.strip()
        return make_corpus(rev, "--corpus" in sys.argv)
    named = [a for a in sys.argv[1:] if not a.startswith("--")]
    unknown = [rel for rel in named if rel not in CARDS]
    if unknown:
        sys.exit(f"not in CARDS (tests/golden_cards.py): {', '.join(unknown)}")
    if "--vocab" in sys.argv or not VOCAB.exists():
        freeze_vocab()
    cards = CARDS if "--all" in sys.argv else named or [rel for rel in CARDS if _changed(rel)]
    if not cards:
        print("Every golden matches its PSD; nothing re-pinned.")
    p = parser()
    for rel in cards:
        out = {"path": rel, "blob": blob_sha(CLONE / rel), **parse(p, CLONE, rel)}
        golden_file(rel).write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(rel)
    if "--all" in sys.argv:
        make_corpus(json.loads(CORPUS.read_text(encoding="utf-8"))["rev"], True)


if __name__ == "__main__":
    main()
