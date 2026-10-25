"""
Write tests/goldens/ from the CURRENT parser. By default only the goldens whose PSD changed upstream (or that have none
yet); name card paths to re-pin just those; --all only after a parser change was accepted (validation-and-qa section 4):
    python tests/make_goldens.py [--vocab] [--all | <card path> ...]
(--vocab also re-freezes Types/, Types/Stars/ and the manifest)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import CLONE, blob_sha  # noqa: E402  (also puts the repo on sys.path, with config)
from golden_cards import CARDS, VOCAB, golden_file, parse, parser  # noqa: E402


def freeze_vocab() -> None:
    from commands.card_data import CardBuild
    from commands.psd_analyzer import ICON_MANIFEST_PATH, StatsDatabase
    db = StatsDatabase()
    tr = CardBuild(db)
    tr._populate_types_from_local(str(CLONE))
    stars_dir = CLONE / "Types" / "Stars"
    stars = [tr._star_name(f.name) for f in sorted(stars_dir.iterdir()) if f.name.lower().endswith(".png")]
    manifest = json.loads((CLONE / ICON_MANIFEST_PATH).read_text(encoding="utf-8"))
    VOCAB.parent.mkdir(exist_ok=True)
    VOCAB.write_text(json.dumps({"all_types": db.all_types, "stars": stars, "manifest": manifest},
                                indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _changed(rel: str) -> bool:
    golden = golden_file(rel)
    return not golden.exists() or json.loads(golden.read_text(encoding="utf-8"))["blob"] != blob_sha(CLONE / rel)


def main() -> None:
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


if __name__ == "__main__":
    main()
