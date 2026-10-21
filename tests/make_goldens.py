"""
Write tests/goldens/ from the CURRENT parser. Only after a parser change was accepted (validation-and-qa section 4),
or to re-pin a golden whose PSD changed upstream:
    python tests/make_goldens.py [--vocab]      (--vocab also re-freezes Types/, Types/Stars/ and the manifest)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import CLONE, blob_sha  # noqa: E402  (also puts the repo on sys.path, with config)
from golden_cards import CARDS, VOCAB, golden_file, parse, parser  # noqa: E402


def freeze_vocab() -> None:
    from commands.psd_analyzer import ICON_MANIFEST_PATH, RepositoryTraverser, StatsDatabase
    db = StatsDatabase()
    tr = RepositoryTraverser(db)
    tr._populate_types_from_local(str(CLONE))
    stars_dir = CLONE / "Types" / "Stars"
    stars = [tr._star_name(f.name) for f in sorted(stars_dir.iterdir()) if f.name.lower().endswith(".png")]
    manifest = json.loads((CLONE / ICON_MANIFEST_PATH).read_text(encoding="utf-8"))
    VOCAB.parent.mkdir(exist_ok=True)
    VOCAB.write_text(json.dumps({"all_types": db.all_types, "stars": stars, "manifest": manifest},
                                indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if "--vocab" in sys.argv or not VOCAB.exists():
        freeze_vocab()
    p = parser()
    for rel in CARDS:
        out = {"path": rel, "blob": blob_sha(CLONE / rel), **parse(p, CLONE, rel)}
        golden_file(rel).write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(rel)


if __name__ == "__main__":
    main()
