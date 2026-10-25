"""Parse every card PSD with the bot's parser (HEAD snapshot) and dump every field to JSON. Usage: parse_all_ns.py <out.json> [ROOT]"""
import json, sys, os, time
from concurrent.futures import ProcessPoolExecutor
from os.path import expanduser, relpath
from pathlib import Path

ROOT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/psdreg/ac"
os.chdir(ROOT); sys.path.insert(0, ROOT)
from global_config import LOCAL_DIR_LOC
CLONE = Path(expanduser(LOCAL_DIR_LOC)).resolve()
_parser = None

def _init():
    global _parser
    from commands.card_data import CardBuild
    from commands.psd_analyzer import StatsDatabase
    from commands.psd_parser import PSDParser
    db = StatsDatabase()
    tr = CardBuild(db)
    tr._populate_types_from_local(str(CLONE))
    _parser = PSDParser(db.all_types, db, tr._icons_from_local(str(CLONE)))

def one(path):
    from commands.psd_parser import CardValidator
    rel = relpath(path, CLONE).replace("\\", "/")
    try:
        card = _parser.parse(path, rel)
        d = card.to_dict(); d.pop("timestamp", None); d.pop("author", None)
        d["validator"] = CardValidator.validate(card)
        return rel, d
    except Exception as e:
        return rel, {"ERROR": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    paths = sorted(str(p) for p in CLONE.rglob("*.psd") if "/MDW/" not in str(p) and "/Markers/" not in str(p))
    t = time.time(); out = {}
    with ProcessPoolExecutor(max_workers=16, initializer=_init) as ex:
        for i, (rel, d) in enumerate(ex.map(one, paths, chunksize=4), 1):
            out[rel] = d
            if i % 100 == 0: print(i, "/", len(paths), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=str, sort_keys=True)
    import psd_tools; print(f"psd-tools {psd_tools.__version__} {sys.version.split()[0]}: {len(out)} cards, {sum('ERROR' in d for d in out.values())} errors, {time.time()-t:.0f}s")
