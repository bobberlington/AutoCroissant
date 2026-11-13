"""
Why today's parser reads no ability or no stats on old creature/minion PSDs (oldgit_index.py's reads): the layers
behind each such read, one pass, read-only. Writes diag_empty_reads.json beside the reads ({blob: {ability, stats}});
oldgit_index.py report summarizes it.

    python oldgit_diag.py
"""
import io
import json
import re
import subprocess
from collections import Counter
from os.path import join

from psd_tools import PSDImage

from oldgit_index import OG, OUT, read_jsonl


def main():
    todo = {}
    for r in read_jsonl(join(OUT, "reads.jsonl")):
        if r["kind"] == "psd" and r.get("read") and r["read"]["type"] in ("creature", "minion"):
            no_ability = not (r["read"]["ability"] or "").strip()
            no_stats = any("NO STATS FOUND" in p for p in r["read"]["problems"])
            if no_ability or no_stats:
                todo.setdefault(r["blob"], [no_ability, no_stats])
    cat = subprocess.Popen(OG + ["cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    found = {}
    for sha, (no_ability, no_stats) in todo.items():
        cat.stdin.write(sha.encode() + b"\n")
        cat.stdin.flush()
        head = cat.stdout.readline().split()
        data = cat.stdout.read(int(head[2]))
        cat.stdout.read(1)
        psd = PSDImage.open(io.BytesIO(data))
        texts = []
        for layer in psd.descendants():
            if layer.kind == "type" and layer.is_visible():
                try:
                    texts.append((layer, layer.text or ""))
                except Exception:
                    texts.append((layer, ""))
        long_low = [l for l, t in texts if l.bbox[1] > 0.6 * psd.height and len(re.sub(r"\s", "", t)) >= 15]
        named = [l for l in psd.descendants() if l.kind == "type" and l.name.strip().lower().startswith("ability")]
        groups = [g for g in psd.descendants() if g.is_group() and re.search(r"stat|bar", g.name, re.I)]
        bars = [c for g in groups for c in g if c.kind in ("pixel", "shape", "smartobject")]
        named_bars = [c for c in bars if re.match(r"(hp|def|atk|spd|health|defense|attack|speed)\b", c.name, re.I)]
        out = {}
        if no_ability:
            if named and all(not l.is_visible() for l in named):
                out["ability"] = "Ability layer hidden"
            elif named:
                out["ability"] = "an Ability layer, empty or unread"
            elif long_low:
                out["ability"] = "ability text in a layer named after its text (no 'Ability' layer)"
            else:
                out["ability"] = "no ability text on the card"
        if no_stats:
            if groups and bars and not named_bars:
                out["stats"] = f"bars: {'one layer per row' if len(bars) <= 6 else 'many unnamed layers'} in " \
                               f"'{groups[0].name}', unnamed"
            elif not groups:
                out["stats"] = "no stat/bar group"
            else:
                out["stats"] = "named bar layers (one pixel layer per row, e.g. 'HP'/'Healthbar'), unread"
        found[sha] = out
    cat.stdin.close()
    cat.wait()
    with open(join(OUT, "diag_empty_reads.json"), "w") as f:
        json.dump(found, f, indent=0)
    for field in ("ability", "stats"):
        print(field, Counter(v[field] for v in found.values() if field in v).most_common())


if __name__ == "__main__":
    main()
