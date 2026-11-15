"""
discord_preview check: every published version's fields read again from its webp (docs/Future/CARD_TIMELINE.md,
Verification: "every published version's fields re-read from its webp agree with timeline.json"). The readers are the
timelines' own: scripts/timeline_text.py (Vision OCR of the whole picture and its title band, the title's nearest name,
the stars by the era's calibrated rule, the ability's words, the types' icons) and commands/era_bars.py (the bars by
the era's readers). Per field:
  same      the read agrees (name: the OCR'd title within NAME_RATIO, or its nearest name; ability: at least
            ABILITY_RECALL of the published words read; stats: every stat read is the published one; stars: the count;
            types: every published type's icon at least TYPE_ON)
  near      ability words between ABILITY_NEAR and ABILITY_RECALL
  unread    the reader read nothing there (a field the readers can't see on this picture)
  eye       types published as none: an absent icon can't be read again (a recoloured one, Nailsage Sly's wind, matches
            no Types/ picture, while some picture scores TYPE_ON on every card's band): each listed for the eye
  differs   anything else: listed for the eye (check.txt names each)
-> derived/preview/check.jsonl (one row per version: its fields' results and reads) and check.txt (counts, the
differing ones). Nothing is changed: a field the eye rules wrong is fixed upstream (the chain) and the preview rebuilt.
"""
import importlib.util
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from os.path import dirname, join

from commands import discord_match as dm

OUT = ("derived", "preview")
HISTORY = ("derived", "preview", "history")
TEXT_RAW = ("derived", "readers", "text", "raw.jsonl")
BAR_READS = ("derived", "readers", "bars", "reads.jsonl")
CALIBRATION = ("derived", "readers", "text", "calibration.json")
TEXT = join(dirname(dirname(__file__)), "scripts", "timeline_text.py")
NAME_RATIO = 0.8
ABILITY_RECALL, ABILITY_NEAR = 0.75, 0.5
TYPE_ON = 0.6
READ_KINDS = {"name": str, "ability": str, "stars": int, "stats": dict, "types": list}   # fields check_one reads
DEFAULT_STAR_RULE = {"ncc": 0.7, "mad": 45}
TOKEN = re.compile(r"\[[^\]]*\]|\([0-9X/G-]*(?:CD|U|PT|PGT|T|GT)\)")     # icon tokens and timing tags: not OCR words
_TT = None


def text_readers():
    """scripts/timeline_text.py, loaded once per process."""
    global _TT
    if _TT is None:
        spec = importlib.util.spec_from_file_location("timeline_text", TEXT)
        _TT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_TT)
    return _TT


def words(text: str) -> list:
    return text_readers().ability_words(TOKEN.sub(" ", text or ""))


def recall(published: str, read: str) -> float:
    want = [w for w in words(published) if not w.startswith("<")]
    if not want:
        return 1.0
    got = Counter(words(read))
    hit = 0
    for w in want:
        if got[w] > 0:
            got[w] -= 1
            hit += 1
    return hit / len(want)


def result(same: bool, seen) -> str:
    """unread when nothing was read, else same or differs."""
    return "unread" if not seen else "same" if same else "differs"


def check_one(job: dict) -> dict:
    """One version: its webp OCR'd and read, each published field compared."""
    tt = text_readers()
    from commands import era_bars
    import cv2
    import numpy as np
    out = {"card": job["card"], "id": job["id"], "image": job["image"], "fields": {}}
    try:
        item = {"id": job["id"], "src": "discord", "file": job["path"]}
        rec = tt.ocr_task(item)
        if rec.get("error"):
            return {**out, "error": rec["error"]}
        f = job["fields"]
        rule = job.get("star_rule") or DEFAULT_STAR_RULE
        parts = {"ability"} if "ability" in f else set()     # only what is published (types: its own check below)
        if isinstance(f.get("stars"), int):
            parts.add("blobs" if rule.get("kind") else "stars")
        rd = tt.read_one(item, rec, parts)
        era = job.get("era") or rd.get("era")
        out["era"] = era
        res = out["fields"]
        if "name" in f:
            title = rd.get("title") or {}
            text = tt.norm(title.get("text"))
            r = SequenceMatcher(None, tt.norm(f["name"]), text, autojunk=False).ratio() if text else 0.0
            ok = r >= NAME_RATIO or tt.norm(title.get("name")) == tt.norm(f["name"])
            res["name"] = {"r": result(ok, text), "read": title.get("text"), "ratio": round(r, 3)}
        if "ability" in f and isinstance(f["ability"], str):
            read = " ".join(filter(None, [(rd.get(k) or {}).get("text")
                                          for k in ("ability_fix", "ability_raw", "ability_icons")]))
            whole = " ".join(line[0] for line in (rec.get("fix") or []) + (rec.get("raw") or []))   # anywhere on it
            r = max(recall(f["ability"], read), recall(f["ability"], whole))
            near = ABILITY_NEAR <= r < ABILITY_RECALL
            res["ability"] = {"r": "near" if near else result(r >= ABILITY_RECALL, (read + whole).strip()),
                              "recall": round(r, 3)}
        if "stars" in f and isinstance(f["stars"], int):
            n, _ = (tt.count_blobs(rd.get("star_blobs"), rule) if rule.get("kind")
                    else tt.count_stars(rd.get("stars"), rule))
            res["stars"] = {"r": result(n == f["stars"], n is not None), "read": n}
        if isinstance(f.get("stats"), dict):
            img = tt.load(item)
            lines = [line[:6] for line in rec.get("fix") or []]
            got = era_bars.read_card(img, lines, era=era if era in era_bars.CREATURE_ERAS + ("D-item",) else "D2",
                                     title=f.get("name") or "", card=job["card"])
            halves = [got.get("stats") or {}, got.get("stats2") or {}]
            want = {f"{k}{i or ''}": c[k] for i, c in enumerate(f["stats"]["creatures"][:2]) for k in c} if \
                isinstance(f["stats"].get("creatures"), list) else {k: v for k, v in f["stats"].items()}
            read = {k: (halves[1 if k[-1] == "2" else job.get("half", 0)].get(k.rstrip("2")) or {}).get("v")
                    for k in want}         # two creatures: the bottom half's keys end '2'; a split half its own rows
            seen = {k: v for k, v in read.items() if v is not None}
            diff = {k: [want[k], v] for k, v in seen.items() if v != want[k]}
            res["stats"] = {"r": result(not diff, seen), "read": read, **({"diff": diff} if diff else {})}
        if f.get("types") == []:                 # none published: for the eye (an absent icon can't be read again)
            res["types"] = {"r": "eye"}
        elif isinstance(f.get("types"), list):
            a = np.asarray(tt.load(item).convert("RGB"))
            H0, W0 = a.shape[:2]
            work = cv2.resize(a, (tt.WORK_W, max(2, round(tt.WORK_W * H0 / W0))), interpolation=cv2.INTER_AREA)
            names = [tt.canon_types([t])[0] for t in f["types"]]
            known = {tt.canon_types([i[0]])[0] for i in tt.bank("types")}
            have = [t for t in names if t in known]
            pres = tt.type_presence(work, tt.ICON_BAND[tt.shape(W0, H0)], have) if have else {}
            best = {t: (pres.get(t) or [[0.0]])[0][0] for t in have}
            res["types"] = {"r": result(all(v >= TYPE_ON for v in best.values()), have),
                            "read": {t: round(v, 3) for t, v in best.items()}}
            if len(have) < len(names):
                res["types"]["no_icon"] = [t for t in names if t not in known]
        return out
    except Exception as e:                       # one picture never stops the check: listed as an error
        return {**out, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def jobs_of(root: str) -> list:
    """Every published picture version with a sure field: its webp, fields, era and the era's star rule."""
    hist = join(root, *HISTORY)
    index = dm.read_json(join(hist, "index.json"))
    calib = (dm.read_json(join(root, *CALIBRATION), {}) or {}).get("eras", {})
    old_era = {}
    for line in open(join(root, *TEXT_RAW)):
        r = json.loads(line)
        if r.get("id", "").startswith("o:") and r.get("era"):
            old_era[r["id"][2:14]] = r["era"]
    jobs = []
    for cid, row in sorted(index["cards"].items()):
        t = dm.read_json(join(hist, row["dir"], "timeline.json"))
        for e in t["entries"]:
            if not e.get("image") or not e.get("fields"):
                continue
            src = "old" if e["kind"] == "old repo" else "discord"
            era = e.get("era") or (old_era.get(e["image"][2:14]) if src == "old" else None)
            stars = (calib.get(f"{era}/{src}") or calib.get(f"{era}/discord") or {}).get("stars") or {}
            rule = stars.get("how", {}).get("rule")
            jobs.append({"card": cid, "id": e["id"], "image": e["image"], "path": join(hist, row["dir"], e["image"]),
                         "era": era, "star_rule": rule, "kind": e["kind"],
                         "fields": {k: v["value"] for k, v in e["fields"].items()},
                         "half": 1 if str((e["fields"].get("stats") or {}).get("half", "")).startswith("bottom") else 0})
    return jobs


def check(root: str, workers: int = 3, redo: bool = False):
    """redo: keep the last check's rows (a field no longer published left out), read again those with an error, an
    ability that didn't read the same (the ability rule changed last), or a field published since or read now that
    has no result there (types none published: read since); also the rows an unfinished run left in
    check.partial.jsonl (each row is kept there as it is read; removed when check.jsonl is written)."""
    root = dm.check_root(root)
    jobs = jobs_of(root)
    out = join(root, *OUT)
    partial = join(out, "check.partial.jsonl")
    last = {}
    if redo:
        for r in dm.read_jsonl(join(out, "check.jsonl")):
            if not r.get("error") and (r.get("fields", {}).get("ability") or {"r": "same"})["r"] == "same":
                last[(r["card"], r["id"], r["image"])] = r
        for r in dm.read_jsonl(partial):                      # an unfinished run's reads: the rules of now
            if not r.get("error"):
                last[(r["card"], r["id"], r["image"])] = r
    key = lambda j: (j["card"], j["id"], j["image"])
    fresh = lambda r, j: all(k in r.get("fields", {}) and r["published"].get(k) == v
                             for k, v in j["fields"].items() if isinstance(v, READ_KINDS.get(k, ()))) and \
        r.get("half", 0) == j.get("half", 0)                  # a split half read from the other rows: again
    kept = {i: last[key(j)] for i, j in enumerate(jobs) if key(j) in last and fresh(last[key(j)], j)}
    todo = [j for i, j in enumerate(jobs) if i not in kept]
    print(f"check: {len(jobs)} versions with sure fields, {len(todo)} to read", flush=True)
    done = {}
    with open(partial, "a", encoding="utf-8") as fh, ProcessPoolExecutor(workers) as pool:
        for i, (job, r) in enumerate(zip(todo, pool.map(check_one, todo, chunksize=4)), 1):
            done[(job["card"], job["id"], job["image"])] = r
            fh.write(json.dumps({**r, "kind": job["kind"], "published": job["fields"],
                                 **({"half": job["half"]} if job.get("half") else {})}, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 250 == 0:
                print(f"check: {i}/{len(todo)}", flush=True)
    rows = []
    for i, job in enumerate(jobs):
        r = kept.get(i) or done[(job["card"], job["id"], job["image"])]
        r = {**r, "fields": {k: v for k, v in r.get("fields", {}).items() if k in job["fields"]}}
        rows.append({**r, "kind": job["kind"], "published": job["fields"],
                     **({"half": job["half"]} if job.get("half") else {})})
    with open(join(out, "check.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.remove(partial)
    tally = Counter((f, v["r"]) for r in rows for f, v in r.get("fields", {}).items())
    fields = sorted({f for f, _ in tally})
    lines = [f"PREVIEW CHECK: {len(rows)} published versions with sure fields, read again from their webps",
             f"errors: {sum(1 for r in rows if r.get('error'))}"]
    for f in fields:
        n = sum(v for (g, _), v in tally.items() if g == f)
        got = ", ".join(f"{k} {tally[(f, k)]}" for k in ("same", "near", "unread", "eye", "differs") if tally[(f, k)])
        lines.append(f"{f}: {n} published; {got}")
    lines.append("")
    lines.append("DIFFERS (each for the eye: card, version, field, published -> read)")
    for r in rows:
        for f, v in r.get("fields", {}).items():
            if v["r"] == "differs":
                pub = json.dumps(r["published"].get(f), ensure_ascii=False)[:120]
                lines.append(f"  {r['card']} {r['id']} {r['image']} {f}: {pub} -> "
                             f"{json.dumps({k: v[k] for k in v if k != 'r'}, ensure_ascii=False)[:160]}")
    lines += ["", "EYE (types published as none: each for the eye; card, version, image)"]
    lines += [f"  {r['card']} {r['id']} {r['image']}" for r in rows if (r.get("fields", {}).get("types") or {}).get("r") == "eye"]
    dm.write_text(join(out, "check.txt"), "\n".join(lines) + "\n")
    print("\n".join(lines[:2 + len(fields)]), flush=True)
    return rows
