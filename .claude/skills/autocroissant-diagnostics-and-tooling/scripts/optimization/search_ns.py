"""Search timings + every result, on a sandbox copy. Usage: search_ns.py ROOT OUT.json (diff two OUTs for identity)."""
import json, os, random, statistics, subprocess, sys, time
ROOT, OUT = sys.argv[1], sys.argv[2]
os.chdir(ROOT); sys.path.insert(0, ROOT)
from os.path import expanduser
from commands.psd_analyzer import stats_db, CardEdits, EDITS_PATH
from commands.query_card import card_repo
CLONE = expanduser("~/Desktop/TTSCardMaker")

from commands.card_db import read_db
stats_db.stats, stats_db.old_stats, _, _ = read_db("card_data", details=False); card_repo.prep_dataframes()
out = subprocess.run(["git", "-C", CLONE, "ls-files", "-s"], capture_output=True, text=True).stdout
tree = {"tree": [{"path": l.split("\t", 1)[1], "sha": l.split()[1]} for l in out.splitlines()]}
sha = next(i["sha"] for i in tree["tree"] if i["path"] == EDITS_PATH)
card_repo._edit_names = (sha, CardEdits.parse(open(os.path.join(CLONE, EDITS_PATH), encoding="utf-8").read()).names())
card_repo._populate_site_ids = lambda: None          # offline
card_repo._fetch_tree = lambda repo: (tree, "", 200, "")
card_repo.populate_files()

rnd = random.Random(7)
abil = [a for a in card_repo.cards_dff["ability"] if a and len(a.split()) > 4]
def phrase(k):
    w = rnd.choice(abil).split(); i = rnd.randrange(0, max(1, len(w) - k)); return " ".join(w[i:i + k])
plain = [phrase(rnd.choice([1, 2, 3])) for _ in range(60)]
plain += [phrase(rnd.choice([6, 10, 15])) for _ in range(20)]                      # long phrases: fuzzy hits
plain += [rnd.choice(abil)[:rnd.randrange(20, 120)] for _ in range(10)]          # whole-ability prefixes
plain += ["Damage", "DRAW 2", "", "x", "a" * 300]                                   # case, empty, tiny, autojunk-length
other = ['"' + phrase(1) + '"' for _ in range(10)] + ["!" + phrase(1) for _ in range(5)]
other += ["hp>5", "def<=3", "atk>=7", "spd==4", "stars<3", "type==creature", "type~aux", "series~kirby", "subtype==attack", "hp>9"]
other += [f"{phrase(1)} | {phrase(1)}" for _ in range(5)] + [f"type==creature & {phrase(1)}" for _ in range(5)] + ["type==creature & hp>7"] * 5
rb_abil = [a for a in card_repo.rulebook_dff["ability"] if a and len(a.split()) > 4]
rb = []
for _ in range(100):
    w = rnd.choice(rb_abil).split(); i = rnd.randrange(0, len(w) - 2); rb.append(" ".join(w[i:i + rnd.choice([1, 2])]))
names = sorted(stats_db.stats)
nm = []
for i in range(200):
    n = rnd.choice(names).lower()
    if i % 10 < 5: nm.append(n)                                      # exact
    elif i % 10 < 8:                                                 # one typo
        j = rnd.randrange(len(n)); nm.append(n[:j] + n[j + 1:])
    else: nm.append(n.split()[0])                                    # first word
nm += ["Items/" + names[0], "", "zzzz", "x" * 250]

def run(fn, qs, reps=3):
    res, ts = [], []
    for q in qs:
        best = None
        for _ in range(reps):
            a = time.perf_counter(); r = fn(q); d = time.perf_counter() - a
            best = d if best is None else min(best, d)
        res.append(r); ts.append(best)
    return res, ts

def idx(df):
    return [list(map(str, df.index)), str(df.index.dtype), [str(t) for t in df.dtypes], df.to_csv()]   # order, dtypes, cells

def stat(ts):
    s = sorted(ts); return {"n": len(ts), "total_ms": round(sum(ts) * 1e3, 1), "mean_ms": round(statistics.mean(ts) * 1e3, 3),
                            "median_ms": round(s[len(s) // 2] * 1e3, 3), "max_ms": round(s[-1] * 1e3, 3)}

report, results = {}, {}
card_repo.ability_search_engine(plain[0])                            # warm
r, t = run(card_repo.ability_search_engine, plain); results["plain"] = [idx(x) for x in r]; report["ability plain"] = stat(t)
r, t = run(card_repo.ability_search_engine, other); results["other"] = [idx(x) for x in r]; report["ability quoted/NOT/stat/combos"] = stat(t)
r, t = run(card_repo.rulebook_search_engine, rb); results["rulebook"] = [idx(x) for x in r]; report["rulebook"] = stat(t)
r, t = run(card_repo._find_closest_match, nm); results["names"] = r; report["_find_closest_match"] = stat(t)
ex = [q for i, q in enumerate(nm[:200]) if i % 10 < 5]
report["_find_closest_match exact names"] = stat(run(card_repo._find_closest_match, ex)[1])
r, t = run(card_repo.get_card_path, nm); results["paths"] = r; report["get_card_path"] = stat(t)
for k, v in report.items(): print(f"{k:34s} {v}")
json.dump({"report": report, "results": results}, open(OUT, "w"))
