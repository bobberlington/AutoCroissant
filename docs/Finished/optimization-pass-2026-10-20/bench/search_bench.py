"""Time the three search paths on the real stats.pkl (sandbox copy, read-only). Usage: search_bench.py [profile]"""
import os, sys, time, random, subprocess, statistics
ROOT = "/tmp/acprof/sb"; os.chdir(ROOT); sys.path.insert(0, ROOT)
from os.path import expanduser
from commands.psd_analyzer import stats_db, CardEdits, EDITS_PATH
from commands.query_card import card_repo
CLONE = expanduser("~/Desktop/TTSCardMaker")

def setup():
    t = time.perf_counter(); stats_db.load(); t_load = time.perf_counter() - t
    t = time.perf_counter(); card_repo.prep_dataframes(); t_prep = time.perf_counter() - t
    out = subprocess.run(["git", "-C", CLONE, "ls-files", "-s"], capture_output=True, text=True).stdout
    tree = {"tree": [{"path": l.split("\t", 1)[1], "sha": l.split()[1]} for l in out.splitlines()]}
    sha = next(i["sha"] for i in tree["tree"] if i["path"] == EDITS_PATH)
    card_repo._edit_names = (sha, CardEdits.parse(open(os.path.join(CLONE, EDITS_PATH), encoding="utf-8").read()).names())
    card_repo._populate_site_ids = lambda: None          # offline
    card_repo.load_aliases()
    t = time.perf_counter(); card_repo.populate_files(tree); t_pop = time.perf_counter() - t
    return t_load, t_prep, t_pop

def queries():
    rnd = random.Random(7)
    abil = [a for a in card_repo.cards_dff["ability"] if a and len(a.split()) > 4]
    def phrase(k):
        w = rnd.choice(abil).split(); i = rnd.randrange(0, len(w) - k); return " ".join(w[i:i + k])
    ab = [phrase(rnd.choice([1, 2, 3])) for _ in range(60)]
    ab += ['"' + phrase(1) + '"' for _ in range(10)]
    ab += ["!" + phrase(1) for _ in range(5)]
    ab += ["hp>5", "def<=3", "atk>=7", "spd==4", "stars<3", "type==creature", "type~aux", "series~kirby", "subtype==attack", "hp>9"]
    ab += [f"{phrase(1)} | {phrase(1)}" for _ in range(5)] + [f"type==creature & {phrase(1)}" for _ in range(5)] + ["type==creature & hp>7"] * 5
    rb_abil = [a for a in card_repo.rulebook_dff["ability"] if a and len(a.split()) > 4]
    rb = []
    for _ in range(100):
        w = rnd.choice(rb_abil).split(); i = rnd.randrange(0, len(w) - 2); rb.append(" ".join(w[i:i + rnd.choice([1, 2])]))
    names = sorted(stats_db.stats)
    nm = []
    for i in range(100):
        n = rnd.choice(names).lower()
        if i % 10 < 5: nm.append(n)                                      # exact
        elif i % 10 < 8:                                                 # one typo
            j = rnd.randrange(len(n)); nm.append(n[:j] + n[j + 1:])
        else: nm.append(n.split()[0])                                    # first word
    return ab, rb, nm

def bench(fn, qs, reps=1):
    ts = []
    for q in qs:
        a = time.perf_counter()
        for _ in range(reps): fn(q)
        ts.append((time.perf_counter() - a) / reps)
    return ts

def show(label, ts):
    s = sorted(ts)
    print(f"{label:34s} 100 queries {sum(ts):7.3f}s  mean {statistics.mean(ts)*1e3:7.2f} ms  median {s[len(s)//2]*1e3:7.2f}  p95 {s[int(len(s)*.95)]*1e3:7.2f}  max {s[-1]*1e3:7.2f} ms")

if __name__ == "__main__":
    t_load, t_prep, t_pop = setup()
    print(f"stats_db.load {t_load*1e3:.0f} ms; prep_dataframes {t_prep*1e3:.0f} ms; populate_files(offline tree) {t_pop*1e3:.1f} ms; "
          f"cards {len(card_repo.cards_dff)}, rulebook {len(card_repo.rulebook_dff)}, registry names {len(card_repo.git_files)}")
    ab, rb, nm = queries()
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        import cProfile, pstats
        for label, fn, qs in [("ability", card_repo.ability_search_engine, ab), ("name", card_repo._find_closest_match, nm)]:
            pr = cProfile.Profile(); pr.enable(); [fn(q) for q in qs]; pr.disable()
            print("=====", label); pstats.Stats(pr).sort_stats("tottime").print_stats(12)
        sys.exit()
    bench(card_repo.ability_search_engine, ab[:5])
    show("ability_search_engine", bench(card_repo.ability_search_engine, ab))
    kinds = {"plain text (60)": ab[:60], "exact quoted (10)": ab[60:70], "NOT (5)": ab[70:75], "stat (10)": ab[75:85], "| and & (15)": ab[85:]}
    for k, qs in kinds.items():
        ts = bench(card_repo.ability_search_engine, qs); print(f"   {k:20s} mean {statistics.mean(ts)*1e3:7.2f} ms")
    show("rulebook_search_engine", bench(card_repo.rulebook_search_engine, rb, 5))
    show("_find_closest_match", bench(card_repo._find_closest_match, nm))
