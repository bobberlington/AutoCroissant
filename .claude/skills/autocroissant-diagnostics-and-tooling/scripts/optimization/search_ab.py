"""Interleaved A/B in one process (cancels machine-load drift): old fuzzy step / name lookup vs the new ones, same
queries as search_ns.py, results asserted equal. Usage: search_ab.py ROOT (a sandbox with the new code)"""
import os, statistics, sys, time
from difflib import SequenceMatcher, get_close_matches
sys.argv = [sys.argv[0], sys.argv[1], "/dev/null"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_ns.py")).read().split("def run(")[0])
df = card_repo.cards_dff
def old_fuzzy(q):
    s = df["ability"].apply(lambda x: SequenceMatcher(None, q, x.lower()).ratio()); return list(s[s > card_repo.match_ratio].index)
def new_fuzzy(q):
    return list(card_repo._fuzzy_labels(q))
def old_name(q):
    m = get_close_matches(card_repo._normalize_card_name(q), card_repo.git_filenames, n=1, cutoff=card_repo.match_ratio); return m[0] if m else None
def best(f, q, reps=5):
    t = None
    for _ in range(reps):
        a = time.perf_counter(); r = f(q); d = time.perf_counter() - a; t = d if t is None else min(t, d)
    return r, t
for label, old, new, qs in (("fuzzy step, plain queries", old_fuzzy, new_fuzzy, plain),
                            ("fuzzy step, 60 short phrases (1-3 words)", old_fuzzy, new_fuzzy, plain[:60]),
                            ("_find_closest_match, all", old_name, card_repo._find_closest_match, nm),
                            ("_find_closest_match, exact names", old_name, card_repo._find_closest_match, [q for i, q in enumerate(nm[:200]) if i % 10 < 5])):
    to, tn = [], []
    for q in qs:
        ro, a = best(old, q); rn, b = best(new, q); assert ro == rn, q; to.append(a); tn.append(b)
    print(f"{label:42s} n={len(qs):3d}  old total {sum(to)*1e3:8.1f} ms (median {statistics.median(to)*1e3:6.2f})  "
          f"new total {sum(tn)*1e3:8.1f} ms (median {statistics.median(tn)*1e3:6.3f})  x{sum(to)/max(sum(tn),1e-9):.1f}  identical")
