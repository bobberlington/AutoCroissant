"""Measure candidate search speedups against the current code; asserts identical results."""
import sys, time, statistics
sys.argv = sys.argv[:1]
sys.path.insert(0, "/tmp/acprof")
import search_bench as sb
from difflib import SequenceMatcher, get_close_matches
from pandas import concat
sb.setup(); ab, rb, nm = sb.queries()
repo = sb.card_repo
orig = repo.ability_search_engine
df = repo.cards_dff

def fuzzy_current(ability):
    scores = df["ability"].apply(lambda x: SequenceMatcher(None, ability, x.lower()).ratio())
    return scores[scores > repo.match_ratio].index

lowered = [(idx, x.lower()) for idx, x in df["ability"].items()]
def fuzzy_bound(ability):           # V1: skip cards whose length alone caps the ratio below the cut
    la, r, keep = len(ability), repo.match_ratio, []
    for idx, x in lowered:
        lb = len(x)
        if la + lb and 2.0 * min(la, lb) / (la + lb) <= r:
            continue
        if SequenceMatcher(None, ability, x).ratio() > r:
            keep.append(idx)
    return keep

matchers = [(idx, SequenceMatcher(None, "", x)) for idx, x in lowered]
def fuzzy_cached(ability):          # V2: per-card matcher, b2j built once
    keep = []
    for idx, m in matchers:
        m.set_seq1(ability)
        if m.ratio() > repo.match_ratio: keep.append(idx)
    return keep

def fuzzy_both(ability):
    la, r, keep = len(ability), repo.match_ratio, []
    for idx, m in matchers:
        lb = len(m.b)
        if la + lb and 2.0 * min(la, lb) / (la + lb) <= r:
            continue
        m.set_seq1(ability)
        if m.ratio() > r: keep.append(idx)
    return keep

plain = ab[:60]
base = [list(fuzzy_current(q)) for q in plain]
for name, f in [("current (pandas apply)", fuzzy_current), ("V1 length bound", fuzzy_bound), ("V2 cached matchers", fuzzy_cached), ("V1+V2", fuzzy_both)]:
    res = [list(f(q)) for q in plain]
    assert res == base, name
    ts = sb.bench(f, plain)
    print(f"fuzzy part, {name:24s} 60 plain queries {sum(ts)*1e3:8.1f} ms  mean {statistics.mean(ts)*1e3:6.2f} ms  (identical results)")
print("fuzzy hits per plain query: mean", statistics.mean(len(b) for b in base), "max", max(len(b) for b in base))

# whole engine with V1+V2 swapped in for the fuzzy step
import commands.query_card as qc
def engine_fast(ability, _o=orig):
    ability = ability.strip()
    if "|" in ability or "&" in ability or repo._parse_stat_query(ability) is not None or ability.startswith("!") or (ability.startswith('"') and ability.endswith('"')):
        return _o(ability)
    exact = df[df["ability"].str.contains(qc.escape(ability), na=False)]
    fuzzy = df.loc[fuzzy_both(ability)]
    combined = concat([exact, fuzzy]); return combined[~combined.index.duplicated(keep='first')]
for q in plain: assert list(orig(q).index) == list(engine_fast(q).index)
print(f"engine on 60 plain queries: current {sum(sb.bench(orig, plain))*1e3:.0f} ms, with V1+V2 {sum(sb.bench(engine_fast, plain))*1e3:.0f} ms")

# name lookup
names = repo.git_filenames
def closest_fast(query):
    key = repo._normalize_card_name(query)
    if key in repo.git_files: return key
    m = get_close_matches(key, names, n=1, cutoff=repo.match_ratio); return m[0] if m else None
assert [repo._find_closest_match(q) for q in nm] == [closest_fast(q) for q in nm]
t0 = sb.bench(repo._find_closest_match, nm); t1 = sb.bench(closest_fast, nm)
print(f"_find_closest_match 100 queries: current {sum(t0)*1e3:.0f} ms, exact-key fast path + cached list {sum(t1)*1e3:.0f} ms (identical results)")
ex = [q for i, q in enumerate(nm) if i % 10 < 5]
print(f"   exact-name queries only (50): current {sum(sb.bench(repo._find_closest_match, ex))*1e3:.0f} ms, fast {sum(sb.bench(closest_fast, ex))*1e3:.1f} ms")
a = time.perf_counter(); [repo.git_filenames for _ in range(1000)]; print(f"   git_filenames list rebuild: {(time.perf_counter()-a):.3f} ms each")
