import sys, statistics; sys.argv=sys.argv[:1]; sys.path.insert(0,"/tmp/acprof")
import search_bench as sb
sb.setup(); ab, rb, nm = sb.queries()
c = [len(sb.card_repo.ability_search_engine(q)) for q in ab]
r = [len(sb.card_repo.rulebook_search_engine(q)) for q in rb]
for lab, x in (("ability", c), ("rulebook", r)):
    s = sorted(x); print(f"{lab}: results per query mean {statistics.mean(s):.1f} median {s[len(s)//2]} p90 {s[int(.9*len(s))]} max {s[-1]}; queries with >5 results {sum(v>5 for v in s)}")
