import sys, time; sys.argv=sys.argv[:1]; sys.path.insert(0,"/tmp/acprof")
import search_bench as sb
sb.setup(); ab, rb, nm = sb.queries()
from commands.query_card import query_ability, query_name, query_rulebook
from commands.utils import dispatch_queue, make_fake_interaction
fake = make_fake_interaction(1, 1)
for label, fn, qs in (("query_ability worker", query_ability, ab), ("query_name worker", query_name, nm), ("query_rulebook worker", query_rulebook, rb)):
    dispatch_queue.clear(); t = time.perf_counter()
    for q in qs: fn(fake, q)
    dt = time.perf_counter() - t
    print(f"{label:22s} 100 queries {dt:.2f}s ({dt*10:.1f} ms each); messages queued {len(dispatch_queue)} ({len(dispatch_queue)/100:.1f} per query)")
