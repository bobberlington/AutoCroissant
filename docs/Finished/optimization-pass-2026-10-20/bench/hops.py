"""Latency from queue_message() in a worker thread to the dispatch loop's send call (stub interaction, no Discord)."""
import os, sys, time, asyncio, random, threading, statistics
ROOT = "/tmp/acprof/sb"; os.chdir(ROOT); sys.path.insert(0, ROOT)
import main
from commands.utils import queue_message
lat = []
class Resp:
    def is_done(self): return False
    async def send_message(self, **kw): lat.append(time.perf_counter() - kw["content"])
class Stub:
    channel_id = 1
    response = Resp()
def producer(n):
    rnd = random.Random(3)
    for _ in range(n):
        time.sleep(rnd.uniform(0.05, 0.6)); queue_message(Stub(), time.perf_counter())
async def run():
    main.process_dispatch_queue.start()
    await asyncio.to_thread(producer, 60); await asyncio.sleep(1.2)
    main.process_dispatch_queue.cancel()
    s = sorted(lat)
    print(f"{len(lat)} messages: queue_message -> send call mean {statistics.mean(s)*1e3:.0f} ms, median {s[len(s)//2]*1e3:.0f}, p95 {s[int(.95*len(s))]*1e3:.0f}, max {s[-1]*1e3:.0f} ms")
asyncio.run(run())
