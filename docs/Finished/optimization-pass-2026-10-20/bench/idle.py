"""Run main's three queue loops with no Discord connection and measure idle cost. Usage: idle.py <seconds> <n_reminders>"""
import os, sys, time, asyncio, resource, threading
ROOT = "/tmp/acprof/sb"; os.chdir(ROOT); sys.path.insert(0, ROOT)
secs, nrem = float(sys.argv[1]), int(sys.argv[2])
import main
import commands.analytics as an
from datetime import datetime, timedelta
an.init_reminder()
if nrem:
    future = datetime.now(an.TIMEZONE) + timedelta(days=3)
    sample = {"id": "x", "channel_id": 1, "when": future, "frequency": "", "msg": "hi", "command": "", "creator_id": 1,
              "version": an.REMINDER_VERSION}
    an.reminders[1] = [dict(sample, id=str(i)) for i in range(nrem)]
calls = {"n": 0, "t": 0.0}
orig = an.check_reminder
def timed():
    a = time.perf_counter(); orig(); calls["t"] += time.perf_counter() - a; calls["n"] += 1
main.check_reminder = timed
async def run():
    for lp in (main.process_command_queue, main.process_dispatch_queue, main.process_edit_queue): lp.start()
    await asyncio.sleep(1.5)
    r0 = resource.getrusage(resource.RUSAGE_SELF); w0 = time.perf_counter(); n0, t0 = calls["n"], calls["t"]
    await asyncio.sleep(secs)
    r1 = resource.getrusage(resource.RUSAGE_SELF); w1 = time.perf_counter()
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    n = calls["n"] - n0
    print(f"{nrem} reminders, {w1-w0:.0f}s idle: CPU {cpu*1e3/(w1-w0):.2f} ms per second ({100*cpu/(w1-w0):.3f}% of one core); "
          f"check_reminder ran {n}x, {1e6*(calls['t']-t0)/max(n,1):.0f} us each; threads alive {threading.active_count()}")
    for lp in (main.process_command_queue, main.process_dispatch_queue, main.process_edit_queue): lp.cancel()
asyncio.run(run())
