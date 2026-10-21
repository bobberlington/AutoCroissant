# Optimization audit (2026-10-20)

Read-only audits of AutoCroissant main at 7f6bd05 and the pass that followed: profile.md (queue hops, search, import time, RSS), network-robustness.md, network-speed-results.md, code-health.md, docs.md (refs_table.md is its skill line-ref table).
bench/: the profiling scripts behind profile.md; they expect the /tmp/acprof/sb sandbox (git archive HEAD plus config.py and the pickles).
The regression and line-ref scripts (parse_all_ns.py, compare.py, *_ns.py, refs6.py ...) live on main in .claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/.
Moved here from ~/Documents/ac-optimization on 2026-10-20; the numbers are summarised in docs/BUGS_AND_IMPROVEMENTS.md §2.5.
