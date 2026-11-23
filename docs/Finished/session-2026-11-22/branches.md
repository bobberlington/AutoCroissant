# Branches and bundles (11-23)

Every ref of the session was recorded with its full SHA before it was deleted. Records (scratch
`~/Documents/Tools/wt/scratch/s1122/`):

- `wrap/branch_record.tsv`: wrapA-1122's cleanup (11-23 01:20-01:26), 49 rows: AC 23, TC 19 + the detached
  `fork-release-tc` worktree, fork 2, and 4 verdicts on the s1109 bundles. 44 branches and 33 worktrees deleted.
- `owner5/rn-land/branch_record.tsv`: `work/owner5-rn` (fork baa9a901 = the tag `cards-20261123`; AC 541feb9c = AC
  main), worktree `owner5-rn-tc` (detached 25077c28) removed.
- The land steps' own deletions: their LANDED lines in `BRIEF.md` and `ship2|ship3/<lane>/ship_record.tsv`;
  `wrap/typos2/land/branch_record.tsv` (`work/wrapb-typos` TC daca7927). `work/wrapb-forkdocs` = fork 1b970405 (landed).

| Verdict | Refs |
|---|---|
| Landed | The deferred five `s2/*` (AC 6, TC 6; tips in `scratch/s1110/bundles/s2-deferred-1121-{ac,tc}.bundle`). The lanes' `work/actbake-1122` (AC ee429674, TC 14cbdc62) and `work/cards-1122` (AC c60c64c1, TC 0cd15b61; only 1eab1edd, r7-3's original, isn't on main, on purpose). `work/fork-1122` (fork e5915747, AC 5e489de4). Every `ship/*`, `work/d1122-*`, `work/owner3-1122`, `work/owner5-*` and `work/wrapb-*` |
| On main, nothing of its own | AC `work/actbake-1122-r5-2` (f279dbe1), `work/cards-1122-r1-2` (ce4bae3a) |
| Rejected or not kept: bundled | `rejected-1122-ac.bundle` (13 refs) and `rejected-1122-tc.bundle` (13): actbake r1-2, r5-2 (TC), r7-0, r7-1, x0, x1; cards r1-2 (TC), r2-2, r3-2, r4-3, r6-0, r6-1, r7-2, r8-2 (AC), r8-3 (AC). Reasons: rejected.md |
| Fork lens rejects: bundled | `work/fork-1122-r3-0`, `r4-0` (fork + AC): `fork-1122-lens-{fork,ac}.bundle` |
| s1109 leftovers | `ac-actions-1108.bundle` and `tc-actions-1108.bundle` (all on main) and `fetchblobs.sh` deleted 11-23 01:26; `tc-unlanded-1110.bundle` deleted ~04:20 (owner 24; its heads `work/links-1107` 78ec3257, `work/stoogedings-linux` 9386f8fb in `wrap/branch_record.tsv.tc-unlanded-heads.txt`) |

## Bundles

`~/Documents/Tools/wt/scratch/s1122/bundles/`, listed in `INDEX.tsv` (34 rows; every bundle passes `git bundle
verify`; prerequisites are main commits): `rejected-1122-{ac,tc}`, `fork-1122-lens-{fork,ac}`,
`actbake-1122-final-{ac,tc}`, `cards-1122-final-{ac,tc}`. Restore in the repo INDEX names (ac: the AC repo, tc: a TC
work clone, fork: `~/Documents/Tools/psd-tools`): `git fetch <bundle> 'refs/heads/*:refs/heads/restored/*'`. Step
bundles (prepare, merge and ship snapshots) stay in their step folders, not indexed. Older:
`~/Documents/Tools/wt/scratch/s1110/bundles/`.

## Left

AC: main, 3 `archive/*`, and `work/wrap-1122` (these docs; deleted once landed). TC on GitHub: main, data, history,
plates, previews. Fork: main and 25 `archive/*` on GitHub, tags `cards-*` (newest `cards-20261123`). The final cleanup
(`tc-1122`, `~/std-work-1107`) and the state check after it are recorded in
`~/Documents/Tools/wt/scratch/s1122/wrap/final_state.md`.
