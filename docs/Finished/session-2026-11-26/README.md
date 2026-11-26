# Session 2026-11-26

Two owner asks: why the Bloons lost their ninja stars, and a TTS Workshop item that updates itself.

## 1. The Bloons' ninja stars

| What | Where | Notes |
|---|---|---|
| Cause | TC d4f2b911 (11-07, big pass 3/17), dddef197 (Prince of Darkness) | `TypesIndex.match` linked star/type row pictures by NAME alone, before any look. The Bloons' star layers were named `star` (`star.png`) but held a 512 px ninja star, so they became `Types/Stars/star.png` (gold) and were redrawn. Bytes and ink missed: `Ninja Star.png` is the same art at 147x150. |
| Scope | audit of every automated pass since 11-07: 3171 pairs, 914 relinks | Row name matches at look distance <= 14 were the same icon (older Wind, Equip, Fragile Equip, Crystal Star). The only different icons: the ninja stars on PoD01..PoD12 and Prince of Darkness (13 cards, distance 69, nearest `Ninja Star.png` at 1). |
| Fix | AC 74d08d09 | A name counts only with a look within `LOOK_NAMED`; else the closest-look Types file, else the picture kept as its own file; name alone only when a look is missing. `test_row_name_needs_a_close_look` fails on the old code. The guards clone AC main per push: live at once. Archaeology 10.45. |
| Cards | not restored | Owner: "no need to fix the bloon cards, but make sure this doesnt happen in the future." A rehearsed repair (pre-pass placement) is kept in `scratch/s1126/repair/repair.py`. |

## 2. TTS Workshop auto-update (LIVE)

| What | Where | Notes |
|---|---|---|
| Workflow | TC 779aefd `.github/workflows/workshop.yml` | After every Card sync run (workflow_run), 10-minute wait, newer runs cancel waiting ones; main's save always wins (re-checked daily). Dispatch modes upload / check / build. |
| Upload path | `.github/workshop/steam.mjs` | TTS's own legacy update, as the Mac's Steam logs and two disassemblies of steamclient.dylib show it: Cloud.BeginUGCUpload (timestamp, no is_public), PUT, CommitUGCUpload, EMsg 7325 `update_file`. steamcmd `workshop_build_item` converts legacy items to folder items TTS can't load: never use it. |
| File | `.github/scripts/workshop.py` | BSON as TTS writes it (floats as singles, BreakForce/BreakTorgue "Infinity" as doubles); byte-identical to the owner's 11-18 upload apart from TTS's physics drift. |
| Login | env `workshop` (main only): `STEAM_REFRESH_TOKEN`, `WORKSHOP_SECRETS_TOKEN`; var `WORKSHOP_ITEM_ID` | Minted by the owner with `mint.mjs` (QR), expires 2027-06-24; renewed by the run within 30 days of expiry after a write probe. |
| Alerts | `DISCORD_STANDARDS_WEBHOOK` | Ping `<@247149441734279169>`, each distinct alert once a UTC day. |
| Proof | runs 45821345302 (check), 45821462011 (upload) | The upload ran headless from GitHub: `result=updated`, 3,594,412 bytes, change note from the changelog. |
| Reviews | two adversarial rounds, mock harness `scratch/s1126/review2/run.mjs` (53 scenarios) | Round 2 kept 25 findings, all fixed (renewal after the upload with a write probe, alert dedup, step timeout, LA-month margin, .gitignore). |

## Owner rulings (memory `owner-decisions-2026-11-26`, `tts-workshop-research`)

- GitHub Actions route accepted with the Steam Subscriber Agreement risk on the owner's main account.
- Every save change; main's save always wins; no trial item; no backup before the first upload.
- SaveName stays `auto_update`; change note = changed cards; QR login; alerts pinging the owner.
- Workflows owner-only: impossible on a public user-owned repo (GitHub: "Source public repos cannot have push rules";
  probe, nothing created). Owner keeps the uploader and secrets in TTSCardMaker (trusts its collaborators).

## Cleanup

AC worktrees and `work/ninja-stars`, `work/wrap-1126` removed after landing; the TC sparse clone removed; scratch
`s1126` 1 GB -> 86 MB (kept: research notes, audit records, repair script, mock harness; its node_modules come back
with `npm ci` from TC's `.github/workshop/package-lock.json`). Disk 283 GB free.
