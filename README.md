# AutoCroissant

Discord bot for the Impossibility Simulator card game. It parses the card PSDs of the TTSCardMaker repo into a card
database, answers queries about cards and rulebook pages, runs reminders and scheduled commands, plays music, keeps the
Tabletop Simulator save up to date, and builds the card data the card website (https://michaeljsr.github.io/TTSCardMaker/)
reads: TTSCardMaker's Card sync Action runs this repo's builder (`commands/card_data.py`) and commits the data to its
`data` branch, which the bot syncs.

## Run

`python main.py` with a `config.py` beside it (token, admins, GitHub token, ...; fields in
`.claude/skills/autocroissant-config-and-flags/SKILL.md`). Never commit `config.py`. Install: `pip install -r requirements.txt`,
then `python setup.py build_ext --inplace` (10 Cython modules; rebuild after every pull: it rebuilds only the modules whose source changed, `python setup.py clean` removes everything):
`.claude/skills/autocroissant-build-and-env/SKILL.md`.

## Test

`~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests` (offline; run before every push).

## Deploy

Production is the jetson (`ssh jetson`, systemd unit `autocroissant`): `scripts/deploy.sh` on the jetson (pull, psd-tools
pin, build, restart, journal check; prints OK or FAILED; `--bot ojwikibot` for the other bot; refuses 05:00-05:40 Pacific).
Runbook: `.claude/skills/autocroissant-run-and-operate/SKILL.md`.

## Docs

- `docs/` (index in `docs/README.md`): the session plan, bugs and improvements, campaign designs, finished-pass reports.
- `.claude/skills/`: 17 skills on how the bot works and is operated; start with `autocroissant-architecture-contract`.
- The card database is `db/` on TTSCardMaker's `data` branch (JSON, written by the Card sync Action only); the bot keeps a
  copy in `card_data/` (gitignored). The pickles it replaced (`stats.pkl`, `old_stats.pkl`, `aliases.pkl`) are in git history
  up to 2026-10-25.
