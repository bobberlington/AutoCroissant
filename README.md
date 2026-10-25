# AutoCroissant

Discord bot for the Impossibility Simulator card game. It parses the card PSDs of the TTSCardMaker repo into a card
database, answers queries about cards and rulebook pages, runs reminders and scheduled commands, plays music, keeps the
Tabletop Simulator save up to date, and feeds the card website (https://michaeljsr.github.io/TTSCardMaker/) through the
pickles it commits.

## Run

`python main.py` with a `config.py` beside it (token, admins, GitHub token, ...; fields in
`.claude/skills/autocroissant-config-and-flags/SKILL.md`). Never commit `config.py`. Install: `pip install -r requirements.txt`,
then `python setup.py build_ext --inplace` (11 Cython modules; rebuild after every pull: it rebuilds only the modules whose source changed, `python setup.py clean` removes everything):
`.claude/skills/autocroissant-build-and-env/SKILL.md`.

## Test

`~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests` (offline; run before every push).

## Deploy

Production is the jetson (`ssh jetson`, systemd unit `autocroissant`): the `jetson-deploy` skill (~/.claude/skills, outside the repo) on the jetson (pull, psd-tools
pin, build, restart, journal check; prints OK or FAILED; `--bot ojwikibot` for the other bot; refuses 05:00-05:40 Pacific).
Runbook: `.claude/skills/autocroissant-run-and-operate/SKILL.md`.

## Docs

- `docs/` (index in `docs/README.md`): the session plan, bugs and improvements, campaign designs, finished-pass reports.
- `.claude/skills/`: 17 skills on how the bot works and is operated; start with `autocroissant-architecture-contract`.
- The pickles (`stats.pkl`, `old_stats.pkl`, `aliases.pkl`) are the card database, committed to git: never edit them by hand.
