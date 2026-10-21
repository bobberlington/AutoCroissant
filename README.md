# AutoCroissant

Discord bot for the Impossibility Simulator card game. It parses the card PSDs of the TTSCardMaker repo into a card
database, answers queries about cards and rulebook pages, runs reminders and scheduled commands, plays music, keeps the
Tabletop Simulator save up to date, and feeds the card website (https://michaeljsr.github.io/TTSCardMaker/) through the
pickles it commits.

## Run

`python main.py` with a `config.py` beside it (token, admins, GitHub token, ...; fields in
`.claude/skills/autocroissant-config-and-flags/SKILL.md`). Never commit `config.py`. Install: `pip install -r requirements.txt`,
then `python setup.py build_ext --inplace` (11 Cython modules; rebuild after every pull; any `setup.py` run first deletes the built `.so`/`.c` files):
`.claude/skills/autocroissant-build-and-env/SKILL.md`.

## Test

`~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests` (offline; run before every push).

## Deploy

Production is the jetson (`ssh jetson`, systemd unit `autocroissant`): pull, rebuild, restart.
Runbook: `.claude/skills/autocroissant-run-and-operate/SKILL.md`.

## Docs

- `docs/` (index in `docs/README.md`): the session plan, bugs and improvements, campaign designs, finished-pass reports.
- `.claude/skills/`: 17 skills on how the bot works and is operated; start with `autocroissant-architecture-contract`.
- The pickles (`stats.pkl`, `old_stats.pkl`, `aliases.pkl`) are the card database, committed to git: never edit them by hand.
