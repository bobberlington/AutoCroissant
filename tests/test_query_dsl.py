"""/query_ability's query language, and the website's (TTSCardMaker site/assets/search.js) on the same table: the same
cards, the same queries. Where the two differ on purpose (synonyms, field errors and other columns are the bot's so far), the row
says what the website finds."""
import json
import shutil
import subprocess
from os import environ
from os.path import expanduser
from pathlib import Path

import pytest

import commands.card_models as cm
import commands.query_card as qc
from conftest import CLONE

SEARCH_JS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "assets" / "search.js"

# website-shaped cards: name (as shown), aliases, ability, fields
CARDS = [
    dict(name="Kirby", type="creature", series="kirby", hp=5, defense=3, atk=4, spd=6, stars=3, types=["wind"],
         ability="Deal 2 DMG to the enemy."),
    dict(name="Meta Knight", type="creature", series="kirby", hp=7, defense=5, atk=7, spd=8, stars=4, aliases=["mk"],
         ability="Deals double damage. Die at the end of the turn."),
    dict(name="sans.", file="Sans", type="creature", series="undertale", hp=1, defense=1, atk=1, spd=9, stars=5,
         ability="Dodge every attack. +1 SPD."),
    dict(name="King Dedede", type="creature", series="kirby", hp=4, defense=4, atk=4, spd=2, stars=4, ability="Hammer time."),
    dict(name="Mettaton EX", type="creatures", hp=6, defense=6, atk=6, spd=6, ability="Gain +2 DEF."),
    dict(name="Fire Bomb", type="item", subtype="attack", stars=2, ability="Deal 3 damage to a [fire] creature."),
    dict(name="Panda Shield", type="item", subtype="equip", stars=1, aliases=["buckler"], ability="+1 defense."),
    dict(name="Bloon", type="minion", hp=2, defense=1, atk=1, spd=3, ability="Pop: heal 2 health."),
    dict(name="Burning", type="debuff", stars=1, ability="Burning: 3 DMG, can't affect [fire]."),
]
ERROR = "error"                                          # the bot answers with an error; the website finds nothing

QUERIES = [                                              # query, the bot's cards, the website's when it differs
    ("damage", {"Meta Knight", "Fire Bomb", "Kirby", "Burning"}, {"Meta Knight", "Fire Bomb"}),
    ("dmg", {"Kirby", "Burning", "Meta Knight", "Fire Bomb"}, {"Kirby", "Burning"}),
    ("defense", {"Panda Shield", "Mettaton EX"}, {"Panda Shield"}),
    ("health", {"Bloon"}, None), ("speed", {"sans."}, set()),
    ('"damage"', {"Meta Knight", "Fire Bomb"}, None),
    ("!type==creature", {"Mettaton EX", "Fire Bomb", "Panda Shield", "Bloon", "Burning"}, None),
    ("type==creature", {"Kirby", "Meta Knight", "sans.", "King Dedede"}, None),
    ("type~creature", {"Kirby", "Meta Knight", "sans.", "King Dedede", "Mettaton EX"}, None),
    ("hp>5", {"Meta Knight", "Mettaton EX"}, None), ("hp > 5", {"Meta Knight", "Mettaton EX"}, None),
    ("total>=24", {"Meta Knight", "Mettaton EX"}, None), ("spd>=8", {"Meta Knight", "sans."}, None),
    ("stars<=2", {"Fire Bomb", "Panda Shield", "Burning"}, None),
    ("fire", {"Fire Bomb", "Burning"}, None), ("dedede", {"King Dedede"}, None), ("mk", {"Meta Knight"}, None),
    ("sans.", {"sans."}, None), ("name~knight", {"Meta Knight"}, None), ("types~wind", {"Kirby"}, None),
    ("buckler", {"Panda Shield"}, None), ("alias~mk", {"Meta Knight"}, None), ("aliases==buckler", {"Panda Shield"}, None),
    ("display_name~sans", {"sans."}, set()),           # any column of the bot's (the website has no such field)
    ("die & series==kirby", {"Meta Knight"}, None), ("fire | hp>6", {"Fire Bomb", "Burning", "Meta Knight"}, None),
    ("!die", {c["name"] for c in CARDS} - {"Meta Knight"}, None),
    ("series==kirby & !hp>5", {"Kirby", "King Dedede"}, None), ("subtype==equip", {"Panda Shield"}, None),
    ("atack>3", ERROR, set()), ("total~5", ERROR, set()), ("hp>=x", ERROR, set()), ("type>3", ERROR, set()),
    ("type!=creature", ERROR, set()),
    ('"atack>3"', set(), None),
]


@pytest.fixture(scope="module")
def repo():
    stats = {}
    for c in CARDS:
        name = c.get("file", c["name"])
        stats[name] = cm.CardInfo(name=name, card_type=c["type"], path=f"X/{name}.psd", ability=c["ability"],
                                  stars=c.get("stars"), series=c.get("series"), subtype=c.get("subtype"),
                                  types=c.get("types", []), display_name=c["name"] if "file" in c else None,
                                  stats=cm.CardStats(c.get("hp", -1), c.get("defense", -1), c.get("atk", -1), c.get("spd", -1)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm.stats_db, "stats", stats)
        repo = qc.CardRepository()
        repo._edit_names = ("sha", {qc.CardAuthors.key(c.get("file", c["name"])): c["aliases"]   # as Card Edits.json
                                    for c in CARDS if c.get("aliases")})
        repo.prep_dataframes()
    return repo


def _shown(repo, names) -> set:
    titles = repo.cards_dff["display_name"] if "display_name" in repo.cards_dff else {}
    return {titles[n] if isinstance(titles.get(n), str) else n for n in names}


@pytest.mark.parametrize("query, bot, site", QUERIES)
def test_bot(repo, query, bot, site):
    if bot == ERROR:
        with pytest.raises(qc.QueryError):
            repo.ability_search_engine(query)
        return
    assert _shown(repo, repo.ability_search_engine(query).index) == bot


@pytest.mark.skipif(not shutil.which("node") or not SEARCH_JS.is_file(), reason=f"needs node and {SEARCH_JS}")
def test_website_on_the_same_table(tmp_path):
    cards = [{**{k: v for k, v in c.items() if k not in ("defense", "file")}, "def": c.get("defense")} for c in CARDS]
    script = tmp_path / "probe.mjs"
    script.write_text(f"const S = await import({json.dumps(SEARCH_JS.as_uri())});\n"
                      f"const cards = {json.dumps(cards)}, queries = {json.dumps([q for q, _, _ in QUERIES])};\n"
                      "console.log(JSON.stringify(queries.map(q => { const p = S.compile(q);"
                      " return cards.filter(c => p(c)).map(c => c.name); })));\n", encoding="utf-8")
    done = subprocess.run(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    found = dict(zip([q for q, _, _ in QUERIES], json.loads(done.stdout)))
    assert {q: set(found[q]) for q, _, _ in QUERIES} == {q: bot if site is None else site for q, bot, site in QUERIES}
