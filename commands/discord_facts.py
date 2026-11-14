"""
Changelog facts for the card timelines (docs/Future/CARD_TIMELINE.md; owner answers Q4/Q5, 2026-11-13). Reads the
archive the miner stored (commands/discord_archive.py) and writes structured facts only, never a message's text:

    python -m commands.discord_facts build  [--root <dir>] [--cards-repo <dir>] [--no-first-repo-rows]
    python -m commands.discord_facts sample [--root <dir>] [--cards-repo <dir>] [--n 40] [--seed 1] [--sure]

build   Under the archive root's derived/facts/:
        facts.jsonl     a field change {card, date, channel, field, from, to, by, op} (hp, def, atk, spd, stars, type,
                        name) or {card, date, channel, kind: "ability changed"} (no text), from the update channels'
                        changelog lines (update-01..06, finished-updates; source "changelog": sure or unsure) and the
                        other card channels' stat-change texts (source "chat": evidence only, never an entry), each
                        with its line id (message id:line) as evidence; plus push evidence {date, author, cards} from
                        github-updates before the old repo (source "push"). A mark is sure only on the line's subject
                        (a 'Name:' prefix, the first words, after a leading verb, 'X's ability', 'add ... to X'), and
                        never on a card with no ability today; numbers beside a timing tag or 'ATK gain' wording are
                        the ability's. A change posted again folds into its earliest post ('also'); a later to/by
                        repeat that may be a second change keeps 'repeat_of'; one mark per card and message.
                        Stat values are bars (the card data's unit). Each stat fact has a unit: ATK/SPD 'bars'; HP/DEF
                        'bars', 'points' (game points, 2 HP or 0.5 DEF a bar: converted, the line's numbers kept in
                        'stated') or 'unknown' (the line's numbers; a sure fact becomes unsure). The unit comes from
                        the card's nearest picture reads (Discord pictures the matcher is sure of, the old repo's
                        rows), else from what bars can hold (whole, at most 10; raid bosses 20), else from the line
                        ('bars', a 4-number stat list): 'unit_how' read, range, said or list.
        firstrepo.jsonl text-only rows (date, mapped author, commit subject, badge "first repo") for push notices
                        naming exactly one card and nothing else (no list in the subject); shown while the switch is
                        on (switches.json: on, owner Q5 = A; --no-first-repo-rows turns it off).
        review.jsonl    changelog lines the owner decides (ambiguous card, unsure fact, with why): ids only.
sample  Prints n random lines that gave facts, with their facts, for an eye check (terminal only).
A card is named by today's names (stats.json keys, titles, display and website names, file names) or by an old name
through the old repo's sure renames (derived/oldgit/map.json) or the Discord-era names the timelines settled (the
Morgue's renamed entries, eye.json 'titled', the owner's owner.json old_names, some with a time limit); never through
an alias, and no alias is written. The 2019 balance-change lists posted as pictures (CHANGELOG_PICTURES) are read too:
OCR'd, each 'Name (Buff/Nerf/Rework)' entry a changelog line '<message>:img<n>' (no text kept).
Facts stop at the card repo's population (2023-11-05 23:15 PST); push notices at the old repo's first commit.
"""
from argparse import ArgumentParser
from collections import Counter
from copy import copy
from itertools import chain
from datetime import datetime, timezone
from hashlib import sha1
from json import dumps, loads
from os.path import basename, exists, expanduser, join, splitext
from random import Random
from re import IGNORECASE, compile as re_compile, finditer, fullmatch, match, sub
from typing import Iterable, Optional
from unicodedata import normalize

from commands.discord_match import (CATALOGUE, DEFAULT_ROOT, OLDGIT, check_root, git_dir_of, git_out, read_json,
                                    read_jsonl, slugify, write_json, write_text)
from global_config import CARD_BRANCH, DATA_BRANCH, EDITS_PATH, LOCAL_DIR_LOC

# ========================
# Configuration
# ========================
OUT = ("derived", "facts")
STATS_FILE, OLD_STATS_FILE = "db/stats.json", "db/old_stats.json"
TC_CUT = "2023-11-06T07:15:30+00:00"              # TTSCardMaker's last 'init' (36d62e53): git has it after this
OLD_START = "2023-04-21T08:30:28+00:00"            # the old repo's first commit: push notices before it only
UPDATE_IDS = {"692548660151582780", "737246900645920828", "745593281466990654", "765812180238598194",
              "910809664785088532", "935455829124067348", "1091208792575844444"}   # update-01..06, finished-updates
GITHUB_ID = "1011982154143633548"                  # github-updates
SKIP_BRANCHES = {"meme"}                           # the first repo's meme branch: never a live card's
CHAT_RELEVANCE = 0.1                               # other channels: the catalogue's card relevance at least this
CHAT_SKIP_KINDS = {"meme/joke", "bot", "update"}   # morgue channels never date a live card (owner Q1)
CHAT_LINES = 12                                    # longer chat messages are game reports and idea lists
AUTHORS = "~/Documents/Tools/wt/scratch/s1110/timelines/authors.json"
MORE_AUTHORS = {"Croissant2": "Croissant"}         # owner 10-30: the GitHub account Croissant2 is Croissant
SWITCH = "first_repo_rows"
DICT = "/usr/share/dict/words"
STAT = {"hp": "hp", "health": "hp", "def": "def", "defense": "def", "defence": "def", "atk": "atk",
        "attack": "atk", "spd": "spd", "speed": "spd"}
STAT_MAX, STARS_MAX = 40, 10
UP = {"buff", "buffed", "buffs", "increase", "increased", "increases", "raise", "raised", "raises", "up", "plus",
      "gain", "gains", "gained", "add", "added"}
DOWN = {"nerf", "nerfed", "nerfs", "reduce", "reduced", "reduces", "decrease", "decreased", "decreases", "lower",
        "lowered", "lowers", "down", "minus", "lose", "loses", "lost"}
FILLER = UP | DOWN | {"set", "change", "changed", "changes", "make", "made", "give", "gives", "given", "gave", "has",
                      "have", "had", "now", "base", "max", "maximum", "its", "his", "her", "their", "the", "to",
                      "from", "is", "be", "should", "slightly", "fix", "fixed", "a", "an", "will", "total", "starting",
                      "only", "just", "also", "it", "and", "bug", "properly", "so", "bars", "bar", "of", "stat",
                      "stats", "by", "with", "get", "gets", "back", "again", "start", "starts", "each", "both", "for"}
STAR_FILLER = FILLER | {"star", "stars", "item", "card", "creature", "minion", "debuff", "buff", "status", "effect",
                        "cost", "level", "rank", "not", "instead", "rather", "than", "field", "equip", "active"}
TYPE_FILLER = {"add", "added", "adds", "give", "gave", "gives", "remove", "removed", "removes", "type", "types",
               "and", "from", "to", "the", "a", "an", "now", "is", "be", "should", "instead", "of", "icon", "as",
               "second", "first", "it", "make", "made", "change", "changed", "not", "no", "longer", "replace",
               "replaced", "with", "card", "item", "also", "fix", "fixed", "only"}
NOT_TYPES = {"attack", "counter", "field", "buff", "debuff", "null", "unknown", "stars", "panda", "nme", "girl",
             "negation", "infliction"}          # icon names that are words here, not a type change
TYPE_WORDS = {"dark matter": "darkmatter", "fragile equip": "fragile equip"}
ABILITY = re_compile(
    r"\b(abilit(y|ies)|desc(ription)?s?|word(ing|ed)?|reword(ed|ing|s)?|rewr(i|o)te|rewritten|rework(ed|s)?|"
    r"overhaul(ed|s)?|text|clarif(y|ied|ies)|specif(y|ied|ies)|passive|effects?|damage|heal(s|ed|ing)?|"
    r"(\d+|one|two|three|four|five|of|more|less|fewer|extra) uses?|uses (per|left)|uses?\s*:?\s*\d|"
    r"counters?|inflict(s|ed)?|immun(e|ity)|summon(s|ed|ing)?|negat(e|es|ed|ion)|lifesteal|piercing|actions?|"
    r"turns?|clause|trigger(s|ed)?|cooldown|charges?|points?|spells?|costs?)\b|\((?:[a-z0-9/]{2,6})\)", IGNORECASE)
TAG = re_compile(r"\((\d*(?:/\d+)?(?!(?:HP|DEF|ATK|SPD)\))(?:[A-Z]{2,5}|opt|aat|eot|oyt|bot|endt|ofs|cd))\)")
TYPO_NOTE = re_compile(r"\((?:[^()]*\b(?:spelled|spelt|misspel\w*)\b[^()]*|\s*wrong\s+[^()\s]+\s*)\)", IGNORECASE)
PROTECTIVE = re_compile(r"\b(non[\s-]?)?protect(ive|ing)\s+minions?\b|\bdo(es)?\s*n[o']?t\s+protect\b", IGNORECASE)
EFFECT_WORDING = re_compile(r"\b(hp|def|atk|spd|health|defen[cs]e|attack|speed)\s+(gain|gains|loss|boost|bonus)\b",
                            IGNORECASE)                # '+1 ATK gain': an effect's amount, not the card's stat
PROSE = re_compile(r"\b(upon|when|whenever|after|while|until|each time|every|gain|gains|gained|lose|loses)\b",
                   IGNORECASE)                         # chat: stat words inside ability text
OBJECT_OF = re_compile(r"\b(summon\w*|generat\w*|give[sn]?|gave|giving|inflict\w*|specif\w*|spawn\w*|creat\w*)"
                       r"(\s+(a|an|the|one|two|three|\d+|more|another|its|his|her|their|them|turns?|of|x))*[\s\"']*$",
                       IGNORECASE)
DOUBT = {"field": "a change with no stated field: is it the ability?",
         "tag": "numbers beside a timing tag: the ability's numbers or the card's stats?",
         "position": "the card is named mid-sentence: is it the line's subject?",
         "minion": "the card has no ability today: is it the card meant?",
         "unit": "HP/DEF in points or bars?"}         # review page reasons (no line text)
SUBJECT_OK = {"prefix", "carried", "start", "lead", "owner", "target"}   # where a sure ability mark's card stands
OWNER = re_compile(r"^\x01?\s*(abilit(y|ies)|desc(ription)?s?|text|wording|passive)\b", IGNORECASE)
END = re_compile(r"^[\s.!:;]*(\([^()]*\)[\s.!]*)?$")
LOOKS = re_compile(r"\b(art|artwork|appearance|image|picture|background|font|border|icons?|colou?r\w*|visual|"
                   r"look(s)?|typos?|spelling|spelt|misspel\w*|format(ting)?|flavou?r|png|psd|template|layout|"
                   r"size|bold|italics?|spac(e|es|ing)|missing|capitali[sz]\w*|grammar|punctuation|attributes?)\b",
                   IGNORECASE)
NOT_A_CHANGE = re_compile(r"\b(remove[sd]?|delete[sd]?)\b.*\b(from the (deck|game))\b|\b(new card|add(ed)? (a|an|the)? "
                          r"?(new )?card)\b|^\s*(add|added|removed?)\s+\S+\s*$", IGNORECASE)
PROPOSAL = re_compile(r"\?|\b(maybe|perhaps|what if|how about|idea|suggest\w*|think|thoughts|could|would|might|"
                      r"probably|possibly|guess|proposal|propose|wonder|considering|if)\b", IGNORECASE)
RENAME_HINT = re_compile(r"\brenam|\btitle\b|\bname to\b|\bnow called\b|\"[^\"]+\"\s+(is now|->|to)\s+\"", IGNORECASE)
RATIONALE = re_compile(r"\s*(to|so|because|since|cause|bc|cuz|as|for)\b", IGNORECASE)
FORM = re_compile(r"\b(forms?|mecha|phase)\b", IGNORECASE)
SWAP = re_compile(r"\bswap(s|ped)?\b|\bswitch(ed)? (their|the)? ?(stats|atk|def|spd|hp)", IGNORECASE)
BLOCK_AFTER = {"minion", "minions", "soul", "souls", "shield", "shields", "totem", "totems", "clone", "clones",
               "tentacle", "tentacles", "spawn", "spawns", "deck", "decks", "type", "types", "series", "cards",
               "items", "family", "fight", "boss", "bosses", "army", "mk"}
VERBS = {"nerf", "buff", "make", "made", "change", "changed", "rework", "reworked", "fix", "fixed", "give", "remove",
         "removed", "add", "added", "replace", "update", "updated", "now", "increase", "reduce", "decrease", "lower",
         "raise", "should", "can", "cannot", "can't", "gains", "gain", "has", "is", "will", "only", "deals", "does",
         "no", "clarify", "specify", "overhaul", "rename", "renamed", "set", "and", "inflict", "inflicts", "heal",
         "heals", "loses", "takes", "summons", "costs", "gets", "becomes", "doesn't", "deal", "redo", "rewrite"}
PREFIX_OK = {"rework", "reworked", "nerf", "nerfed", "buff", "buffed", "description", "desc", "changed", "change",
             "changes", "ability", "stats", "stat", "fix", "fixed", "update", "updated", "overhaul", "overhauled",
             "slight", "minor", "the", "new", "again", "also", "and", "rebalance", "rebalanced", "tweak", "tweaks"}
START_OK = {"is", "should", "now", "has", "gets", "will", "to", "can", "cannot", "only", "no", "does", "deals",
            "gains", "loses"}
QUAD = re_compile(r"(?<![\d/.])(\d{1,2}(?:\.\d)?)\s?/\s?(\d{1,2}(?:\.\d)?)\s?/\s?(\d{1,2}(?:\.\d)?)\s?/\s?"
                  r"(\d{1,2}(?:\.\d)?)(?![\d/])")
QUAD_OK = {"summoned", "by", "non", "protective", "protecting", "all", "stats", "stat", "fix", "fixed", "to", "be",
           "should", "now", "has", "have", "is", "are", "the", "its", "new", "change", "changed", "set", "and", "with",
           "minion", "a", "an", "creature", "hp", "def", "atk", "spd", "of", "it"}
# the catalogue's stat-change pattern (a lower bound; prefilters chat messages)
STAT_RE = r"(?:hp|def|atk|spd|health|defen[cs]e|attack|speed)"
STAT_CHANGE = re_compile(
    rf"\b{STAT_RE}\b[^\n]{{0,25}}?\d+\s*(?:->|→|=>|>|to)\s*\d+|\d+\s*(?:->|→|=>)\s*\d+\s*{STAT_RE}\b"
    rf"|[+-]\d+\s*{STAT_RE}\b|\b{STAT_RE}\s*[+-]\d+"
    rf"|\b(?:reduce|raise|increase|decrease|lower|set)\w*\b[^\n]{{0,30}}\b{STAT_RE}\b[^\n]{{0,20}}\d"
    rf"|\b\d{{1,2}}\s*{STAT_RE}\b[^\n]{{0,6}}\d{{1,2}}\s*{STAT_RE}\b"
    r"|(?<![\d/])\d{1,2}/\d{1,2}/\d{1,2}/\d{1,2}(?![\d/])",
    IGNORECASE)
POSS = "\x01"                                     # a possessive ('s) in normalized text
POSSESSIVE_OK = set(STAT) | {"ability", "abilities", "description", "desc", "text", "stats", "stat", "wording",
                             "passive", "stars", "star", "max", "base", "first", "second", "third", "1st", "2nd", "3rd",
                             "card", "effect", "damage", "healing", "heal", "point", "points", "cost", "uses",
                             "counter", "counters", "title", "name", "type", "types"}
LEAD = {"nerf", "nerfed", "buff", "buffed", "make", "made", "change", "changed", "rework", "reworked", "fix", "fixed",
        "give", "gave", "update", "updated", "increase", "increased", "reduce", "reduced", "decrease", "decreased",
        "lower", "lowered", "raise", "raised", "overhaul", "overhauled", "clarify", "clarified", "specify", "specified",
        "reword", "reworded", "rewrite", "redo", "tweak", "tweaked", "rebalance", "rebalanced"}
LEAD_POS = LEAD | {"edit", "edited", "have", "add", "added", "remove", "removed", "give", "gave", "let", "allow"}
LINKS = {"that", "how", "for", "on", "the"}
LIST_SEP = re_compile(r",|\+|&|/|\band\b", IGNORECASE)
COMMIT = re_compile(r"^\[`([0-9a-f]{7,40})`\]\([^)]*\) (.*) - (\S+)$")
BAR_READS = ("derived", "readers", "bars", "reads.jsonl")   # picture bar reads (Discord and old repo)
VERDICTS = ("derived", "match", "verdicts.jsonl")         # each Discord picture's card (matcher)
OLD_ROWS = ("derived", "oldrows", "rows.jsonl")           # the old repo's versions, picture-checked stats
PER_BAR = {"hp": 2, "def": 0.5}                           # game points a bar (rulebook); ATK, SPD 1
BAR_CELLS, RAID_CELLS, RAID_STARS = 10, 20, 6             # bars a row holds (before 2026); raid bosses
CLUSTER = 2 * 86400                                       # reads this near the nearest one must agree with it
NEAR = 180 * 86400                                        # farther reads never decide a unit (unlogged changes)
BAR_WORD = re_compile(r"\bbars?\b", IGNORECASE)
MORGUE = ("derived", "morgue", "candidates.json")         # the Morgue's renamed entries: old names (evidence only)
MATCH_EYE = ("derived", "match", "eye.json")              # renamed first looks told by eye ('titled')
OWNER_FILE = ("derived", "review", "owner.json")          # the owner's rulings outside the review page (old_names)
CHANGELOG_PICTURES = ("325fb3259b34", "5ba4057218a4", "39ea31046541")   # 2019 balance changes posted as pictures
                                                          # (general-discussion; an OCR scan of every pre-git none, art,
                                                          # unsure and lost picture found no others: critic round 4)
OCR_BIN = ("derived", "readers", "text", "bin", "ocr")    # the text readers' Vision CLI (timeline_text.ocr_bin)
HEAD = re_compile(r"^(?P<head>[A-Za-z][^()]{0,48}?)\s*\(\s*(?P<tag>(?:buff|nerf|rework)[^)]*)\)\s*(?P<rest>.*)$",
                  IGNORECASE)                             # a changelog picture's entry: 'Name (Buff/Nerf) change'
GAP = 1.65                                                # a wrapped line sits within this many line heights
HEAD_REVIEW = {                                           # heads naming a card that isn't sure (by eye): review
    "Frozen in Time": "the item (Standstill, as the owner ruled the 2020 item) or the status Frozen In Time?",
    "Blockade": "today's The Wall By Ech (its 2023 picture still prints Blockade)? its HP change is a share"}
HEAD_ABILITY = {"Metal General": "its numbers are the summoned minion's"}   # by eye: the numbers are the ability's
BREAK = re_compile(r"(?<=\d)\s+(?=[A-Z][a-z])(?!(?:Health|Defen[cs]e|Attack|Speed|Bars?)\b)"
                   r"|(?:(?<=\bitem)|(?<=\bcreature))\s+(?=[A-Z])")   # a picture entry's next sentence (no stop)


# ========================
# Text
# ========================
def iso(stamp: str) -> str:
    return datetime.fromisoformat(stamp).astimezone(timezone.utc).isoformat(timespec="seconds")


def norm(text: str) -> str:
    """Accents folded, possessives marked (POSS), dots and apostrophes inside names joined; case kept."""
    t = str(text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = sub(r"[`*_~]{2,}|\|\|", " ", t)
    t = sub(r"(?<=[A-Za-z])'s\b", POSS, t)
    t = sub(r"(?<=[A-Za-z]s)'(?![A-Za-z])", POSS, t)
    t = sub(r"(?<=[A-Za-z])['](?=[A-Za-z])", "", t)
    t = sub(r"(?<=\b[A-Za-z])\.(?=[A-Za-z]\b|[A-Za-z]\.)", "", t)
    return sub(r"(?<=\b[A-Z])\.(?=\s|$)", "", t)


def clean(text: str) -> str:
    """A changelog line without struck-out text, mentions, emojis or links."""
    t = sub(r"~~.*?~~", " ", text or "")
    t = sub(r"<a?:\w+:\d+>|<[@#][!&]?\d+>|https?://\S+", " ", t)
    return t.replace("→", "->").replace("−", "-")


def squash(text: str) -> str:
    return "".join(ch for ch in norm(text).lower() if ch.isalnum())


def tokens(text: str) -> list:
    """(lower, original, start, end) of each word."""
    return [(m.group().lower(), m.group(), m.start(), m.end()) for m in finditer(r"[A-Za-z0-9]+", text)]


def short_id(*parts) -> str:
    return sha1(":".join(str(p) for p in parts).encode()).hexdigest()[:10]


def file_stem(path: str) -> str:
    return splitext(basename(path or ""))[0].replace("_", " ")


# ========================
# Names
# ========================
class Names:
    """Squashed card names -> {(key, how)}; how: name (today's names), old (the old repo's sure renames), archived."""

    def __init__(self, live: dict, old: Optional[dict] = None, archived: Iterable = (), with_stats: Iterable = (),
                 dictionary: Iterable = (), statuses: Iterable = (), no_ability: Iterable = (), raid: Iterable = (),
                 ids: Optional[dict] = None, internal: Iterable = ()):
        self.by, self.longest, self.names_of = {}, 1, {}
        self.with_stats, self.words, self.statuses = set(with_stats), set(dictionary), set(statuses)
        self.no_ability = set(no_ability)                 # cards with no ability text today (minions)
        self.raid, self.ids = set(raid), ids or {}        # raid bosses; website id -> key
        self.windowed, self._at = [], {}                  # old names meaning another card until a time
        for how, table in (("name", live), ("old", old or {})):
            for key, names in table.items():
                for name in names:
                    self.add(name, key, how)
        for key in archived:
            if squash(key) not in self.by:
                self.add(key, f"archived:{key}", "archived")
        for name, key, until in internal:                 # evidence only: never written as an alias
            if until:
                self.windowed.append((squash(name), name, key, until))
            elif len(squash(name)) >= 2 and squash(name) not in self.by:     # a name a card has: that card's
                self.add(name, key, "internal")
        for form, entries in list(self.by.items()):
            if len(form) >= 8 and any(len(tokens(norm(k.split(":")[-1]))) > 1 for k, _ in entries):
                other = form[:-1] if form.endswith("s") else form + "s"
                self.by.setdefault(other, set(entries))

    def at(self, when: str) -> "Names":
        """The names as a line posted at when (UTC ISO) read them: an old name with a time limit (owner 11-13:
        'Frozen in time' was the item, Standstill, until Standstill's first picture) names its card before then."""
        active = tuple(w for w in self.windowed if when < w[3])
        if not active:
            return self
        if active not in self._at:
            other = copy(self)
            other.by, other.names_of, other._at = dict(self.by), dict(self.names_of), {}
            for form, name, key, _ in active:
                other.by[form] = {(key, "internal")}
                other.names_of[key] = set(other.names_of.get(key, ())) | {name}
                other.longest = max(other.longest, len(tokens(norm(name))))
            self._at[active] = other
        return self._at[active]

    def add(self, name: str, key: str, how: str):
        form = squash(name)
        if len(form) >= 2:
            self.by.setdefault(form, set()).add((key, how))
            self.names_of.setdefault(key, set()).add(name)
            self.longest = max(self.longest, len(tokens(norm(name))))

    def risky(self, form: str, words: int) -> bool:
        """A one-word name that is also an English word, a status effect or very short: needs a capital to count."""
        return words == 1 and (len(form) <= 3 or form in self.words or form.rstrip("s") in self.words or
                               bool(self.by.get(form)) and all(k in self.statuses for k, _ in self.by[form]))

    def find(self, text: str, loose: bool = False) -> list:
        """Card names in normalized text, longest first, left to right: [{start, end, keys, how}]; loose: risky
        names count in any case."""
        toks, found, i = tokens(text), [], 0
        while i < len(toks):
            hit = None
            for n in range(min(self.longest + 2, len(toks) - i), 0, -1):
                gaps = [text[a[3]:b[2]] for a, b in zip(toks[i:i + n], toks[i + 1:i + n])]
                if any(not fullmatch(r"[\s.&'/\x01-]*", g) for g in gaps):
                    continue
                form = "".join(t[0] for t in toks[i:i + n])
                if form in self.by:
                    hit = (n, form)
                    break
            if not hit or (not loose and self.risky(hit[1], hit[0]) and not self.capital_ok(text, toks, i)):
                i += 1
                continue
            n, form = hit
            entries = self.by[form]
            found.append({"start": toks[i][2], "end": toks[i + n - 1][3], "keys": sorted({k for k, _ in entries}),
                          "how": sorted({h for _, h in entries})})
            i += n
        return found

    @staticmethod
    def capital_ok(text: str, toks: list, i: int) -> bool:
        """A risky name counts written with a capital (mid-sentence, or first and followed by ':', a stat change or a
        verb), or after a leading verb ('nerf blockade') or as a 'name:' prefix."""
        after = text[toks[i][3]:].lstrip()
        nxt = toks[i + 1][0] if i + 1 < len(toks) else ""
        if not text[:toks[i][2]].strip() and after[:1] == ":":
            return True
        if i == 1 and toks[0][0] in LEAD and not text[:toks[0][2]].strip() and (
                nxt in STAT or nxt in ("to", "by", "") or after[:1] in (":", POSS)):
            return True
        if not toks[i][1][:1].isupper():
            return False
        if text[:toks[i][2]].strip()[-1:] not in ("", ".", "|", ":", "!", "("):
            return True
        return after[:1] in (":", ",", "-", "(", "+") or nxt in START_OK or nxt in STAT or nxt.isdigit()


def load_names(cards_repo: str, root: str) -> Names:
    """Today's cards (the card repo's data branch, main's Card Edits names) and the old repo's sure renames."""
    git_dir = git_dir_of(cards_repo)
    raw = git_out(git_dir, "show", f"origin/{DATA_BRANCH}:{STATS_FILE}") if git_dir else None
    if raw is None:
        raise SystemExit(f"no card data in {cards_repo} (git fetch origin +{DATA_BRANCH}:refs/remotes/origin/"
                         f"{DATA_BRANCH})")
    stats = loads(raw).get("cards") or {}
    edits = loads(git_out(git_dir, "show", f"origin/{CARD_BRANCH}:{EDITS_PATH}") or b"{}").get("cards") or {}
    edits = {sub(r"\s+", " ", k.replace("_", " ")).strip().lower(): v for k, v in edits.items() if isinstance(v, dict)}
    old_raw = loads(git_out(git_dir, "show", f"origin/{DATA_BRANCH}:{OLD_STATS_FILE}") or b"{}")
    old_stats = old_raw.get("cards", old_raw) if isinstance(old_raw, dict) else {}
    live, with_stats, statuses, no_ability, raid = {}, set(), set(), set(), set()
    for key, card in stats.items():
        edit = edits.get(key.lower()) or {}
        title = (card.get("parsed_values") or {}).get("name")
        names = {key, card.get("name"), card.get("display_name"), title, edit.get("name")}
        if not card.get("variant_of") and not card.get("duplicate_of"):
            names.add(file_stem(card.get("path")))
        live[key] = [n for n in names if n]
        if ((card.get("stats") or {}).get("hp") or -1) >= 0:
            with_stats.add(key)
        if card.get("card_type") in ("buff", "debuff"):
            statuses.add(key)
        if not str(card.get("ability") or "").strip():
            no_ability.add(key)
        if (card.get("stars") or 0) >= RAID_STARS:
            raid.add(key)
    old = {}
    mapping = read_json(join(root, *OLDGIT, "map.json"), {}) or {}
    for old_key, entry in (mapping.get("cards") or {}).items():
        if entry.get("confidence") != "sure" or entry.get("status") not in ("live", "archived", "morgue"):
            continue
        key = entry["key"] if entry["status"] == "live" else f"{entry['status']}:{entry['key']}"
        names = {sub(r"\s*\([^)]*\)$", "", old_key), *(file_stem(p) for p in entry.get("paths") or [])}
        if (entry.get("read") or {}).get("title"):
            names.add(entry["read"]["title"])
        today = {squash(x) for x in live.get(key, [])}
        old.setdefault(key, []).extend(n for n in names if n and squash(n) not in today)
    archived = [k for k in old_stats if k not in stats]
    words = set()
    if exists(DICT):
        with open(DICT, encoding="utf-8", errors="ignore") as f:
            words = {w.strip().lower() for w in f if w.strip()}
    ids = card_ids(stats)
    return Names(live, old, archived, with_stats, words, statuses, no_ability, raid, ids, internal_names(root, ids))


def internal_names(root: str, ids: dict) -> list:
    """[(old name, card key, until)]: the Discord-era names the timelines settled (the Morgue's 'renamed' entries,
    renamed first looks told by eye: eye.json 'titled', the owner's identities: owner.json old_names; until: a time
    limit). Evidence only, never written as aliases."""
    found = []
    morgue = read_json(join(root, *MORGUE), {}) or {}
    for e in morgue.get("not_candidates") or []:
        if e.get("status") == "renamed" and ids.get(e.get("card")):
            found += [(n, ids[e["card"]], None) for n in {e.get("name"), e.get("title_read")} if n]
    for ruling in ((read_json(join(root, *MATCH_EYE), {}) or {}).get("pictures") or {}).values():
        if ruling.get("titled") and ids.get(ruling.get("card")):
            found.append((ruling["titled"], ids[ruling["card"]], None))
    for o in (read_json(join(root, *OWNER_FILE), {}) or {}).get("old_names") or []:
        if ids.get(o.get("card")):
            found.append((o["name"], ids[o["card"]], o.get("until")))
    return sorted(set(found), key=lambda f: (f[1], f[0], f[2] or ""))


def card_ids(stats: dict) -> dict:
    """{website id: card key}: discord_match.load_cards' ids (picture-only duplicates last, numbered on a clash)."""
    used, out = set(), {}
    for name in sorted(stats, key=lambda n: (bool(stats[n].get("duplicate_of") and not stats[n].get("variant_of")),
                                             n.lower())):
        base = found = slugify(name)
        n = 2
        while found in used:
            found, n = f"{base}-{n}", n + 1
        used.add(found)
        out[found] = name
    return out


# ========================
# Changelog lines
# ========================
def subject_of(text: str, spans: list) -> tuple:
    """(subject spans, where the body starts, prefix words, why not) of a line: its 'Card:' prefix, a list of
    cards, else its one card; names inside brackets are never the subject."""
    head = match(r"^\s*([^:]{1,70}?)\s*:\s*\S", text)
    if head and not match(r"^\s*[\d(]", head.group(1)):
        inside = [s for s in spans if s["end"] <= head.end(1)]
        rest = text[:head.end(1)]
        for s in reversed(inside):
            rest = rest[:s["start"]] + " " + rest[s["end"]:]
        extra = {w for w, *_ in tokens(rest)}
        if not inside:
            return [], 0, set(), "unknown prefix"
        if extra - PREFIX_OK - set(STAT) or len(extra & set(STAT)) > 1 or not is_list(text, inside):
            return [], 0, set(), "prefix names something else"
        return inside, head.end(1) + 1, extra, None
    depth, marks = 0, [0] * (len(text) + 1)
    for k, ch in enumerate(text):
        depth += ch == "("
        marks[k] = depth
        depth -= ch == ")" and depth > 0
    outside = [s for s in spans if not marks[s["start"]]]      # names inside brackets are objects
    if not outside:
        return [], 0, set(), "no card"
    lead = [w for w, *_ in tokens(text[:outside[0]["start"]])]
    if len(outside) > 1 and len(lead) == 1 and lead[0] in LEAD:
        run = [outside[0]]
        for s in outside[1:]:
            if not is_list(text, [run[-1], s]):
                break
            run.append(s)
        return run, 0, set(), None
    if len(outside) == 1 or is_list(text, outside):
        return outside, 0, set(), None
    return outside, 0, set(), "several cards"


def is_list(text: str, spans: list) -> bool:
    """Names only joined by commas, '&', 'and' or 'or'."""
    return all(sub(r"[\s,&\x01]|\band\b|\bor\b", "", text[a["end"]:b["start"]]) == "" for a, b in
               zip(spans, spans[1:]))


def subject_position(text: str, subject: list, cut: int) -> str:
    """Where the line's card stands: prefix ('Name:'), carried (a heading's), start, lead (after a leading verb),
    owner ('X's ability'), target ('add ... to X' ending the line), quoted, object (of summon, give, inflict...)
    or mid."""
    if cut:
        return "prefix"
    if not subject:
        return "carried"
    before = text[:subject[0]["start"]]
    if before.count('"') % 2 and before.strip() != '"':
        return "quoted"
    words = [w for w, *_ in tokens(before) if not w.isdigit()]
    if not words:
        return "start"
    verb = words[0] in LEAD_POS
    if verb and set(words[1:]) <= LINKS:
        return "lead"
    if OBJECT_OF.search(before):
        return "object"
    if OWNER.match(text[subject[-1]["end"]:]):
        return "owner"
    if verb and words[-1] in ("to", "on", "from", "with") and END.match(text[subject[-1]["end"]:]):
        return "target"
    return "mid"


def blocked(text: str, span: dict, names: Optional[Names] = None) -> bool:
    """A card name naming something of the card (its minion, soul, form) or part of a longer unknown name (or the
    start or end of a longer card's: 'dark matter swords', 'crowned magalor')."""
    rest = text[span["end"]:]
    if names and longer_before(text, span, names):
        return True
    if rest.startswith(POSS):
        nxt = tokens(rest)
        return bool(nxt) and rest[1:].lstrip()[:1] not in (":", ",", ";", "|", "(", ".", "!", "?") and \
            nxt[0][0] not in POSSESSIVE_OK and nxt[0][0] not in VERBS
    if rest.lstrip()[:1] in ("", ":", ",", ";", "|", "(", "-", "+", ".", "!", "?"):
        return False
    after = tokens(rest)
    if not after:
        return False
    low, orig = after[0][0], after[0][1]
    if low in BLOCK_AFTER:
        return True
    if names and rest[:after[0][2]].strip() == "" and low not in STAT and not low.isdigit():
        form = squash(text[span["start"]:span["end"]]) + low
        if any(f.startswith(form) for f in names.by):
            return True
    return orig[:1].isupper() and low not in STAT and low not in VERBS and not low.isdigit()


def longer_before(text: str, span: dict, names: Names) -> bool:
    """The word before the name starts another card's name ending like this card's ('crowned magalor' is Crowned
    Magolor, not Magolor by its old file name)."""
    before = tokens(text[:span["start"]])
    if not before or not fullmatch(r"[\s\x01-]*", text[before[-1][3]:span["start"]]):
        return False
    word = before[-1][0]
    if word in FILLER | VERBS | LEAD_POS | LINKS or word.isdigit():
        return False
    ends = {tokens(norm(n))[-1][0] for k in span["keys"] for n in names.names_of.get(k, ()) if tokens(norm(n))}
    return any(form.startswith(word) and any(form.endswith(e) and len(form) >= len(word) + len(e) for e in ends)
               and any(k not in span["keys"] for k, _ in entries) for form, entries in names.by.items())


def numbers(value: str):
    v = float(value.replace(" ", ""))
    return int(v) if v == int(v) else v


def clause_tokens(clause: str) -> list:
    out = []
    for m in finditer(r"->|(?<![\w.])(?:\+\s?|-)\d+(?:\.\d+)?|\d+(?:\.\d+)?|[a-z]+", clause.lower()):
        g = m.group()
        if g == "->":
            out.append(("arrow", g))
        elif g[0] in "+-":
            out.append(("signed", numbers(g)))
        elif g[0].isdigit():
            out.append(("num", numbers(g)))
        elif g in STAT:
            out.append(("stat", STAT[g]))
        elif g in ("star", "stars"):
            out.append(("star", g))
        else:
            out.append(("word", g))
    return out


def prepare(clause: str) -> str:
    """Arrows, 'from a to b' and 'to b from a' made 'a -> b'; 'N-star' made 'N star'."""
    c = clause.lower().replace("=>", "->").replace("-->", "->")
    c = sub(r"(\d)\s*-\s*star", r"\1 star", c)
    c = sub(r"(\d)\s*>\s*(\d)", r"\1 -> \2", c)
    c = sub(r"\bfrom\s+([+-]?\d+(?:\.\d+)?)\s*(?:->|to)\s*([+-]?\d+(?:\.\d+)?)", r"\1 -> \2", c)
    c = sub(r"\bto\s+(\d+(?:\.\d+)?)\s+from\s+(\d+(?:\.\d+)?)", r"\2 -> \1", c)
    return sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)", r"\1 -> \2", c)


def direction_of(words: list, default: int) -> int:
    d = default
    for w in words:
        d = 1 if w in UP else -1 if w in DOWN else d
    return d


def stat_facts(clause: str, direction: int) -> Optional[list]:
    """[(field, from, to, by)] when the clause is only stat changes, else None."""
    toks = clause_tokens(prepare(clause))
    words = [v for k, v in toks if k == "word"]
    if any(w not in FILLER for w in words) or not any(k == "stat" for k, _ in toks):
        return None
    d, by_word = direction_of(words, direction), "by" in words
    seq = [(k, v) for k, v in toks if k != "word"] + [("end", None)] * 3
    out, i = [], 0
    while seq[i][0] != "end":
        k, v = seq[i]
        a, b, c = seq[i + 1:i + 4]
        if k == "stat" and a[0] == "num" and b[0] == "arrow" and c[0] in ("num", "signed"):
            out.append((v, a[1], abs(c[1]), None))
            i += 4
        elif k == "stat" and a[0] == "arrow" and b[0] == "num":
            out.append((v, None, b[1], None))
            i += 3
        elif k == "stat" and a[0] == "signed":
            out.append((v, None, None, a[1]))
            i += 2
        elif k == "stat" and a[0] == "num":
            if by_word and not d:
                return None
            out.append((v, None, None, d * a[1]) if by_word else (v, None, a[1], None))
            i += 2
        elif k in ("num", "signed") and a[0] == "arrow" and b[0] in ("num", "signed") and c[0] == "stat":
            out.append((c[1], abs(v), abs(b[1]), None))
            i += 4
        elif k in ("num", "signed") and a[0] == "stat":
            out.append((a[1], None, None, v) if k == "signed" else (a[1], None, v, None))
            i += 2
        else:
            return None
    if len({f for f, *_ in out}) != len(out):
        return None
    for field, x, y, by in out:
        if any(n is not None and not 0 <= n <= STAT_MAX for n in (x, y)) or (by is not None and not 0 < abs(by) <= 20):
            return None
    return out


def star_facts(clause: str, direction: int) -> Optional[list]:
    toks = clause_tokens(prepare(clause))
    words = [v for k, v in toks if k == "word"]
    if not any(k == "star" for k, _ in toks) or any(k == "stat" for k, _ in toks) or \
            any(w not in STAR_FILLER for w in words):
        return None
    d = direction_of(words, direction)
    seq = [(k, v) for k, v in toks if k in ("num", "signed", "arrow")]
    shape = [k for k, _ in seq]
    if shape == ["num", "arrow", "num"]:
        x, y = seq[0][1], seq[2][1]
    elif shape == ["num"] and ("minus" in words or "by" in words):
        return [("stars", None, None, d * seq[0][1])] if d else None
    elif shape == ["num"]:
        x, y = None, seq[0][1]
    elif shape == ["signed"]:
        return [("stars", None, None, seq[0][1])]
    else:
        return None
    if not all(n is None or (isinstance(n, int) and 0 <= n <= STARS_MAX) for n in (x, y)):
        return None
    return [("stars", x, y, None)]


def type_facts(clause: str, vocabulary: set) -> Optional[list]:
    """[(add|remove, type)] when the clause is only a type change."""
    c = clause.lower()
    for words, one in TYPE_WORDS.items():
        c = c.replace(words, one.replace(" ", "_"))
    toks = [t.replace("_", " ") for t in sub(r"[^a-z_ ]", " ", c).split()]
    types = [t for t in toks if t in vocabulary]
    rest = {t for t in toks if t not in vocabulary}
    if not types or rest - TYPE_FILLER:
        return None
    if "instead" in rest and len(types) == 2:
        return [("add", types[0]), ("remove", types[1])]
    if rest & {"replace", "replaced"} and "with" in rest and len(types) == 2:
        return [("remove", types[0]), ("add", types[1])]
    if rest & {"remove", "removed", "removes"}:
        return [("remove", t) for t in types] if rest & {"type", "types", "icon"} else None
    if rest & {"not", "longer"}:
        return [("remove", t) for t in types] if "be" in rest or rest & {"type", "types"} else None
    if rest & {"add", "added", "adds", "give", "gave", "gives", "make", "made", "be", "now"}:
        return [("add", t) for t in types]
    return None


RENAME = [re_compile(r"\brenamed?\s+(?P<from>[^,;|()]+?)\s+(?:to|into|->)\s+(?P<to>[^,;|()]+)", IGNORECASE),
          re_compile(r"\b(?:renamed?|(?:change )?(?:the )?(?:title|name))\s+(?:it\s+)?(?:to|as|into)\s+"
                     r"(?P<to>[^,;|()]+)", IGNORECASE),
          re_compile(r"\bnow called\s+(?P<to>[^,;|()]+)", IGNORECASE),
          re_compile(r"\"(?P<from>[^\"]{2,40})\"\s+(?:is now|->|to)\s+\"(?P<to>[^\"]{2,40})\"", IGNORECASE)]
NOT_NAME = {"for", "some", "stuff", "change", "changes", "all", "every", "them", "it", "this", "that", "its", "their",
            "him", "her", "his", "be", "is", "was", "more", "less", "a"}


def name_like(text: Optional[str]) -> Optional[str]:
    t = sub(r"[\"'.!]+$", "", (text or "").replace(POSS, "'s").strip().strip("\"'")).strip()
    words = t.split()
    if not 1 <= len(words) <= 4 or any(w.lower() in NOT_NAME for w in words) or \
            not all(sub(r"[^\w&-]", "", w) for w in words):
        return None
    return t


def rename_facts(text: str, names: Names) -> Optional[tuple]:
    """(card key or None, from, to) from a rename in the line: one side must be a card's name."""
    for pattern in RENAME:
        m = pattern.search(text)
        if not m:
            continue
        to = name_like(sub(r"\s+(and|but|also|because|since|so)\b.*$", "", m.group("to")))
        frm = name_like(m.groupdict().get("from"))
        prefix = None
        if not frm and "from" not in pattern.groupindex:
            prefix = sub(r"^\W+|[\s:,-]+$", "", sub(r"[\s:,-]+(?:changed?|renamed?)\s*$", "", text[:m.start()],
                                                    flags=IGNORECASE))
            frm = name_like(prefix)
        if frm and prefix and squash(to or "") == squash(frm):
            frm = None                    # 'X: changed name to X': the line's subject is already the new name
        if not to or (frm and squash(to) == squash(frm)):
            return None
        hits = [{k for k, _ in names.by.get(squash(x or ""), set())}
                if x and (x[:1].isupper() or not names.risky(squash(x), len(x.split()))) else set() for x in (to, frm)]
        keys = hits[0] or hits[1]
        if not keys:
            return None
        # a side naming no card is kept only when it reads like a title (never a phrase of the line)
        frm, to = [x if found or (x and "'s" not in x and all(w[:1].isupper() or not w[:1].isalpha()
                                                                  for w in x.split())) else None
                   for x, found in ((frm, hits[1]), (to, hits[0]))]
        if not to:
            return None
        return (sorted(keys)[0] if len(keys) == 1 else None), frm, to
    return None


def split_clauses(text: str) -> list:
    t = sub(r"(?<!\d)\.(?!\d)|[;|\n,]|\b(?:but|that|which|also|and|then|plus|so)\b", "\x00", text, flags=IGNORECASE)
    return [c.strip() for c in t.split("\x00") if c.strip()]


def split_parts(text: str) -> list:
    """Sentences and comma parts (a timing tag marks its part)."""
    return [p for p in sub(r"(?<!\d)\.(?!\d)|[;|\n,]", "\x00", text).split("\x00") if p.strip()]


def words_of(text: str) -> list:
    return sub(r"[^a-z ]", " ", text.lower()).split()


def parse_line(raw: str, names: Names, vocabulary: set, source: str, carry: Optional[list] = None) -> dict:
    """One line: {facts: [{card, field|kind, from, to, by, op, sure}], unsure: [card keys], why, heading} (no text);
    carry: the subject of a heading line before it in the message (a line naming only a card)."""
    text = norm(clean(raw)).strip()
    out = {"facts": [], "unsure": [], "why": None, "heading": None}
    if not text or len(text) > 600:
        out["why"] = "empty" if not text else "too long"
        return out
    if source == "chat" and PROPOSAL.search(text):
        out["why"] = "proposal"
        return out
    if source == "chat" and (PROSE.search(text) or TAG.search(text)):
        out["why"] = "ability text"
        return out
    if RATIONALE.match(text):
        out["why"] = "a reason, not a change"
        return out
    renamed = RENAME_HINT.search(text) and rename_facts(text, names)
    if renamed:
        key, frm, to = renamed
        if key:
            out["facts"].append({"card": key, "field": "name", "from": frm, "to": to,
                                 "sure": bool(frm) and source == "changelog"})
        else:
            out["why"] = "rename of no known card"
        return out
    spans = names.find(text)
    subject, cut, prefix, why = subject_of(text, spans)
    if not why and any(blocked(text, s, names) for s in subject):
        why = "names something of a card"
    if not why and not cut and sub(r"[\W_]+", "", text[:subject[0]["start"]] + text[subject[-1]["end"]:]) == "" \
            and all(len(s["keys"]) == 1 for s in subject) and is_list(text, subject):
        out["heading"] = [s["keys"] for s in subject]
        out["why"] = "heading"
        return out
    if why == "no card" and carry:
        why, subject, cut, prefix, out["carried"] = None, [], 0, set(), True
    if why:
        out["why"] = why
        if why == "several cards" and source == "changelog" and (STAT_CHANGE.search(text) or ABILITY.search(text)):
            out["unsure"] = sorted({k for s in subject for k in s["keys"]})
        return out
    groups = [s["keys"] for s in subject] or carry
    out["position"] = position = subject_position(text, subject, cut)
    body = text[cut:] if cut else text
    for s in sorted(subject if not cut else [], key=lambda s: -s["start"]):
        body = body[:s["start"]] + " " + body[s["end"]:]
    body = sub(r"^[\s>]*[:,\-\u2013\u2014]+\s+", " ", body)
    if prefix & set(STAT):
        body = f"{next(iter(prefix & set(STAT)))} {body}"
    swap = bool(SWAP.search(body))
    form = bool(prefix & {"form", "forms"})
    ability = bool(prefix & {"rework", "reworked", "overhaul", "overhauled", "description", "desc", "ability"})
    loose, fields, tagged_stats = False, [], False
    body = sub(r"(\d+(?:\.\d+)?)\s*stars?,?\s*not\s*(\d+(?:\.\d+)?)", r"\2 -> \1 stars", body, flags=IGNORECASE)
    body = sub(r"(\d+(?:\.\d+)?)(\s*(?:stars?|hp|def|atk|spd)?)\s*\(\s*(?:should be|not)\s*(\d+(?:\.\d+)?)\s*\)",
               r"\1 -> \3\2", body, flags=IGNORECASE)
    quads = {m.groups() for m in QUAD.finditer(body)}
    if len(quads) == 1 and not form and not swap and not FORM.search(body):
        rest = sub(r"\bsummoned by [^,()]*", " ", QUAD.sub(" ", body), flags=IGNORECASE)
        for s in reversed(names.find(rest)):
            rest = rest[:s["start"]] + " " + rest[s["end"]:]
        if set(words_of(rest)) <= QUAD_OK:
            q = [numbers(x) for x in next(iter(quads))]
            fields += [(f, None, v, None, None, None, "?" in body, "list") for f, v in zip(("hp", "def", "atk", "spd"),
                                                                                        q)]
            body = ""
    body = TAG.sub(" \x02 ", TYPO_NOTE.sub(" typo ", body))     # a typo note skips its clause (LOOKS)
    body = sub(r"\([^()]*\)", " ", body)
    direction = direction_of([w for w in words_of(body)[:3]], 0)
    clauses = [(c, "\x02" in part) for part in split_parts(body) for c in split_clauses(part.replace("\x02", " "))]
    for clause, tagged in clauses:
        if swap or LOOKS.search(clause):
            continue
        target, asked = None, "?" in clause
        if len(groups) > 1:
            own = [s for s in names.find(clause) if s["keys"] in groups]
            if own and not words_of(clause[:own[0]["start"]]):
                target = own[0]["keys"]
                clause = clause[own[0]["end"]:]
        got = stat_facts(FORM.sub(" ", clause), direction)
        if got:
            if form or FORM.search(clause) or EFFECT_WORDING.search(clause):
                ability = True
            elif tagged:                                        # beside a timing tag: an effect's numbers
                loose = tagged_stats = True
            else:
                fields += [(*g, None, target, asked, "bars" if BAR_WORD.search(clause) else None) for g in got]
            continue
        got = star_facts(clause, direction)
        if got:
            fields += [(*g, None, target, asked, None) for g in got]
            continue
        got = type_facts(clause, vocabulary)
        if got:
            fields += [("type", None, t, None, op, target, asked, None) for op, t in got]
            continue
        if source != "changelog" or NOT_A_CHANGE.search(clause) or PROPOSAL.search(clause):
            continue
        if ABILITY.search(clause) and not asked:
            ability = True
        elif ABILITY.search(clause) or len([w for w in words_of(clause) if w not in FILLER]) >= 3:
            loose = True
    for field, frm, to, by, op, target, asked, said in fields:
        for ks in ([target] if target else groups):
            stat = field in ("hp", "def", "atk", "spd")
            if len(ks) > 1:
                ks = [k for k in ks if k in names.with_stats or k.startswith("archived:")] \
                    if stat and source == "changelog" else ks
                if len(ks) != 1:
                    out["unsure"] = sorted(set(out["unsure"]) | set(ks))
                    continue
            key = ks[0]
            if stat and key not in names.with_stats and not key.startswith(("archived:", "morgue:")):
                ability = ability or source == "changelog"
                continue
            fact = {"card": key, "field": field, "from": frm, "to": to, "by": by, "op": op, "sure": not asked,
                    "said": said}
            out["facts"].append(fact)
    if PROTECTIVE.search(text):                                  # (non) protective minion: not the ability
        ability = loose = False
    if source == "changelog" and (ability or loose):
        for ks in groups:
            if len(ks) == 1:
                key = ks[0]
                doubt = DOUBT["tag"] if tagged_stats and not ability else DOUBT["field"] if not ability else \
                    DOUBT["position"] if position not in SUBJECT_OK or key in names.statuses and \
                    position not in ("prefix", "carried", "start") else DOUBT["minion"] if key in names.no_ability \
                    else None
                out["facts"].append({"card": key, "kind": "ability changed", "sure": not doubt, "doubt": doubt})
            else:
                out["unsure"] = sorted(set(out["unsure"]) | set(ks))
    if not out["facts"] and not out["unsure"]:
        out["why"] = "no change stated"
    return out


# ========================
# Push notices
# ========================
def notice_commits(message: dict) -> list:
    """[(sha, subject, author, branch)] of a github-updates push notice (not force pushes, not the meme branch)."""
    found = []
    for embed in message.get("embeds") or []:
        head = match(r"\[[^:\]]+:(?P<branch>[^\]]+)\]", embed.get("title") or "")
        branch = sub(r"\\(.)", r"\1", head.group("branch")) if head else None
        if "force-pushed" in (embed.get("title") or "") or branch in SKIP_BRANCHES:
            continue
        for line in (embed.get("description") or "").split("\n"):
            m = COMMIT.match(line.strip())
            if m:
                found.append((*m.groups(), branch))
    return found


def subject_text(subject: str) -> str:
    return norm(sub(r"\.(png|psd|jpe?g|webp)\b", " ", subject.replace("_", " "), flags=IGNORECASE)).strip()


def commit_cards(subject: str, names: Names) -> list:
    """Today's cards a commit subject names (file names, titles, old names); a subject that is only a name counts
    in any case."""
    text = subject_text(subject)
    whole = {k for k, _ in names.by.get(squash(text), set())}
    found = whole if len(whole) == 1 else {s["keys"][0] for s in names.find(text) if len(s["keys"]) == 1
                                           and not blocked(text, s, names)}
    return sorted(k for k in found if not k.startswith(("archived:", "morgue:")))


def only_card(subject: str, card: str, names: Names) -> bool:
    """A subject about this one card: no list (and , + & /) unless every item is this card, and no other card's
    name in any case."""
    items = [x for x in LIST_SEP.split(subject) if x.strip()]
    if len(items) > 1 and any(commit_cards(x, names) != [card] for x in items):
        return False
    return all(s["keys"] == [card] for s in names.find(subject_text(subject), loose=True))


# ========================
# Archive
# ========================
def messages(root: str, channel: str) -> Iterable[dict]:
    path = join(root, "messages", f"{channel}.jsonl")
    if not exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.endswith("\n"):
                yield loads(line)


def chat_channels(root: str) -> dict:
    """{channel id: (name, kind)} of the other card channels (catalogue relevance; morgue and bot channels left
    out)."""
    catalogue = read_json(join(root, *CATALOGUE), {}) or {}
    return {c["id"]: (c["name"], c.get("kind")) for c in catalogue.get("channels") or []
            if c.get("kind") not in CHAT_SKIP_KINDS and (c.get("card_relevance") or 0) >= CHAT_RELEVANCE
            and c["id"] not in UPDATE_IDS and c["id"] != GITHUB_ID}


def human(message: dict) -> bool:
    return not ((message.get("author") or {}).get("bot") or message.get("webhook_id"))


def changelog_lines(root: str, names: Names, vocabulary: set, chat: bool = True) -> Iterable[tuple]:
    """(source, channel id, channel name, message, line number, line, parse) before the card repo, humans only."""
    channels = [(c, "changelog") for c in sorted(UPDATE_IDS)]
    titles = {}
    if chat:
        titles = chat_channels(root)
        channels += [(c, "chat") for c in sorted(titles)]
    for channel, source in channels:
        for m in messages(root, channel):
            if iso(m["timestamp"]) >= TC_CUT or not human(m) or not (m.get("content") or "").strip():
                continue
            if source == "chat" and (not STAT_CHANGE.search(m["content"]) or
                                     len(m["content"].strip().split("\n")) > CHAT_LINES):
                continue
            carry, at = None, names.at(iso(m["timestamp"]))
            for n, line in enumerate(m["content"].split("\n")):
                if not line.strip():
                    continue
                got = parse_line(line, at, vocabulary, source, carry)
                carry = got["heading"] or (carry if got.get("carried") or got["why"] == "no card" else None)
                yield source, channel, m.get("channel_name") or titles.get(channel, (channel,))[0], m, n, line, got


def picture_entries(lines: list) -> list:
    """[(head, tag, text)]: a changelog picture's OCR lines ([text, conf, x, y, w, h, ...], top down) as its entries:
    each 'Name (Buff|Nerf|Rework) change' head with the lines wrapped under it (a line within GAP line heights of the
    one before, or starting in lower case or a digit); any other line (a title, 'Items') ends the entry."""
    out, prev, open_ = [], None, False
    for text, _conf, _x, y, _w, h, *_ in sorted(lines, key=lambda line: line[3]):
        text = text.strip()
        m = HEAD.match(text)
        if m:
            out.append([m["head"].strip(), m["tag"].strip(), m["rest"].strip()])
            open_ = True
        elif open_ and prev and (y - prev[0] < GAP * prev[1] or text[:1].islower() or text[:1].isdigit()):
            out[-1][2] = f"{out[-1][2]} {text}".strip()
        else:
            open_ = False
        prev = (y, h)
    return [tuple(e) for e in out]


def picture_line(head: str, rest: str, names: Names, base: Names, vocabulary: set) -> dict:
    """One entry of a changelog picture: parsed as 'Head: change' when the head is exactly one card's name (today's,
    an old one) and not a head the eye doubts (HEAD_REVIEW); else review (the cards it may name). The head's tag
    states a change, so a change no stated field explains is the ability's."""
    form = squash(head)
    keys = sorted({k for k, _ in names.by.get(form, set())})
    doubt = {squash(h): why for h, why in HEAD_REVIEW.items()}.get(form)
    if len(keys) != 1 or doubt:
        near = keys + sorted({k for n in (names, base) for f, e in n.by.items() if len(form) >= 4 and
                              (f.startswith(form) or form.startswith(f) and len(f) >= 4) for k, _ in e} - set(keys))
        return {"facts": [], "unsure": near[:4], "why": "picture head", "heading": None,
                "head": f"a changelog picture's head '{head}': {doubt or 'no card has this name'}"}
    key, only = keys[0], {squash(h): why for h, why in HEAD_ABILITY.items()}.get(form)
    got = parse_line(f"{head}: {BREAK.sub('; ', rest)}", names, vocabulary, "changelog")
    if only:                                      # by eye: the numbers are the ability's
        got["facts"] = [f for f in got["facts"] if f.get("kind") == "ability changed"]
    for f in got["facts"]:
        if f.get("doubt") == DOUBT["field"]:
            f.update(doubt=None, sure=True)
    marked = any(f.get("kind") == "ability changed" for f in got["facts"])
    if not got["unsure"] and not marked and (not got["facts"] or not only and unexplained(rest, vocabulary)):
        minion = key in names.no_ability          # the tag says it changed: what no field states is the ability's
        got.update(why=None, facts=got["facts"] + [{"card": key, "kind": "ability changed", "sure": not minion,
                                                    "doubt": DOUBT["minion"] if minion else None}])
    return got


def unexplained(rest: str, vocabulary: set) -> bool:
    """A picture entry has a clause no stated field (stats, stars, types) explains."""
    body = sub(r"\([^()]*\)", " ", BREAK.sub("; ", rest))
    return any(not (stat_facts(FORM.sub(" ", c), 0) or star_facts(c, 0) or type_facts(c, vocabulary))
               and any(w not in FILLER for w in words_of(c)) for part in split_parts(body) for c in split_clauses(part))


def picture_lines(root: str, names: Names, vocabulary: set, counts: Counter) -> Iterable[tuple]:
    """(source, channel id, channel name, message, 'img<n>', entry text, parse) of the changelog pictures
    (CHANGELOG_PICTURES), OCR'd with the text readers' Vision CLI (no text stored); none without it."""
    import subprocess
    tool = join(root, *OCR_BIN)
    if not exists(tool):
        counts["changelog pictures: no OCR tool"] += 1
        return
    files = {}
    for a in read_jsonl(join(root, "attachments.jsonl")):
        if str(a.get("sha256", ""))[:12] in CHANGELOG_PICTURES and a.get("file"):
            files.setdefault(a["sha256"][:12], a)
    wanted = {a["message_id"]: a["container_id"] for a in files.values()}
    found = {}
    for channel in sorted(set(wanted.values())):
        for m in messages(root, channel):
            if m["id"] in wanted:
                found[m["id"]] = m
    p = subprocess.Popen([tool], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        for pid in CHANGELOG_PICTURES:
            a = files.get(pid)
            m = found.get((a or {}).get("message_id"))
            if not a or not m:
                counts["changelog pictures: not in the archive"] += 1
                continue
            p.stdin.write(dumps({"path": join(root, a["file"]), "scale": 1.0, "fix": False}) + "\n")
            p.stdin.flush()
            lines = loads(p.stdout.readline()).get("lines") or []
            counts["changelog pictures"] += 1
            at = names.at(iso(m["timestamp"]))
            for n, (head, _tag, rest) in enumerate(picture_entries(lines)):
                yield ("changelog", a["container_id"], m.get("channel_name") or a["container_id"], m, f"img{n}",
                       f"{head}: {rest}", picture_line(head, rest, at, names, vocabulary))
    finally:
        p.stdin.close()
        p.wait()


def line_order(line_id: str) -> int:
    """A line id's place in its message ('<message>:<n>', a picture's '<message>:img<n>')."""
    part = line_id.split(":")[1]
    return int(part[3:]) if part.startswith("img") else int(part)


def build(root: str, names: Names, vocabulary: set, first_repo_rows: bool = True, log=print) -> dict:
    out = join(root, *OUT)
    facts, review, counts = [], [], Counter()
    kinds = {c: kind for c, (_, kind) in chat_channels(root).items()}
    messages_seen = set()
    seen = set()
    lines = chain(changelog_lines(root, names, vocabulary), picture_lines(root, names, vocabulary, counts))
    for source, channel, cname, m, n, line, got in lines:
        counts[f"{source} lines"] += 1
        if m["id"] not in messages_seen:
            messages_seen.add(m["id"])
            counts[f"{source} messages"] += 1
        line_id = f"{m['id']}:{n}"
        if got["facts"]:
            counts[f"{source} lines with facts"] += 1
        for f in got["facts"]:
            sure, doubt = f.pop("sure"), f.pop("doubt", None)
            confidence = "evidence" if source == "chat" else "sure" if sure else "unsure"
            fact = {"id": "f" + short_id(line_id, f.get("card"), f.get("field") or f.get("kind"), f.get("to"),
                                         f.get("op"), f.get("by")),
                    "card": f["card"], "date": iso(m["timestamp"]), "channel": cname, "channel_id": channel,
                    "source": source, "message": m["id"], "line": line_id,
                    **{k: v for k, v in f.items() if k != "card" and v is not None}, "confidence": confidence}
            if source == "chat":
                fact["channel_kind"] = kinds.get(channel)
            if f["card"] and f["card"].startswith(("archived:", "morgue:")):
                fact["status"] = f["card"].split(":")[0]
            if fact["id"] in seen:
                continue
            seen.add(fact["id"])
            facts.append({**fact, "_doubt": doubt} if doubt and confidence == "unsure" else fact)
        if (got["unsure"] or got.get("head")) and source == "changelog":
            review.append({"id": "R" + short_id(line_id, "cards"), "kind": "ambiguous card",
                           "candidates": got["unsure"],
                           "date": iso(m["timestamp"]), "channel": cname, "channel_id": channel, "message": m["id"],
                           "line": line_id, "why": got.get("head") or
                           "the line names several cards (or a name two cards had)"})
            counts["changelog lines with ambiguous cards"] += 1
        if got["why"]:
            counts[f"{source} why {got['why']}"] += 1
    stat_units(facts, load_reads(root, names.ids), names.raid, counts)
    facts = dedupe(facts, counts)
    doubts = {f["id"]: f.pop("_doubt") for f in facts if "_doubt" in f}
    for fact in facts:
        counts[f"{fact['source']} facts {fact['confidence']}"] += 1
        counts[f"kind {fact.get('field') or fact.get('kind')}"] += 1
        if fact["confidence"] == "unsure":
            review.append({"id": "R" + short_id(fact["line"], fact["id"]), "kind": "unsure fact", "fact": fact["id"],
                           "card": fact["card"], "date": fact["date"], "channel": fact["channel"],
                           "message": fact["message"], "line": fact["line"], **({"also": fact["also"]}
                                                                               if fact.get("also") else {}),
                           "why": doubts.get(fact["id"]) or ("a rename with no old name" if fact.get("field") == "name"
                                                             else "asked as a question" if fact.get("field")
                                                             else DOUBT["field"])})
    review.sort(key=lambda r: (r["date"], r["line"]))
    pushes, rows = push_facts(root, names, counts)
    facts += pushes
    switches = {SWITCH: bool(first_repo_rows),
                "why": "owner Q5 = A (2026-11-13): text-only rows for notices naming exactly one card, shown in "
                       "the preview; the other notices are evidence only",
                "owner_answers": ["Q4 = B", "Q5 = A"]}
    for row in rows:
        row["shown"] = bool(first_repo_rows)
    write_text(join(out, "facts.jsonl"), "".join(dumps(f, ensure_ascii=False) + "\n" for f in facts))
    write_text(join(out, "firstrepo.jsonl"), "".join(dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_text(join(out, "review.jsonl"), "".join(dumps(r, ensure_ascii=False) + "\n" for r in review))
    write_json(join(out, "switches.json"), switches)
    summary = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "facts": len(facts),
               "first_repo_rows": len(rows), "review": len(review), "counts": dict(sorted(counts.items())),
               "cards": len({f["card"] for f in facts if f.get("card")} |
                            {c for f in facts for c in f.get("cards", [])}),
               "cards_with_sure_facts": len({f["card"] for f in facts if f.get("confidence") == "sure"})}
    write_json(join(out, "summary.json"), summary)
    log(f"facts {len(facts)}, first repo rows {len(rows)} (shown: {bool(first_repo_rows)}), review {len(review)}")
    return summary


def dedupe(facts: list, counts: Counter) -> list:
    """A change posted again folds into its earliest post ('also': the other line ids) when on the same day, or when
    it is a full change (from and to, or a type add/remove) with nothing else on that field between; a later
    to-only or by-only repeat may be a second change: kept, with 'repeat_of'. One ability mark per card and message
    (a sure one first). Sorted by date."""
    rank = {"sure": 0, "unsure": 1, "evidence": 2}
    facts = sorted(facts, key=lambda f: (f["date"], f["message"], line_order(f["line"])))
    marks = {}
    for f in facts:
        if f.get("kind") == "ability changed":
            marks.setdefault((f["source"], f["card"], f["message"]), []).append(f)
    best = {id(min(group, key=lambda f: rank[f["confidence"]])): group for group in marks.values()}
    out, last = [], {}
    for f in facts:
        if f.get("kind") == "ability changed":
            group = best.get(id(f))
            if group is None:
                counts["repeats folded (ability marks)"] += 1
                continue
            also = [g["line"] for g in group if g is not f and g["line"] != f["line"]]
            if also:
                f["also"] = sorted(set(also))
            out.append(f)
            continue
        field = (f["source"], f["card"], f["field"], f["to"] if f["field"] == "type" else None)
        prev = last.get(field)
        if prev and all(prev.get(k) == f.get(k) for k in ("from", "to", "by", "op", "unit")):
            if prev["date"][:10] == f["date"][:10] or f["field"] == "type" or \
                    prev.get("from") is not None and prev.get("to") is not None:
                prev.setdefault("also", []).append(f["line"])
                if rank[f["confidence"]] < rank[prev["confidence"]]:
                    prev["confidence"] = f["confidence"]
                counts["repeats folded (field facts)"] += 1
                continue
            f["repeat_of"] = prev["id"]
            counts["repeats kept (maybe a second change)"] += 1
        last[field] = f
        out.append(f)
    return out


# ========================
# Units (bars or game points)
# ========================
def stamp(date: str) -> float:
    return datetime.fromisoformat(date).astimezone(timezone.utc).timestamp()


def load_reads(root: str, ids: dict) -> dict:
    """{card key: [(time, {hp, def})]} oldest first: sure bar reads of the Discord pictures whose card the matcher is
    sure of (a git look or a draft; dated by the first post) and the old repo's picture-checked stats. Two-creature
    pictures left out."""
    verdicts, reads = {}, {}
    for v in read_jsonl(join(root, *VERDICTS)):
        if v.get("verdict") in ("version_sure", "version_unsure", "draft") and not v.get("joke") and \
                v.get("card") in ids and (v.get("first") or {}).get("date"):
            verdicts[v["file"]] = (ids[v["card"]], v["first"]["date"])
    for r in read_jsonl(join(root, *BAR_READS)):
        if r.get("src") == "discord" and r.get("file") in verdicts and not r.get("stats2"):
            key, date = verdicts[r["file"]]
            vals = {f: s["v"] for f, s in (r.get("stats") or {}).items()
                    if f in PER_BAR and isinstance(s, dict) and s.get("sure") and s.get("v") is not None}
            if vals:
                reads.setdefault(key, []).append((stamp(date), vals))
    for r in read_jsonl(join(root, *OLD_ROWS)):
        got = (r.get("fields") or {}).get("stats") or {}
        vals = {f: v for f, v in (got.get("value") or {}).items() if f in PER_BAR and v is not None}
        if got.get("confidence") == "sure" and vals and r.get("card") and r.get("date"):
            reads.setdefault(r["card"], []).append((stamp(r["date"]), vals))
    for rows in reads.values():
        rows.sort(key=lambda x: x[0])
    return reads


def nearest(reads: list, field: str):
    """The nearest read's value of field (reads nearest first), None when none or the reads near it disagree."""
    reads = [(t, r[field]) for t, r in reads if field in r]
    if not reads:
        return None
    near = {v for t, v in reads if abs(t - reads[0][0]) <= CLUSTER}
    return near.pop() if len(near) == 1 else None


def unit_of(fact: dict, said: Optional[str], others: list, reads: list, raid: bool) -> tuple:
    """(unit, how) of an HP/DEF fact's numbers: the units whose bars match the nearest reads (after the fact for
    'to', before it for 'from', both for 'by'; none past another change of that field or over NEAR away), else the
    one unit bars can hold, else what the line says; ('unknown', None) when none decides."""
    field, t, keys = fact["field"], stamp(fact["date"]), ("from", "to", "by")
    stated = {k: fact[k] for k in keys if fact.get(k) is not None}
    edges = [stamp(o["date"]) for o in others if any(o.get(k) != fact.get(k) for k in keys)]
    lo, hi = max((e for e in edges if e < t), default=None), min((e for e in edges if e > t), default=None)
    before = nearest([r for r in reversed(reads) if t - NEAR <= r[0] < t and (lo is None or r[0] > lo)], field)
    after = nearest([r for r in reads if t <= r[0] <= t + NEAR and (hi is None or r[0] < hi)], field)
    cap, fits, matched, checks = RAID_CELLS if raid else BAR_CELLS, [], [], 0
    for unit, per in (("bars", 1), ("points", PER_BAR[field])):
        bars = {k: v / per for k, v in stated.items()}
        ends = [bars[k] for k in ("from", "to") if k in bars]
        if "by" in bars:                                  # the bars before or after it, where a read gives one
            ends += [before + bars["by"]] if before is not None else []
            ends += [after - bars["by"]] if after is not None else []
        if all(v == int(v) for v in bars.values()) and all(0 <= v <= cap for v in ends) and \
                ("by" not in bars or 0 < abs(bars["by"]) <= cap):
            fits.append(unit)
        tests = [(bars["to"], after) if "to" in bars and after is not None else None,
                 (bars["from"], before) if "from" in bars and before is not None else None,
                 (bars["by"], after - before) if "by" in bars and None not in (before, after) else None]
        tests = [x for x in tests if x]
        checks = max(checks, len(tests))
        if tests and unit in fits and all(a == b for a, b in tests):
            matched.append(unit)
    if checks and matched:
        return ("bars" if "bars" in matched else matched[0]), "read"   # both: zeros, the same in either unit
    if len(fits) == 1:
        return fits[0], "range"
    if said and "bars" in fits:
        return "bars", "said" if said == "bars" else "list"
    return "unknown", None


def stat_units(facts: list, reads: dict, raid: set, counts: Counter):
    """Each stat fact's unit (unit_of): ATK/SPD bars; HP/DEF in points converted to bars ('stated': the line's
    numbers); a sure changelog fact of unknown unit becomes unsure (review: points or bars?)."""
    changes = {}
    for f in facts:
        if f.get("field") in PER_BAR and f["source"] == "changelog":
            changes.setdefault((f["card"], f["field"]), []).append(f)
    big = set(raid) | {k for k, rows in reads.items() if any(v > BAR_CELLS for _, r in rows for v in r.values())}
    units = [unit_of(f, f.get("said"), [o for o in changes.get((f["card"], f["field"]), []) if o is not f],
                     reads.get(f["card"], []), f["card"] in big or f["card"].startswith(("archived:", "morgue:")))
             if f.get("field") in PER_BAR else None for f in facts]   # big: raid bosses, or a template unknown
    for f, got in zip(facts, units):
        f.pop("said", None)
        if f.get("field") in ("atk", "spd"):
            f["unit"] = "bars"
        if not got:
            continue
        field, (unit, how) = f["field"], got
        f["unit"] = unit
        if how:
            f["unit_how"] = how
        counts[f"{f['source']} {field} unit {unit} ({how})"] += 1
        if unit == "points":
            f["stated"] = {k: f[k] for k in ("from", "to", "by") if f.get(k) is not None}
            for k, v in f["stated"].items():
                bars = v / PER_BAR[field]
                f[k] = int(bars) if bars == int(bars) else bars
        if unit == "unknown" and f["confidence"] == "sure":
            f["confidence"], f["_doubt"] = "unsure", DOUBT["unit"]
            counts["sure facts made unsure (unit)"] += 1


def authors() -> dict:
    return {**(read_json(expanduser(AUTHORS), {}) or {}), **MORE_AUTHORS}


def push_facts(root: str, names: Names, counts: Counter) -> tuple:
    """Evidence facts (one per commit naming cards) and first-repo rows (notices naming exactly one card)."""
    who = authors()
    facts, rows = [], []
    for m in messages(root, GITHUB_ID):
        date = iso(m["timestamp"])
        if date >= OLD_START:
            break
        commits = notice_commits(m)
        if not commits:
            continue
        counts["push notices"] += 1
        counts["push commits"] += len(commits)
        named = [(sha, subject, raw, commit_cards(subject, names), branch) for sha, subject, raw, branch in commits]
        union = sorted({c for *_, cards, _ in named for c in cards})
        if union:
            counts["push notices naming cards"] += 1
        for sha, subject, raw, cards, branch in named:
            author = who.get(raw)
            if raw not in who:
                counts["push commits with unmapped author"] += 1
            if not cards:
                continue
            facts.append({"id": "p" + short_id(m["id"], sha), "source": "push", "kind": "push", "date": date,
                          "channel": "github-updates", "channel_id": GITHUB_ID, "message": m["id"], "commit": sha,
                          "author": author or raw, "author_mapped": author is not None, "cards": cards,
                          "branch": branch, "confidence": "evidence"})
        if len(union) != 1:
            continue
        counts["push notices naming one recognised card"] += 1
        own = [(sha, subject, raw) for sha, subject, raw, cards, _ in named if cards]
        if not all(only_card(subject, union[0], names) for _, subject, _ in own):
            counts["push notices also listing other things"] += 1      # evidence only
            continue
        counts["push notices naming one card"] += 1
        rows += [{"id": "q" + short_id(m["id"], sha), "card": union[0], "date": date, "author": who.get(raw, raw),
                  "subject": subject.strip(), "badge": "first repo", "message": m["id"], "commit": sha,
                  "switch": SWITCH} for sha, subject, raw in own]
    counts["push cards named"] = len({c for f in facts for c in f["cards"]})
    return facts, rows


def vocabulary_of(root: str) -> set:
    v = read_json(join(root, *OLDGIT, "vocabulary.json"), {}) or {}
    return ({t.replace("_", " ") for t in v.get("types") or []} | set(TYPE_WORDS.values())) - NOT_TYPES


def sample(root: str, names: Names, vocabulary: set, n: int, seed: int, sure: bool = False):
    """Random lines that gave facts, printed with their facts (terminal only: the text never goes to a file)."""
    hits = []
    for source, channel, cname, m, i, line, got in changelog_lines(root, names, vocabulary):
        if got["facts"] and (not sure or any(f["sure"] for f in got["facts"]) and source == "changelog"):
            hits.append((cname, m["timestamp"][:10], f"{m['id']}:{i}", line, got["facts"]))
    for k, (cname, day, line_id, line, facts) in enumerate(Random(seed).sample(hits, min(n, len(hits))), 1):
        print(f"[{k}] {day} {cname} {line_id}\n    {line[:300]!r}")
        for f in facts:
            print("    ->", {a: b for a, b in f.items() if b is not None})


def main(argv=None):
    parser = ArgumentParser(prog="python -m commands.discord_facts", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("build", "sample"))
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--cards-repo", default=LOCAL_DIR_LOC)
    parser.add_argument("--no-first-repo-rows", dest="first_repo_rows", action="store_false",
                        help="hide the first repo's text-only rows (owner Q5 = A shows them)")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sure", action="store_true", help="sample: only lines with a sure changelog fact")
    args = parser.parse_args(argv)
    root = check_root(expanduser(args.root))
    names = load_names(expanduser(args.cards_repo), root)
    vocabulary = vocabulary_of(root)
    if args.command == "build":
        build(root, names, vocabulary, args.first_repo_rows)
    else:
        sample(root, names, vocabulary, args.n, args.seed, args.sure)


if __name__ == "__main__":
    main()
