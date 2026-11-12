"""
TTSCardMaker's hand-kept card files: site/Card Authors.txt (CardAuthors), site/Card Edits.json (CardEdits),
site/Card Creation Dates.json (CardDates) and site/Card Ratings.json (CardRatings), and their layout (dump_edits: the
edit.js editsFile port; dump_ratings). No Discord, network or psd-tools.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from json import dumps as json_dumps, loads as json_loads
from math import floor
from re import compile as re_compile, fullmatch, sub
from typing import Optional
from unicodedata import combining, normalize

from commands.card_models import CardInfo, CardStats, card_name

# ========================
# Configuration
# ========================
# edit.js FIELDS, in order: how an entry lists its fields
ORDER = ("name", "type", "subtype", "series", "subseries", "stars", "types", "hp", "def", "atk", "spd", "ability",
         "parts", "colors", "note", "art", "ability_art", "aliases", "dup_of", "hidden", "problems_ok", "allow_high_stats", "creatures",
         "flags", "names_ok")
# what localeCompare(…, {sensitivity: "base"}) puts before digits and letters, in CLDR root order
PUNCTUATION = "_-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$"
# per-layer text edits (docs/Finished/session-2026-10-26/PARTS_EDITOR.md §3): build_site.valid_parts' rules
ADD_ABILITY = "+ability"                        # the key of an ability added to a card that has none
PART_KEY = re_compile(r"^(\d+|\+ability)\Z")   # a layer id, or ADD_ABILITY
HEX8 = re_compile(r"^[0-9a-f]{8}\Z")
PARTS_MAX, PART_TEXT_MAX = 64, 4096
RENAME = "_rename"                              # true: the bake renames the card's files to its name (owner 2026-10-25)
SITE = "_site"                                  # an entry's website-only fields: their values differ from the card image on purpose
ART_FIELDS = ("art", "ability_art")             # artwork sources: art a list (one string in older entries), ability_art retired (10-31)
ART_MAX = 500                                   # characters a source (art.js ART_MAX, build_site ART_MAX)
FLAGS = ("hand", "bars", "name")                # what a card needs from a developer, each with a reason or "" (flags.js CARD_FLAGS, build_site FLAGS)
FLAG_MAX = 300                                  # characters a reason (flags.js, build_site)
# the name flag's "intended" (owner 2026-11-11): [website name, printed title, file stem] as a person marked them; the flag
# stays off until one of them changes (card_checks.name_note; edit.js, build_site NAMES_OK)
NAMES_OK = "names_ok"
# text colours: [{"field", "phrase"?, "color"}], a phrase as the effect layers' ("word#n"; none: the whole field); the
# card image's alone (owner 2026-11-11: the website's text and TTS's details draw the site's palette instead)
COLOR_FIELDS = ("name", "ability")      # textcolor.js FIELDS, build_site COLOR_FIELDS
COLOR = re_compile(r"^#[0-9a-f]{6}\Z")
COLORS_MAX, PHRASE_MAX = 64, 200


def valid_colors(value) -> bool:
    """A `colors` edit: a list of {field, color "#rrggbb", phrase?}, one per field and phrase (build_site.valid_colors)."""
    if not isinstance(value, list) or not value or len(value) > COLORS_MAX:
        return False
    seen = set()
    for one in value:
        if not isinstance(one, dict) or set(one) - {"field", "phrase", "color"} or one.get("field") not in COLOR_FIELDS \
                or not isinstance(one.get("color"), str) or not COLOR.match(one["color"]):
            return False
        phrase = one.get("phrase")
        if "phrase" in one and not (isinstance(phrase, str) and phrase.strip() == phrase and 0 < len(phrase) <= PHRASE_MAX):
            return False
        if (one["field"], phrase) in seen:
            return False
        seen.add((one["field"], phrase))
    return True


def color_text(one: dict) -> str:
    """"Ability: 'Burning' coloured #ff7a1a", "Ability: 'Burning' (2nd) coloured ...", "Name coloured #ffffff": one colour
    in plain words (textcolor.js colorText)."""
    phrase, nth = one.get("phrase"), ""
    numbered = phrase and fullmatch(r"(.*\S)#(\d+)", phrase)
    if numbered:
        n = int(numbered.group(2))
        phrase, nth = numbered.group(1), f" ({n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')})"
    return one["field"].capitalize() + (f": '{phrase}'{nth}" if phrase else "") + f" coloured {one['color']}"


def art_list(art, ability_art=None) -> list:
    """The artwork sources, in order: art (a list, or one string) then a legacy ability_art; whitespace collapsed, empty
    ones left out (build_site.art_list, art.js artList)."""
    items = (art if isinstance(art, list) else [art]) + [ability_art]
    return [t for t in (" ".join(x.split()) for x in items if isinstance(x, str)) if t]


def fold_art(fields: dict) -> dict:
    """fields with art as art_list's list (None when empty) and no ability_art, when either is there."""
    if "art" in fields or "ability_art" in fields:
        fields["art"] = art_list(fields.get("art"), fields.pop("ability_art", None)) or None
    return fields


@dataclass
class CardAuthors:
    """
    TTSCardMaker's site/Card Authors.txt: the team's record of who made each card.

    Two sections of "left = right" lines (# starts a comment):
      [people]  git/GitHub name (after the card repo's .mailmap) = name to show for that person
      [cards]   Card Name = person   (names match in any case, with _ and spaces alike)
    A listed card gets its listed person; any other card keeps its author. Either way the result goes through
    [people], so a git name becomes the display name. The website applies the same rules.
    """
    people: dict = field(default_factory=dict)      # lowercase git name -> display name
    cards: dict = field(default_factory=dict)       # normalized card name -> person

    @staticmethod
    def key(name: str) -> str:
        return sub(r"\s+", " ", name.replace("_", " ")).strip().lower()

    @classmethod
    def parse(cls, text: str) -> "CardAuthors":
        authors, section = cls(), None
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]").strip().lower()
                continue
            left, sep, right = line.partition("=")
            left, right = left.strip(), right.strip()
            if not sep or not left or not right:
                continue
            if section == "people":
                authors.people[left.lower()] = right
            elif section == "cards":
                authors.cards[cls.key(left)] = right
        return authors

    def resolve(self, card_name: str, current: Optional[str]) -> Optional[str]:
        person = self.cards.get(self.key(card_name), current)
        return self.people.get(person.lower(), person) if person else person


@dataclass
class CardEdits:
    """
    TTSCardMaker's site/Card Edits.json: card fixes from the website's Edit button.
    {"cards": {"Card Name": {"field": value, "_site": [field, ...], "_by": ..., "_at": ...}}}; names match like the
    authors file. Same fields as the website (site/build/build_site.py). The PSD values are kept in card.parsed_values.
    `_site` (owner 2026-10-25): card-image fields that are website only, applied like any edit but never baked (site_only).
    `_rename: true` (beside a name): the bake also renames the card's files (renames()).
    """
    cards: dict = field(default_factory=dict)       # normalized card name -> {field: value}
    meta: dict = field(default_factory=dict)        # normalized card name -> its _-keys (_by, _at, _site cleaned), never applied

    TYPES = {"name": str, "type": str, "subtype": str, "series": str, "subseries": str, "stars": int, "types": list,
             "hp": int, "def": int, "atk": int, "spd": int, "ability": str, "note": str, "art": list, "ability_art": str,
             "aliases": list, "dup_of": str, "hidden": bool, "problems_ok": bool, "allow_high_stats": bool, "creatures": list,
             "parts": dict, "flags": dict, "colors": list, NAMES_OK: list}
    NULLABLE = {"subtype", "series", "subseries", "stars", "hp", "def", "atk", "spd", "art", "ability_art"}
    ATTRS = {"name": "display_name", "type": "card_type", "subtype": "subtype", "series": "series", "subseries": "subseries",
             "stars": "stars", "types": "types", "ability": "ability", "note": "note", "art": "art", "ability_art": "ability_art",
             "hidden": "hidden", "problems_ok": "problems_ok", "allow_high_stats": "allow_high_stats"}
    STATS = {"hp": "hp", "def": "defense", "atk": "attack", "spd": "speed"}
    FIXED_BY = {"hp": ("HP NOT FOUND",), "def": ("DEF NOT FOUND",), "atk": ("ATK NOT FOUND",), "spd": ("SPD NOT FOUND",),
                "stars": ("NO STARS FOUND",), "type": ("UNKNOWN TYPE",), "types": ("MISSPELT TYPE",),
                "allow_high_stats": ("STATS TOO HIGH",), "creatures": ("NO STATS FOUND",)}    # build_site.FIXED_BY alike
    FIXED_BY_ALL = {("hp", "def", "atk", "spd"): ("NO STATS FOUND",)}   # answered once every one has a value (build_site's)
    # where the value lives: the card image (name: when it differs from the printed title), the card's folder, or this file alone
    KIND = {"name": "psd", "stars": "psd", "types": "psd", "hp": "psd", "def": "psd", "atk": "psd", "spd": "psd", "ability": "psd",
            "parts": "psd", "type": "folder", "subtype": "folder", "series": "folder",
            "subseries": "database", "note": "database", "art": "database", "ability_art": "database", "aliases": "database",
            "dup_of": "database", "hidden": "database", "problems_ok": "database", "allow_high_stats": "database",
            "creatures": "psd", "flags": "database", "colors": "psd", NAMES_OK: "database"}     # creatures: printed on a bar per creature; else website only (psd_edit.classify)
    SITE_FIELDS = ("name", "stars", "types", "hp", "def", "atk", "spd", "ability", "creatures", "colors")    # what `_site` may name (edit.js SITE)

    @classmethod
    def answered(cls, fields: dict) -> tuple:
        """The parser notes these edited fields answer (FIXED_BY; FIXED_BY_ALL's once all its fields have values)."""
        return tuple(p for key in fields for p in cls.FIXED_BY.get(key, ())) + tuple(
            p for keys, notes in cls.FIXED_BY_ALL.items() if all(fields.get(k) is not None for k in keys) for p in notes)

    @classmethod
    def valid(cls, name: str, value) -> bool:
        kind = cls.TYPES.get(name)
        if kind is None:
            return False
        if value is None:
            return name in cls.NULLABLE
        if kind is int:
            return isinstance(value, int) and not isinstance(value, bool) and value >= 0
        if name == "creatures":                 # [{"hp": 7, "def": 4, "atk": 5, "spd": 6}, ...]
            return isinstance(value, list) and all(isinstance(c, dict) and all(
                k in cls.STATS and (v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 0))
                for k, v in c.items()) for c in value)
        if name in ART_FIELDS:                  # a source, or (art) a list of them
            return all(isinstance(x, str) and len(x) <= ART_MAX for x in (value if name == "art" and isinstance(value, list) else [value]))
        if name == "colors":
            return valid_colors(value)
        if name == NAMES_OK:
            return isinstance(value, list) and len(value) == 3 and all(isinstance(x, str) and 0 < len(x) <= FLAG_MAX for x in value)
        if kind is list:
            return isinstance(value, list) and all(isinstance(x, str) for x in value)
        if name == "flags":                     # {"hand": "reason", "bars": "", "psd": the PSD blob it was set on}
            return isinstance(value, dict) and any(k in FLAGS for k in value) and all(
                bool(HEX8.match(v)) if k == "psd" and isinstance(v, str) else k in FLAGS and isinstance(v, str) and len(v) <= FLAG_MAX
                for k, v in value.items())
        if name == "parts":
            return valid_parts(value)
        return isinstance(value, kind)

    @classmethod
    def site_only(cls, entry: dict) -> list:
        """An entry's `_site`, cleaned (edit.js editsFile's rule): card-image fields with a valid value, in ORDER."""
        named = entry.get(SITE)
        if not isinstance(named, list):
            return []
        return [f for f in ORDER if f in cls.SITE_FIELDS and f in named and entry.get(f) is not None and cls.valid(f, entry[f])]

    def site(self, key: str) -> list:
        """A card's website-only fields (key: CardAuthors.key)."""
        return self.meta.get(key, {}).get(SITE) or []

    @classmethod
    def parse(cls, text: str) -> "CardEdits":
        """Raises ValueError for a file that isn't JSON; bad entries are skipped."""
        data = json_loads(text)
        cards = data.get("cards") if isinstance(data, dict) else None
        edits = cls()
        for name, fields in (cards if isinstance(cards, dict) else {}).items():
            if isinstance(fields, dict):
                key = CardAuthors.key(name)
                edits.cards[key] = fold_art({k: v for k, v in fields.items() if not k.startswith("_") and cls.valid(k, v)})
                edits.meta[key] = {k: v for k, v in fields.items() if k.startswith("_") and k != SITE}
                site = cls.site_only(fields)
                if site:
                    edits.meta[key][SITE] = site
        return edits

    def renames(self, key: str) -> bool:
        """Whether the bake renames a card's files to its name (key: CardAuthors.key): `_rename` true beside a name that
        isn't website only (`_site`); build_site.parse_edits' rule."""
        return (self.meta.get(key) or {}).get(RENAME) is True and isinstance(self.cards.get(key, {}).get("name"), str) \
            and "name" not in self.site(key)

    def get(self, card: "CardInfo", name: str):
        if name == "creatures":
            return [{k: (lambda v: v if v >= 0 else None)(getattr(c, a)) for k, a in self.STATS.items()} for c in card.creatures] if getattr(card, "creatures", None) else None
        if name in self.STATS:
            value = getattr(card.stats, self.STATS[name])
            return value if value >= 0 else None
        if name == "problems":
            return list(card.problems)
        value = getattr(card, self.ATTRS[name], None)
        return list(value) if isinstance(value, list) else value

    def set(self, card: "CardInfo", name: str, value) -> None:
        if name == "creatures":
            card.creatures = [CardStats(**{a: -1 if c.get(k) is None else c[k] for k, a in self.STATS.items()}) for c in value] if value else None
            if card.creatures:
                card.stats = deepcopy(card.creatures[0])
        elif name in self.STATS:
            setattr(card.stats, self.STATS[name], -1 if value is None else value)
        elif name == "problems":
            card.problems = list(value or [])
        elif name == "art":                     # the list; the retired second field is in it
            card.art, card.ability_art = art_list(value) or None, None
        else:
            setattr(card, self.ATTRS[name], list(value) if isinstance(value, list) else value)

    def apply(self, name: str, card: "CardInfo") -> bool:
        """Put back the PSD values, then apply this card's edits. Returns whether the card changed."""
        before = repr(card)
        for key, value in (getattr(card, "parsed_values", None) or {}).items():
            self.set(card, key, value)
        fields = {k: v for k, v in self.cards.get(CardAuthors.key(name), {}).items() if k in self.ATTRS or k in self.STATS or k == "creatures"}
        if "creatures" in fields:                   # set first, so a single-stat edit can still change creature 1
            fields = {"creatures": fields.pop("creatures"), **fields}
        saved = {key: self.get(card, key) for key in fields}
        if "creatures" in fields and saved["creatures"] is None:    # it sets the stats: undone, they come back
            saved.update({key: self.get(card, key) for key in self.STATS if key not in saved})
        if fields:
            saved["problems"] = list(card.problems)
            answered = self.answered(fields)
            card.problems = [] if fields.get("problems_ok") else [p for p in card.problems if not answered or not p.startswith(answered)]
        for key, value in fields.items():
            self.set(card, key, value)
        card.parsed_values = saved or None
        return repr(card) != before

    def names(self) -> dict[str, list[str]]:
        """Card name -> website aliases and display name, for name lookups."""
        out = {}
        for key, fields in self.cards.items():
            extra = list(fields.get("aliases") or []) + ([fields["name"]] if fields.get("name") else [])
            if extra:
                out[key] = extra
        return out


def valid_parts(value) -> bool:
    """A `parts` edit: {lyid | "+ability": {field, text | delete, was, sha, psd, layer}} (build_site.valid_parts)."""
    if not isinstance(value, dict) or not value or len(value) > PARTS_MAX:
        return False
    for key, part in value.items():
        if not (isinstance(key, str) and PART_KEY.match(key) and isinstance(part, dict)
                and part.get("field") in ("name", "ability")):
            return False
        text = part.get("text")
        if ("text" in part) == bool(part.get("delete")) or ("text" in part and not (
                isinstance(text, str) and len(text.encode("utf-8")) <= PART_TEXT_MAX)):
            return False
        added = key == ADD_ABILITY
        if not (isinstance(part.get("psd"), str) and HEX8.match(part["psd"]) or added and part.get("psd") in (None, "")):
            return False
        if not (isinstance(part.get("sha"), str) and HEX8.match(part["sha"]) or added and part.get("sha") is None):
            return False
        if added and (part.get("delete") or part["field"] != "ability"):
            return False
    return True


def as_parsed(card: CardInfo) -> CardInfo:
    """
    The card as its PSD reads it: a copy with its parsed_values put back (its Card Edits.json edits undone), so a
    change of the PSD and a change of an edit can be told apart.
    """
    out = deepcopy(card)
    edits = CardEdits()
    for key, value in (getattr(card, "parsed_values", None) or {}).items():
        if key in CardEdits.ATTRS or key in CardEdits.STATS or key in ("creatures", "problems"):
            edits.set(out, key, value)
    out.parsed_values = None
    return out


@dataclass
class CardDates:
    """
    TTSCardMaker's site/Card Creation Dates.json: when each card was made, the earlier of its first Discord post and
    its first upload (UTC day). {"cards": {"Card Name": {"date": "YYYY-MM-DD", "from": "discord", "provisional": true}}};
    names match like the authors file. Written by `python -m commands.card_dates created`, which keeps "from": "hand"
    entries (the owner's). The website and /update_stats read it, each with its own first upload for the cards it
    misses (created()); it is no edit (never in Card Edits.json).
    """
    cards: dict = field(default_factory=dict)       # normalized card name -> {"date", "from"[, "provisional"]}

    FROM = ("discord", "git", "hand")               # its first Discord post, first git upload, or set by hand

    @classmethod
    def valid(cls, entry) -> bool:
        if not isinstance(entry, dict) or entry.get("from") not in cls.FROM or not isinstance(entry.get("provisional", False), bool):
            return False
        try:
            return isinstance(entry.get("date"), str) and date.fromisoformat(entry["date"]).isoformat() == entry["date"]
        except ValueError:
            return False

    @classmethod
    def parse(cls, text: str) -> "CardDates":
        """Raises ValueError for a file that isn't JSON; bad entries are skipped."""
        data = json_loads(text)
        cards = data.get("cards") if isinstance(data, dict) else None
        dates = cls()
        for name, entry in (cards if isinstance(cards, dict) else {}).items():
            if cls.valid(entry):
                dates.cards[CardAuthors.key(name)] = {"date": entry["date"], "from": entry["from"],
                                                      **({"provisional": True} if entry.get("provisional") else {})}
        return dates

    def created(self, name: str, upload: Optional[str] = None) -> Optional[dict]:
        """
        A card's created, by the website's rule (build_site.py apply_created): its entry, unless its first upload is
        earlier and the entry isn't a hand one; no entry: the first upload ("from": "git").

        Args:
            name: Card name
            upload: Its first upload (UTC day, YYYY-MM-DD), or None when unknown

        Returns:
            {date, from[, provisional]}, or None (undated)
        """
        found = self.cards.get(CardAuthors.key(name))
        if found and (found["from"] == "hand" or not upload or found["date"] <= upload):
            return dict(found)
        return {"date": upload, "from": "git"} if upload else None

    def apply(self, name: str, card: "CardInfo", upload: Optional[str] = None) -> bool:
        """Set the card's created (created(); none: the class default). Returns whether it changed."""
        value = self.created(name, upload)
        if value == getattr(card, "created", None):
            return False
        if value is None:
            card.__dict__.pop("created", None)      # the class default
        else:
            card.created = value
        return True


@dataclass
class CardRatings:
    """
    TTSCardMaker's site/Card Ratings.json: dated ratings per card, whole numbers 1-10 (owner, 2026-10-26), oldest first.
    {"about": ..., "cards": {"Card Name": [{"date", "scale", "value", "by", "source"}]}}; names match like the authors
    file. by: "group" (a TTS session) or a GitHub login. Written by `python -m commands.tts_ratings` (the website's Rate it
    until 2026-11-11); read by the website build only, which shows none (never the bot, the card sync or the checks).
    """
    cards: dict = field(default_factory=dict)       # normalized card name -> (name as written, [entry])

    SCALES = ("favorability", "power")
    SOURCES = ("TTS session", "website")
    KEYS = ("date", "scale", "value", "by", "source")
    ABOUT = ("Card ratings, whole numbers 1 to 10. favorability: how much players like the card; power: 1 weak, 10 too "
             "strong; the middle (5.5) is balanced. Keyed by card file name; entries oldest first. by: \"group\" (a TTS "
             "session) or a GitHub login. Read by the website only.")

    @classmethod
    def valid(cls, entry) -> bool:
        if not isinstance(entry, dict) or entry.get("scale") not in cls.SCALES or entry.get("source") not in cls.SOURCES:
            return False
        value, by = entry.get("value"), entry.get("by")
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 10:      # whole numbers only
            return False
        if not isinstance(by, str) or not by or len(by) > 100:
            return False
        try:
            return isinstance(entry.get("date"), str) and date.fromisoformat(entry["date"]).isoformat() == entry["date"]
        except ValueError:
            return False

    @staticmethod
    def number(value: float):
        """The nearest whole number, halves up like JS Math.round (4.5 -> 5; Python's round() gives 4)."""
        return int(floor(value + 0.5))

    @classmethod
    def entry(cls, day: str, scale: str, value: float, by: str, source: str) -> dict:
        return {"date": day, "scale": scale, "value": cls.number(value), "by": by, "source": source}

    @classmethod
    def order(cls, entries: list) -> list:
        """Oldest first: date, then source (TTS sessions first), then by, then scale (build_site rating_order)."""
        rank = lambda seq, v: seq.index(v) if v in seq else len(seq)
        return sorted(entries, key=lambda e: (str(e.get("date", "")), rank(cls.SOURCES, e.get("source")),
                                              str(e.get("by", "")).lower(), rank(cls.SCALES, e.get("scale"))))

    @classmethod
    def parse(cls, text: str) -> "CardRatings":
        """Raises ValueError for a file that isn't JSON; bad entries and non-list values are skipped."""
        data = json_loads(text)
        cards = data.get("cards") if isinstance(data, dict) else None
        ratings = cls()
        for name, entries in (cards if isinstance(cards, dict) else {}).items():
            kept = [cls.entry(*(e[k] for k in cls.KEYS)) for e in entries if cls.valid(e)] if isinstance(entries, list) else []
            if kept:
                ratings.cards[CardAuthors.key(name)] = (name, kept)
        return ratings


# ========================
# The files' layout (edit.js editsFile)
# ========================
def _json(value) -> str:
    """JSON.stringify: compact, UTF-8 as is."""
    return json_dumps(value, ensure_ascii=False, separators=(",", ":"))


def _collate(key: str) -> list:
    """How localeCompare with sensitivity "base" orders keys: case and accents ignored; spaces, punctuation, symbols, digits, letters."""
    out = []
    for char in normalize("NFKD", key):
        if combining(char):
            continue
        if char.isspace():
            out.append((0, 0))
        elif char in PUNCTUATION:
            out.append((1, PUNCTUATION.index(char)))
        elif char.isdigit():
            out.append((3, ord(char)))
        elif char.isalpha():
            out.append((4, ord(char.lower())))
        else:
            out.append((2, ord(char)))
    return out


NO_CARD_FOLDERS = ("MDW", "Markers", "Types", "site", "TTS")    # no card PNG lives under these (card_data's duplicates)


def dup_pictures(paths, cards: Optional[dict]) -> dict:
    """
    {original PSD path: [PNG paths]}: the picture-only duplicates (owner 11-07) that are no numbered picture of a PSD
    (NameN.png beside Name.psd: card_bake's png_for): a card PNG with no PSD of its own whose Card Edits.json entry names
    its original in dup_of (Stubborn_Queen.png), the original a PSD in paths (by card key).

    Args:
        paths: The tree's paths
        cards: Card Edits.json's "cards"
    """
    lower = {path.lower() for path in paths}
    psds = {CardAuthors.key(card_name(path)): path for path in paths if path.lower().endswith(".psd")
            and not any(folder in NO_CARD_FOLDERS for folder in path.split("/"))}
    entries = {CardAuthors.key(key): value for key, value in (cards or {}).items() if isinstance(value, dict)}
    out: dict = {}
    for path in sorted(paths):
        stem = path[:len(path) - 4]
        if (not path.lower().endswith(".png") or "/" not in path or any(f in NO_CARD_FOLDERS for f in path.split("/"))
                or (stem + ".psd").lower() in lower):
            continue
        number = fullmatch(r"(.+?)(\d+)", stem)
        if number and (number.group(1) + ".psd").lower() in lower:
            continue                            # a numbered picture: its PSD's own (png_for)
        target = (entries.get(CardAuthors.key(card_name(path))) or {}).get("dup_of")
        original = psds.get(CardAuthors.key(target)) if isinstance(target, str) else None
        if original:
            out.setdefault(original, []).append(path)
    return out


def dump_edits(data: dict) -> str:
    """
    Card Edits.json as edit.js editsFile writes it: the other top-level keys first, cards sorted, one line per card.

    Args:
        data: The parsed file

    Returns:
        The text (with its final newline)
    """
    cards = data.get("cards") or {}
    rest = "".join(f"  {_json(k)}: {_json(v)},\n" for k, v in data.items() if k != "cards")
    keys = sorted(cards, key=_collate)
    body = ",\n".join(f"    {_json(k)}: {_json(cards[k])}" for k in keys)
    return "{\n" + rest + '  "cards": {' + (f"\n{body}\n  " if keys else "") + "}\n}\n"


def dump_ratings(data: dict) -> str:
    """
    Card Ratings.json in dump_edits' layout, each card's line its list of entries.

    Args:
        data: The parsed file

    Returns:
        The text (with its final newline)
    """
    return dump_edits(data)


# ========================
# Module Exports
# ========================
__all__ = [
    'CardAuthors',
    'CardDates',
    'CardEdits',
    'CardRatings',
    'ADD_ABILITY',
    'ART_FIELDS',
    'ART_MAX',
    'COLOR_FIELDS',
    'FLAGS',
    'FLAG_MAX',
    'NAMES_OK',
    'art_list',
    'color_text',
    'fold_art',
    'ORDER',
    'PUNCTUATION',
    'RENAME',
    'SITE',
    'as_parsed',
    'dump_edits',
    'dup_pictures',
    'dump_ratings',
    'valid_colors',
    'valid_parts',
]
