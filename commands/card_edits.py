"""
TTSCardMaker's hand-kept card files: site/Card Authors.txt (CardAuthors), site/Card Edits.json (CardEdits),
site/Card Creation Dates.json (CardDates) and site/Card Ratings.json (CardRatings), and the layout the website writes
them in (dump_edits, dump_ratings: the edit.js editsFile and ratingsFile ports). No Discord, network or psd-tools.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from json import dumps as json_dumps, loads as json_loads
from math import floor
from re import compile as re_compile, sub
from typing import Optional
from unicodedata import combining, normalize

from commands.card_models import CardInfo, CardStats

# ========================
# Configuration
# ========================
# edit.js FIELDS, in order: how an entry lists its fields
ORDER = ("name", "type", "subtype", "series", "subseries", "stars", "types", "hp", "def", "atk", "spd", "ability",
         "note", "aliases", "dup_of", "hidden", "problems_ok", "allow_high_stats", "creatures")
# what localeCompare(…, {sensitivity: "base"}) puts before digits and letters, in CLDR root order
PUNCTUATION = "_-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$"


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
class Mailmap:
    """
    git's .mailmap, for names git doesn't map (the GitHub API's commits). The name forms, emails in any case:
      Proper Name <proper@email>                            a commit by that email shows as Proper Name
      Proper Name <proper@email> <commit@email>             the same, by commit@email
      Proper Name <proper@email> Commit Name <commit@email> only a commit by that name and email
    """
    names: dict = field(default_factory=dict)       # lowercase email -> proper name
    pairs: dict = field(default_factory=dict)       # (lowercase name, lowercase email) -> proper name

    LINE = re_compile(r"([^<]*)<([^>]*)>(?:([^<]*)<([^>]*)>)?")

    @classmethod
    def parse(cls, text: str) -> "Mailmap":
        mailmap = cls()
        for line in text.splitlines():
            found = cls.LINE.match(line.strip())
            if line.lstrip().startswith("#") or not found:
                continue
            proper, email, name, other = (" ".join((g or "").split()) for g in found.groups())
            if not proper:                          # <proper@email> <commit@email>: an email only
                continue
            if found.group(4) is None:
                mailmap.names[email.lower()] = proper
            elif name:
                mailmap.pairs[(name.lower(), other.lower())] = proper
            else:
                mailmap.names[other.lower()] = proper
        return mailmap

    def name(self, name: Optional[str], email: Optional[str]) -> Optional[str]:
        """The commit author's proper name (the name itself when unmapped)."""
        email = (email or "").strip().lower()
        return self.pairs.get(((name or "").strip().lower(), email)) or self.names.get(email) or name


@dataclass
class CardEdits:
    """
    TTSCardMaker's site/Card Edits.json: card fixes from the website's Edit button.
    {"cards": {"Card Name": {"field": value, "_by": ..., "_at": ...}}}; names match like the authors file.
    Same fields as the website (site/build/build_site.py). The PSD values are kept in card.parsed_values.
    """
    cards: dict = field(default_factory=dict)       # normalized card name -> {field: value}
    meta: dict = field(default_factory=dict)        # normalized card name -> its _-keys (_by, _at, _psd), never applied

    TYPES = {"name": str, "type": str, "subtype": str, "series": str, "subseries": str, "stars": int, "types": list,
             "hp": int, "def": int, "atk": int, "spd": int, "ability": str, "note": str, "aliases": list,
             "dup_of": str, "hidden": bool, "problems_ok": bool, "allow_high_stats": bool, "creatures": list}
    NULLABLE = {"subtype", "series", "subseries", "stars", "hp", "def", "atk", "spd"}
    ATTRS = {"name": "display_name", "type": "card_type", "subtype": "subtype", "series": "series", "subseries": "subseries",
             "stars": "stars", "types": "types", "ability": "ability", "note": "note", "hidden": "hidden",
             "problems_ok": "problems_ok", "allow_high_stats": "allow_high_stats"}
    STATS = {"hp": "hp", "def": "defense", "atk": "attack", "spd": "speed"}
    FIXED_BY = {"hp": ("HP NOT FOUND",), "def": ("DEF NOT FOUND",), "atk": ("ATK NOT FOUND",), "spd": ("SPD NOT FOUND",),
                "ability": ("ABILITY TEXT NOT FOUND", "NO ABILITY LAYER"), "stars": ("NO STARS FOUND",),
                "type": ("UNKNOWN TYPE",), "types": ("MISSPELT TYPE",), "allow_high_stats": ("STATS TOO HIGH",),
                "creatures": ("SEVERAL CREATURES",)}
    # where the value lives: the card image (name: when it differs from the printed title), the card's folder, or this file alone
    KIND = {"name": "psd", "stars": "psd", "types": "psd", "hp": "psd", "def": "psd", "atk": "psd", "spd": "psd", "ability": "psd",
            "type": "folder", "subtype": "folder", "series": "folder",
            "subseries": "database", "note": "database", "aliases": "database", "dup_of": "database", "hidden": "database",
            "problems_ok": "database", "allow_high_stats": "database", "creatures": "database"}
    PSD_STATES = ("baked", "refused")               # "_psd" meta states written by the edit pipeline; "pending" is computed by the website

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
        if kind is list:
            return isinstance(value, list) and all(isinstance(x, str) for x in value)
        return isinstance(value, kind)

    @classmethod
    def parse(cls, text: str) -> "CardEdits":
        """Raises ValueError for a file that isn't JSON; bad entries are skipped."""
        data = json_loads(text)
        cards = data.get("cards") if isinstance(data, dict) else None
        edits = cls()
        for name, fields in (cards if isinstance(cards, dict) else {}).items():
            if isinstance(fields, dict):
                key = CardAuthors.key(name)
                edits.cards[key] = {k: v for k, v in fields.items() if not k.startswith("_") and cls.valid(k, v)}
                edits.meta[key] = {k: v for k, v in fields.items() if k.startswith("_")}
        return edits

    def psd_state(self, name: str) -> Optional[dict]:
        """
        A card's "_psd" meta from the edit pipeline.

        Args:
            name: Card name (matched like the authors file).

        Returns:
            {"state": "baked" | "refused", ...} or None when absent or malformed.
        """
        value = self.meta.get(CardAuthors.key(name), {}).get("_psd")
        return value if isinstance(value, dict) and value.get("state") in self.PSD_STATES else None

    def set_psd_state(self, name: str, value: Optional[dict]) -> None:
        """
        Write a card's "_psd" meta (None removes it); its edits, _by and _at stay as they are.

        Args:
            name: Card name (matched like the authors file).
            value: The meta dict; its "state" must be one of PSD_STATES.

        Raises:
            ValueError: for a state outside PSD_STATES.
        """
        key = CardAuthors.key(name)
        if value is None:
            self.meta.get(key, {}).pop("_psd", None)
        elif not isinstance(value, dict) or value.get("state") not in self.PSD_STATES:
            raise ValueError(f"_psd state must be one of {self.PSD_STATES}")
        else:
            self.meta.setdefault(key, {})["_psd"] = dict(value)

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
            answered = tuple(p for key in fields for p in self.FIXED_BY.get(key, ()))
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
    names match like the authors file. Written by `python -m commands.discord_edits created`, which keeps "from": "hand"
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
            card.__dict__.pop("created", None)      # the class default: an undated card pickles as before
        else:
            card.created = value
        return True


@dataclass
class CardRatings:
    """
    TTSCardMaker's site/Card Ratings.json: dated 1-10 ratings per card, oldest first.
    {"about": ..., "cards": {"Card Name": [{"date", "scale", "value", "by", "source"}]}}; names match like the authors
    file. by: "group" (a TTS session) or a GitHub login. Written by `python -m commands.tts_ratings` and the website's
    Rate it (edit.js ratingsFile); read by the website only (never the bot, the card sync or the checks).
    """
    cards: dict = field(default_factory=dict)       # normalized card name -> (name as written, [entry])

    SCALES = ("favorability", "power")
    SOURCES = ("TTS session", "website")
    KEYS = ("date", "scale", "value", "by", "source")
    ABOUT = ("Card ratings, 1 to 10. favorability: how much players like the card; power: 1 weak, 5.5 balanced, 10 too "
             "strong. Keyed by card file name; entries oldest first. by: \"group\" (a TTS session) or a GitHub login. "
             "Read by the website only.")      # edit.js RATINGS_ABOUT

    @classmethod
    def valid(cls, entry) -> bool:
        if not isinstance(entry, dict) or entry.get("scale") not in cls.SCALES or entry.get("source") not in cls.SOURCES:
            return False
        value, by = entry.get("value"), entry.get("by")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 1 <= value <= 10:
            return False
        if not isinstance(by, str) or not by or len(by) > 100:
            return False
        try:
            return isinstance(entry.get("date"), str) and date.fromisoformat(entry["date"]).isoformat() == entry["date"]
        except ValueError:
            return False

    @staticmethod
    def number(value: float):
        """One decimal, as JS Math.round(v * 10) / 10; whole values as int (JSON.stringify writes 7, not 7.0)."""
        value = floor(value * 10 + 0.5) / 10
        return int(value) if value == int(value) else value

    @classmethod
    def entry(cls, day: str, scale: str, value: float, by: str, source: str) -> dict:
        return {"date": day, "scale": scale, "value": cls.number(value), "by": by, "source": source}

    @classmethod
    def order(cls, entries: list) -> list:
        """Oldest first: date, then source (TTS sessions first), then by, then scale (edit.js ratingsFile, build_site)."""
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
# The files' layout (edit.js editsFile, ratingsFile)
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
    Card Ratings.json as edit.js ratingsFile writes it: dump_edits' layout, each card's line its list of entries.

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
    'ORDER',
    'PUNCTUATION',
    'as_parsed',
    'dump_edits',
    'dump_ratings',
]
