"""
TTSCardMaker's hand-kept card files: site/Card Authors.txt (CardAuthors) and site/Card Edits.json (CardEdits).
No Discord, network or psd-tools.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from json import loads as json_loads
from re import sub
from typing import Optional

from commands.card_models import CardInfo, CardStats


@dataclass
class CardAuthors:
    """
    TTSCardMaker's site/Card Authors.txt: the team's record of who made each card.

    Two sections of "left = right" lines (# starts a comment):
      [people]  git/GitHub name = name to show for that person
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


# ========================
# Module Exports
# ========================
__all__ = [
    'CardAuthors',
    'CardEdits',
]
