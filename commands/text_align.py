"""
How an edit of card text lines up with the text it replaces (owner 2026-11-03, Pot Full of Spiders!): paragraphs first,
then words, never letters across a break. The bake keeps what the edit keeps (its runs: colours, badges, sizes) and
gives each changed piece the style of what it replaces, so the alignment decides the colours. difflib over letters
paired "When damaged, spend all spider counters" with leftover letters of the paragraph above it (" a", "nd ",
" spider counters "): the new line took that paragraph's colour.

    1. Paragraphs (split at \\r, \\n and \\x03): identical ones anchor; between anchors each pairs with the one in its
       place, or with a likelier one when paragraphs were added or deleted (most words alike, PAIR_BONUS per pair).
    2. Inside a paired paragraph: words, [tokens] and (badges) as units, so a kept "to" is never the end of "into".
       One word, token or badge changed into one alike of its kind (REFINE_MIN) is compared letter by letter:
       "Poison" -> "Poisoned" keeps "Poison", "(1GCD)" -> "(1CD)" its letters.
    3. Every character has a group: its paragraph pair, or its unpaired paragraph (added or deleted). A break goes with
       the paragraph after it ("\\rNew line" after the line above, "\\rGone line"), but with the one before it when
       that one is unpaired and nothing before it is paired ("Gone line\\r" first), or when the paragraph after it is
       paired with a first paragraph (no break before it there).
    4. segments() splits the changes at group changes: a deleted or added paragraph is a change of its own, never
       one with the paragraph next to it (Morpho Knight's deleted first line took the rewritten second line's start
       into its colour). moves(): an added paragraph alike a deleted one (MOVE_MIN) is that one moved; the bake
       colours it as it was.

blocks() is difflib's get_matching_blocks() and opcodes() its get_opcodes() (consecutive changes possible), so callers
swap them in.
"""
from collections import Counter
from difflib import SequenceMatcher
from re import DOTALL
from re import compile as re_compile
from typing import Optional

# ========================
# Configuration
# ========================
BREAKS = "\r\n\x03"
UNIT = re_compile(r"\[[^\[\]\r\n\x03]{1,52}\]|\(-?[0-9A-Za-z/]{2,6}\)|[^\W_]+(?:['’][^\W_]+)*|.", DOTALL)
WORD = re_compile(r"[^\W_]")
PAIR_BONUS = 0.05       # a pair's worth beyond its likeness: paragraphs pair in place unless added or deleted ones
REFINE_MIN = 0.5        # one word into one word: letters compared when this alike (difflib ratio)
MOVE_MIN = 0.6          # an added paragraph this alike a deleted one is it moved


# ========================
# Pieces
# ========================
def paragraphs(text: str) -> list:
    """[(start, end)] of each paragraph's text, the breaks between them left out ("A\\r\\rB": A, "", B)."""
    out, at = [], 0
    for i, c in enumerate(text):
        if c in BREAKS:
            out.append((at, i))
            at = i + 1
    return out + [(at, len(text))]


def units(text: str, start: int = 0, end: Optional[int] = None) -> list:
    """[(start, end)] of text[start:end]'s units: [tokens], (badges), words (apostrophes inside), single characters."""
    return [(start + m.start(), start + m.end()) for m in UNIT.finditer(text[start:len(text) if end is None else end])]


def likeness(a: str, b: str) -> float:
    """How alike two paragraphs are, 0..1: the mean of difflib's ratio over their words (case aside) and the share of
    words they have in common, so a reworded line still is its old self; 1.0 for two blank ones."""
    x = [a[s:e].lower() for s, e in units(a) if not a[s:e].isspace()]
    y = [b[s:e].lower() for s, e in units(b) if not b[s:e].isspace()]
    if not x and not y:
        return 1.0
    shared = sum((Counter(x) & Counter(y)).values())
    return (SequenceMatcher(None, x, y, autojunk=False).ratio() + 2 * shared / (len(x) + len(y))) / 2


# ========================
# Paragraph pairs
# ========================
def _best_pairs(old: list, new: list) -> list:
    """The pairs of two runs of paragraphs (texts) in order with the most likeness, PAIR_BONUS each."""
    n, m = len(old), len(new)
    like = [[likeness(a, b) + PAIR_BONUS for b in new] for a in old]
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j] = max(score[i - 1][j - 1] + like[i - 1][j - 1], score[i - 1][j], score[i][j - 1])
    out, i, j = [], n, m
    while i and j:
        if score[i][j] == score[i - 1][j - 1] + like[i - 1][j - 1]:
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif score[i][j] == score[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def pairs(old: list, new: list) -> list:
    """
    [(old index, new index)] of the paragraphs (texts) that are one paragraph before and after the edit. A paragraph
    likelier (MOVE_MIN at least) one outside its stretch between anchors than any inside it is left unpaired: it moved
    (Z!!!!!'s first and last lines swapped kept their places' colours).
    """
    ops = SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
    stretches = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in ops if tag != "equal"]
    loose_a = [k for i1, i2, _, _ in stretches for k in range(i1, i2)]
    loose_b = [k for _, _, j1, j2 in stretches for k in range(j1, j2)]
    moved_a, moved_b = set(), set()
    for i1, i2, j1, j2 in stretches:
        for k in range(i1, i2):
            inside = max([likeness(old[k], new[m]) for m in range(j1, j2)] or [0.0])
            outside = max([likeness(old[k], new[m]) for m in loose_b if not j1 <= m < j2] or [0.0])
            if outside >= MOVE_MIN and outside > inside:
                moved_a.add(k)
        for m in range(j1, j2):
            inside = max([likeness(old[k], new[m]) for k in range(i1, i2)] or [0.0])
            outside = max([likeness(old[k], new[m]) for k in loose_a if not i1 <= k < i2] or [0.0])
            if outside >= MOVE_MIN and outside > inside:
                moved_b.add(m)
    out = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            out += [(i1 + k, j1 + k) for k in range(i2 - i1)]
        else:
            xs = [k for k in range(i1, i2) if k not in moved_a]
            ys = [m for m in range(j1, j2) if m not in moved_b]
            out += [(xs[a], ys[b]) for a, b in _best_pairs([old[k] for k in xs], [new[m] for m in ys])]
    return out


class Layout:
    """An edit's paragraphs (pa, pb), their pairs, and each character's group (ga, gb): ("p", pair number, place), or
    ("a", old paragraph, place) / ("b", new paragraph, place) for an unpaired one (rule 3; _groups' places)."""

    def __init__(self, old: str, new: str):
        self.old, self.new = old, new
        self.pa, self.pb = paragraphs(old), paragraphs(new)
        self.pairs = pairs([old[s:e] for s, e in self.pa], [new[s:e] for s, e in self.pb])
        self.ga = self._groups(old, self.pa, {i: (n, j) for n, (i, j) in enumerate(self.pairs)}, "a")
        self.gb = self._groups(new, self.pb, {j: (n, i) for n, (i, j) in enumerate(self.pairs)}, "b")

    @staticmethod
    def _groups(text: str, ps: list, partner: dict, side: str) -> list:
        """Each character's group (kind, number, place): place 1 its paragraph's text, 0 the break before it, 2 the
        break after it; a paired paragraph's break is a change of its own (a soft break made a paragraph break never
        takes the words typed after it into the line above's colour), an unpaired one's goes with its text."""
        own = [("p", partner[k][0]) if k in partner else (side, k) for k in range(len(ps))]
        first = min(partner) if partner else len(ps)
        out = [None] * len(text)
        for k, (s, e) in enumerate(ps):
            out[s:e] = [own[k] + (1,)] * (e - s)
            if k + 1 < len(ps):                 # the break after paragraph k
                after = k + 1
                if after in partner:
                    before = partner[after][1] > 0
                else:
                    before = after > first
                owner = after if before else k      # an unpaired paragraph is one change with its break
                out[e] = own[owner] + ((0,) if before else (2,) if own[owner][0] == "p" else (1,))
        return out


# ========================
# Matching blocks, opcodes and segments
# ========================
def _kind(unit: str) -> str:
    """A unit's kind for letter comparisons: "word", "[" (a token), "(" (a badge), else "" (never compared)."""
    return "word" if WORD.match(unit) else unit[0] if len(unit) > 2 and unit[0] in "[(" else ""


def _word_blocks(old: str, new: str, a: tuple, b: tuple) -> list:
    """The matching blocks of the paragraph old[a] against new[b]: units, then letters for one word, token or badge
    into one alike of its kind ("(1GCD)" -> "(1CD)" keeps its letters, Turtle)."""
    ua, ub = units(old, *a), units(new, *b)
    out, pa, pb = [], 0, 0
    for i, j, size in SequenceMatcher(None, [old[s:e] for s, e in ua], [new[s:e] for s, e in ub],
                                      autojunk=False).get_matching_blocks():
        if i - pa == 1 and j - pb == 1:         # one unit changed into one: letters when two alike of a kind
            (s0, e0), (s1, e1) = ua[pa], ub[pb]
            x, y = old[s0:e0], new[s1:e1]
            if _kind(x) and _kind(x) == _kind(y):
                matcher = SequenceMatcher(None, x, y, autojunk=False)
                if matcher.ratio() >= REFINE_MIN:
                    out += [(s0 + p, s1 + q, n) for p, q, n in matcher.get_matching_blocks() if n]
        if size:
            out.append((ua[i][0], ub[j][0], ua[i + size - 1][1] - ua[i][0]))
        pa, pb = i + size, j + size
    return out


def _blocks(lay: Layout) -> list:
    old, new, found = lay.old, lay.new, []
    for i, j in lay.pairs:
        if i and j and old[lay.pa[i][0] - 1] == new[lay.pb[j][0] - 1]:
            found.append((lay.pa[i][0] - 1, lay.pb[j][0] - 1, 1))      # the break before it (both its pair's)
        found += _word_blocks(old, new, lay.pa[i], lay.pb[j])
    merged = []
    for i, j, n in found:
        if merged and merged[-1][0] + merged[-1][2] == i and merged[-1][1] + merged[-1][2] == j:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + n)
        else:
            merged.append((i, j, n))
    return merged + [(len(old), len(new), 0)]


def blocks(old: str, new: str) -> list:
    """
    difflib's get_matching_blocks() for an edit old -> new, by paragraphs then words (the module's rules): [(i, j, n)]
    in order, adjacent ones merged, ending with (len(old), len(new), 0).
    """
    return _blocks(Layout(old, new))


def _runs(groups: list, lo: int, hi: int) -> list:
    """[(start, end, group)] of groups[lo:hi] in runs of one group."""
    out = []
    for k in range(lo, hi):
        if out and out[-1][2] == groups[k]:
            out[-1] = (out[-1][0], k + 1, groups[k])
        else:
            out.append((k, k + 1, groups[k]))
    return out


def _first(x, y, lay: Layout) -> bool:
    """Whether the old run's group x goes before the new run's group y in a change (both orders kept): by the old
    paragraphs' order (a deleted one against a pair), the new ones' (a pair against an added one), the pairs' own; a
    deleted paragraph before an added one."""
    if x[:2] == y[:2]:
        return x[2] < y[2]
    if x[0] == "p" and y[0] == "p":
        return x[1] < y[1]
    if x[0] == "a" and y[0] == "p":
        return x[1] < lay.pairs[y[1]][0]
    if x[0] == "p" and y[0] == "b":
        return lay.pairs[x[1]][1] < y[1]
    return x[0] == "a"


def segments(old: str, new: str) -> list:
    """
    The edit's opcodes, each change split where its group changes (rule 4): [(tag, i1, i2, j1, j2, group)], tags
    equal, replace, delete, insert, in order over both texts.
    """
    lay = Layout(old, new)
    out, i, j = [], 0, 0
    for a, b, n in _blocks(lay):
        olds, news, ci, cj = _runs(lay.ga, i, a), _runs(lay.gb, j, b), i, j
        while olds or news:                     # in paragraph order, so typed text sits in its own paragraph
            x, y = (olds[0] if olds else None), (news[0] if news else None)
            if x and y and x[2] == y[2]:
                out.append(("replace", x[0], x[1], y[0], y[1], x[2]))
                ci, cj = x[1], y[1]
                olds.pop(0), news.pop(0)
            elif x and (not y or _first(x[2], y[2], lay)):
                out.append(("delete", x[0], x[1], cj, cj, x[2]))
                ci = x[1]
                olds.pop(0)
            else:
                out.append(("insert", ci, ci, y[0], y[1], y[2]))
                cj = y[1]
                news.pop(0)
        if n:
            out.append(("equal", a, a + n, b, b + n, lay.ga[a]))
        i, j = a + n, b + n
    return out


def opcodes(old: str, new: str) -> list:
    """difflib's get_opcodes() over segments(): [(tag, i1, i2, j1, j2)]; a change of two groups is two opcodes."""
    return [op[:5] for op in segments(old, new)]


def moves(old: str, new: str) -> list:
    """
    [((old start, end), (new start, end), blocks)]: each added paragraph that is a deleted one moved (MOVE_MIN alike,
    the likeliest first), paragraph texts only; blocks: the letters it kept, [(i, j, n)] from each paragraph's start.
    """
    lay = Layout(old, new)
    paired_a, paired_b = {i for i, _ in lay.pairs}, {j for _, j in lay.pairs}
    gone = [k for k, (s, e) in enumerate(lay.pa) if k not in paired_a and old[s:e].strip()]
    came = [k for k, (s, e) in enumerate(lay.pb) if k not in paired_b and new[s:e].strip()]
    scored = sorted(((likeness(old[slice(*lay.pa[a])], new[slice(*lay.pb[b])]), a, b) for a in gone for b in came),
                    reverse=True)
    out, used_a, used_b = [], set(), set()
    for score, a, b in scored:
        if score >= MOVE_MIN and a not in used_a and b not in used_b:
            (s0, e0), (t0, t1) = lay.pa[a], lay.pb[b]
            out.append(((s0, e0), (t0, t1), [(i - s0, j - t0, n) for i, j, n in _word_blocks(old, new, (s0, e0), (t0, t1))]))
            used_a.add(a)
            used_b.add(b)
    return sorted(out, key=lambda m: m[1])


# ========================
# Module Exports
# ========================
__all__ = [
    'BREAKS',
    'Layout',
    'blocks',
    'likeness',
    'moves',
    'opcodes',
    'pairs',
    'paragraphs',
    'segments',
    'units',
]
