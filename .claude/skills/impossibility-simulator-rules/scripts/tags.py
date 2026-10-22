"""Timing tag counts over the distinct cards. Usage: python3 -I tags.py <cards.json | loadpkl.py dump>"""
import collections, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # rows.py (-I drops the script's folder)
import rows

INNER = r'(?:-?[0-9]+/)?[A-Z0-9]+'  # 1CD, OYT, and 1/2CD, -1/1GCD
TAG = r'\((' + INNER + r')\)'
cards = rows.unique(rows.load(sys.argv[1]))
print('unique rows', len(cards))
strict = collections.Counter(); cardsper = collections.Counter(); bytype = collections.defaultdict(collections.Counter)
paren = collections.Counter(); tagged = 0
for c in cards:
    a = c.get('ability') or ''
    tags = re.findall(TAG, a)
    tagged += bool(tags)
    for t in tags: strict[t] += 1; bytype[t][c['type']] += 1
    for t in set(tags): cardsper[t] += 1
    for t in re.findall(r'\(([^()\n]{1,25})\)', a):
        if not re.fullmatch(INNER, t): paren[t] += 1
for t, n in strict.most_common(): print(f'{t:8s} {n:4d} occurrences in {cardsper[t]:4d} cards  {dict(bytype[t])}')
print(f'total {sum(strict.values())} occurrences of {len(strict)} tags in {tagged} cards')
print('\nOther parenthesised words (top 40):')
for t, n in paren.most_common(40): print(f'  ({t}) {n}')
