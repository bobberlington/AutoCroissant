import json, re, collections, sys
d = json.load(open(sys.argv[1]))
seen = set(); cards = []
for c in d['cards']:
    k = c.get('psd') or c.get('png')
    if c.get('dup') or k + '|' + c['name'] in seen: continue  # skip copy rows
    seen.add(k + '|' + c['name']); cards.append(c)
print('unique rows', len(cards))
strict = collections.Counter(); cardsper = collections.Counter(); bytype = collections.defaultdict(collections.Counter)
paren = collections.Counter()
for c in cards:
    a = c.get('ability') or ''
    tags = re.findall(r'\(([A-Z0-9]+)\)', a)
    for t in tags: strict[t] += 1; bytype[t][c['type']] += 1
    for t in set(tags): cardsper[t] += 1
    for t in re.findall(r'\(([^()\n]{1,25})\)', a):
        if not re.fullmatch(r'[A-Z0-9]+', t): paren[t] += 1
for t, n in strict.most_common(): print(f'{t:8s} {n:4d} occurrences in {cardsper[t]:4d} cards  {dict(bytype[t])}')
print('\nOther parenthesised words (top 40):')
for t, n in paren.most_common(40): print(f'  ({t}) {n}')
