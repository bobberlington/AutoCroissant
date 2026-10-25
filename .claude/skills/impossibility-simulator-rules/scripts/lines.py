"""Cooldown and uses tags line by line: what shares their line, what comes before (nU), tags not at a line's start,
(a/bCD) cards. Usage: python3 -I lines.py <cards.json | db/stats.json>"""
import collections, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # rows.py (-I drops the script's folder)
import rows

FAM = [('nCD', r'[0-9X]CD'), ('a/bCD', r'-?[0-9]+/[0-9X]CD'), ('nGCD', r'[0-9X]GCD'), ('a/bGCD', r'-?[0-9]+/[0-9X]GCD'),
       ('nU', r'[0-9X]U')]
TAG = re.compile(r'\(((?:-?[0-9]+/)?[A-Z0-9]+)\)')
fam = lambda t: next((f for f, p in FAM if re.fullmatch(p, t)), None)
cards = rows.unique(rows.load(sys.argv[1]))

with_tag = collections.defaultdict(collections.Counter)   # family -> lines, lines with (OYT), lines with (AAT)
before_u, u_cards, lead, mid, ab, global_oyt = collections.Counter(), set(), collections.Counter(), [], {}, []
for c in cards:
    for line in (c.get('ability') or '').split('\n'):
        tags = TAG.findall(line)
        fams = {fam(t) for t in tags} - {None}
        for f in fams:
            with_tag[f]['lines'] += 1
            for other in ('OYT', 'AAT'):
                with_tag[f][other] += other in tags
        if fams & {'nGCD', 'a/bGCD'} and 'OYT' in tags:
            global_oyt.append(f"{c['name']}: {line.strip()[:90]}")
        midline = False
        for m in TAG.finditer(line):
            t, pre = m.group(1), line[:m.start()]
            if fam(t) == 'nU':
                p = re.search(r'\(([^()]+)\)$', pre)
                before_u[p.group(1) if p else '-'] += 1
                u_cards.add(c['name'])
            elif fam(t) and not midline and not re.fullmatch(r'\s*-?\s*(\([^()]*\))*\s*', pre):  # text before it
                mid.append(f"{c['name']}: {line.strip()[:110]}")
                midline = True
        m = re.match(r'\s*((?:\([^()\n]{1,25}\))+)', line)
        if m and any(fam(t) for t in TAG.findall(m.group(1))):
            lead[re.sub(r'\d+(\.\d+)?', 'n', m.group(1))] += 1
        if 'a/bCD' in fams:
            ab[c['name']] = c

for f, _ in FAM:
    w = with_tag[f]
    print(f'{f:7s} lines {w["lines"]:4d}  with (OYT) {w["OYT"]:3d}  with (AAT) {w["AAT"]:3d}')
print(f'\nGlobal cooldown and (OYT) on one line ({len(global_oyt)}):')
for x in global_oyt: print('  ' + x)
print(f'\n(nU) on {len(u_cards)} cards; tag right before it:', dict(before_u))
print('\nLine starts with a cooldown or uses tag (numbers as n):')
for k, v in lead.most_common(20): print(f'  {v:4d} {k}')
print(f'\nCooldown tag after other text on the line ({len(mid)}):')
for x in mid: print('  ' + x)
both = sorted(n for n, c in ab.items() if re.search(r'\([0-9X]CD\)', c['ability']))
print(f'\n(a/bCD) on {len(ab)} cards {dict(collections.Counter(c["type"] for c in ab.values()))}: {", ".join(sorted(ab))}')
print(f'  of which also carry an (nCD) ({len(both)}): {", ".join(both)}')
