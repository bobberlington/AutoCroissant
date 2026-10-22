"""Status effect cards and how many other cards name each one. Usage: python3 -I mentions.py <cards.json | loadpkl.py dump>"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # rows.py (-I drops the script's folder)
import rows

d = rows.load(sys.argv[1])
se = [c for c in d if c['type'] in ('buff', 'debuff')]
others = {}
for c in d:
    if c['type'] in ('buff', 'debuff') or c.get('dup'): continue
    others[(c.get('psd') or c.get('png'), c['name'])] = c.get('ability') or ''
for s in sorted(se, key=lambda c: (c['type'], c.get('stars') or 0, c['name'])):
    n = s['name']
    pat = re.compile(r'(?<![A-Za-z])' + re.escape(n) + r'(?![A-Za-z])', re.I)
    k = sum(1 for a in others.values() if pat.search(a))
    print(f"{s['type']}|{s.get('stars')}|{n}|{k}")
