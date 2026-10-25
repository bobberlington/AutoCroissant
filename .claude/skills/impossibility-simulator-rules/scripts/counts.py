"""Rows by kind and by folder. Usage: python3 -I counts.py <cards.json | db/stats.json>"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # rows.py (-I drops the script's folder)
import rows

cards = rows.load(sys.argv[1])
print('total cards', len(cards), '(copy rows', sum(1 for c in cards if c.get('dup')), ')')
print('types', collections.Counter(c['type'] for c in cards))
fold = collections.Counter('/'.join((c.get('psd') or c.get('png')).split('/')[:-1]) for c in cards)
for k, v in sorted(fold.items()): print(f'{v:4d} {k}')
