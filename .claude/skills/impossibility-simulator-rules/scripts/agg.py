"""Kinds by stars, series folders, subtypes, types and copies. Usage: python3 -I agg.py <cards.json | loadpkl.py dump>"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # rows.py (-I drops the script's folder)
import rows
cards = rows.load(sys.argv[1])
C=collections.Counter
cr=[c for c in cards if c['type']=='creature']
print('creatures by stars', sorted(C(c.get('stars') for c in cr).items(), key=str))
ser=C(); ss=collections.defaultdict(C)
for c in cr:
    f=(c.get('psd') or c['png']).split('/')[1]; ser[f]+=1; ss[f][c.get('stars')]+=1
for k,v in sorted(ser.items(), key=lambda x:-x[1]): print(f'  {k}: {v}  ', dict(sorted(ss[k].items())))
print('series field (creatures)', len(C(c.get('series') for c in cr)))
it=[c for c in cards if c['type']=='item']
print('items subtype', C(c.get('subtype') for c in it))
print('items stars', sorted(C(c.get('stars') for c in it).items(), key=str))
for t in ['field','nme','minion','aux item','buff','debuff','creatures']:
    x=[c for c in cards if c['type']==t]
    print(t, len(x), 'stars', sorted(C(c.get('stars') for c in x).items(), key=str), 'subtypes', C(c.get('subtype') for c in x))
mi=[c for c in cards if c['type']=='minion']
print('minion types', C(t for c in mi for t in c.get('types',[])))
print('all types used', C(t for c in cards for t in c.get('types',[])))
print('dups', sum(1 for c in cards if c.get('dups')), [ (c['name'],c['dups']) for c in cards if c.get('dups')][:60])
