import json, collections, sys, re
d=json.load(open(sys.argv[1]))
cards=d['cards']
print('total cards', len(cards))
print('keys', [k for k in d if k!='cards'])
print('types', collections.Counter(c['type'] for c in cards))
C=collections.Counter
def top(p): return p.split('/')
# by folder structure
fold=C()
for c in cards:
    p=(c.get('psd') or c.get('png')).split('/')
    fold['/'.join(p[:-1])]+=1
for k,v in sorted(fold.items()): print(f'{v:4d} {k}')
