"""Card rows in cards.json's shape, from site/dist/data/cards.json or from a loadpkl.py dump of stats.pkl.
The other scripts import it: python3 -I <script>.py <cards.json | dump.json>"""
import json


def load(path):
    d = json.load(open(path))
    if isinstance(d.get('cards'), list):
        return sorted(d['cards'], key=order)
    rows = {}  # loadpkl.py dump: {name: {card_type, path, ...}}; rulebook pages live in rules.json instead
    for n, v in d.items():
        if v.get('card_type') == 'rulebook':
            continue
        rows[n] = {'name': v.get('display_name') or n, 'key': n, 'type': v.get('card_type'), 'psd': v.get('path'),
                   'png': None, 'ability': v.get('ability'), 'types': v.get('types') or [], 'stars': v.get('stars'),
                   'subtype': v.get('subtype'), 'series': v.get('series'), 'dup': v.get('duplicate_of'), 'dups': []}
    for r in rows.values():
        if r['dup'] in rows:
            rows[r['dup']]['dups'].append(r['name'])
    return sorted(rows.values(), key=order)


def order(c):  # same order for both shapes, so outputs compare line by line
    return (c.get('psd') or c.get('png') or '', c['name'])


def unique(rows):
    """Distinct cards: copy rows (`dup`) skipped, one row per (psd or png, name)."""
    seen, out = set(), []
    for c in rows:
        k = (c.get('psd') or c.get('png'), c['name'])
        if c.get('dup') or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out
