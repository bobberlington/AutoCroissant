import pickle, sys, collections, json
class Stub:
    def __setstate__(self, st):
        if isinstance(st, tuple) and len(st)==2: d, s = st; st = {**(d or {}), **(s or {})}
        self.__dict__.update(st if isinstance(st, dict) else {'_state': st})
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('commands') or mod.startswith('__main__'):
            return type(name, (Stub,), {})
        return super().find_class(mod, name)
with open(sys.argv[1], 'rb') as f: s = U(f).load()
print(type(s), len(s))
k = next(iter(s)); print(k, vars(s[k]))
C = collections.Counter(getattr(v, 'card_type', None) for v in s.values()); print(C)
def plain(x):  # stat lines (CardStats) as dicts
    if isinstance(x, Stub): return {a: plain(b) for a, b in vars(x).items()}
    if isinstance(x, (list, tuple)): return [plain(y) for y in x]
    if isinstance(x, dict): return {a: plain(b) for a, b in x.items()}
    return x
out = {n: {a: plain(b) for a, b in vars(v).items() if isinstance(b, (str, int, float, list, type(None), dict, Stub))} for n, v in s.items()}
json.dump(out, open(sys.argv[2], 'w'), default=str, indent=0)
