import sys, glob, os
from psd_tools import PSDImage
def walk(g, vis=True, path=""):
    for l in g:
        v = vis and l.visible
        if l.is_group():
            yield from walk(l, v, path + l.name + "/")
        elif l.kind == "type":
            yield v, path + l.name, l.bbox, (l.text or "")
d = sys.argv[1]
for f in sys.argv[2:]:
    psd = PSDImage.open(os.path.join(d, f))
    print("#####", f)
    for v, p, bb, t in walk(psd):
        if not v:
            print("HIDDEN", p, bb, repr(t[:200]))
