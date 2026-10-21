import sys, os
from psd_tools import PSDImage
def walk(g, vis=True, path=""):
    for l in g:
        v = vis and l.visible
        if l.is_group():
            yield from walk(l, v, path + l.name + "/")
        else:
            yield v, l.kind, path + l.name, l.bbox, (l.text if l.kind=="type" else ""), l.opacity
psd = PSDImage.open(sys.argv[1])
print(psd.size)
for v, k, p, bb, t, o in walk(psd):
    if k == "type" or len(sys.argv) > 2:
        print("V" if v else "H", k, o, p, bb, repr((t or "")[:120]))
