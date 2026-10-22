"""
Write tests/fixtures/jsx_fx/ (FX layer name, bbox, 1-bit ink PNG per card) from the JSX trial copies in $AC_JSX_FX_DIR:
    python tests/fixtures/make_jsx_fx.py
"""
import json
from os import environ
from os.path import expanduser
from pathlib import Path

from PIL import Image
from psd_tools import PSDImage

# ========================
# Configuration
# ========================
SOURCE = Path(expanduser(environ.get("AC_JSX_FX_DIR", "~/Documents/Tools/psd-research/fixtures/jsx-fx")))
OUT = Path(__file__).resolve().parent / "jsx_fx"
INDEX = OUT / "index.json"


def main() -> None:
    sources = sorted(SOURCE.glob("*.psd"))
    if not sources:
        raise SystemExit(f"no PSDs in {SOURCE}: set AC_JSX_FX_DIR; fixture left as is")
    OUT.mkdir(exist_ok=True)
    for stale in OUT.glob("*.png"):
        stale.unlink()
    index = {}
    for path in sources:
        psd = PSDImage.open(str(path))
        layer, = [layer for layer in psd.descendants() if " FX: " in layer.name]
        ink = layer.numpy("shape")[..., 0] > 0.5
        Image.fromarray(ink).convert("1").save(OUT / (path.stem + ".png"), optimize=True)
        index[path.stem.replace("__", "/") + ".psd"] = {"source": path.name, "name": layer.name, "bbox": list(layer.bbox),
                                                        "canvas": [psd.width, psd.height], "ink": int(ink.sum())}
    INDEX.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    print(f"{len(index)} cards -> {OUT}")


if __name__ == "__main__":
    main()
