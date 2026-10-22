"""
Write tests/fixtures/jsx_fx/ from Effect Layers - All Cards.jsx's 2026-10-20 trial copies (Convert overlays): per card the JSX
FX layer's name, bbox and ink (alpha > 0.5, a 1-bit PNG of its bbox), all test_fx_layers_match_the_jsx_output reads. The
copies (20+ MB each) live in $AC_JSX_FX_DIR, default ~/Documents/Tools/psd-research/fixtures/jsx-fx/:
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
    OUT.mkdir(exist_ok=True)
    index = {}
    for path in sorted(SOURCE.glob("*.psd")):
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
