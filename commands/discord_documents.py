"""
Card timelines: the card documents posted on Discord before git (docx card sheets, Paint.NET files), as pictures dated
by their post (docs/Future/CARD_TIMELINE.md; gap round 5). discord_archive --files stores the files; then:

    python -m commands.discord_documents build [--root <dir>]

build   derived/documents/pictures.jsonl: one row a picture {message, channel_id, doc (its attachment id), n, file,
        sha256, w, h, how}, each a post picture to the matcher (discord_match.load_posts, kind 'document'):
          docx  page 1 drawn by QuickLook (qlmanage, run in ~/std-work-1107 so no privacy prompt), cut into the card
                cells by eye (derived/documents/boxes.json: {attachment id: {area: [x0, y0, x1, y1], cells: n,
                skip: [cell numbers]}}: equal columns of the card area); only documents listed there
          pdn   the composite Paint.NET keeps in its header (a 256 px thumbnail)
        Pictures go to derived/documents/pictures/ (outside every repo). No text is kept: the documents' change lists
        are read in memory by discord_facts.
"""
from argparse import ArgumentParser
from base64 import b64decode
from hashlib import sha256
from io import BytesIO
from json import dumps, loads
from os import listdir, makedirs, remove
from os.path import basename, exists, expanduser, join
from re import search
from shutil import copyfile
from subprocess import run
from typing import Callable

from commands import discord_match as dm

OUT = ("derived", "documents")
BOXES = ("derived", "documents", "boxes.json")
WORK = "~/std-work-1107/tl-docs"            # QuickLook's in/out (no privacy prompt there)
PAGE = 2000                                 # QuickLook's page width (px)
QL_SECONDS = 120


def stored_documents(root: str) -> list:
    """(message record, attachment) of every stored attachment that is no picture: docx, pdn, pdf, txt, zip."""
    records, _ = dm.load_posts(root)
    found = []
    for record in records.values():
        for a in record.get("attachments") or []:
            if a.get("file") and a.get("sha256") and a["file"].rsplit(".", 1)[-1].lower() in ("docx", "pdn"):
                found.append((record, a))
    return sorted(found, key=lambda x: int(x[0]["id"]))


def pdn_thumb(data: bytes):
    """The composite thumbnail in a Paint.NET file's XML header (PDN3), or None."""
    from PIL import Image
    if data[:4] != b"PDN3":
        return None
    size = int.from_bytes(data[4:7], "little")
    head = data[7:7 + size].decode("utf-8", "replace")
    found = search(r'<thumb png="([^"]+)"', head)
    if not found:
        return None
    with Image.open(BytesIO(b64decode(found.group(1)))) as im:
        return im.convert("RGBA")


def quicklook(path: str, work: str):
    """Page 1 of a document drawn by QuickLook, or None."""
    from PIL import Image
    name = basename(path)
    for sub in ("in", "out"):
        makedirs(join(work, sub), exist_ok=True)
    copyfile(path, join(work, "in", name))
    try:
        run(["qlmanage", "-t", "-s", str(PAGE), "-o", join(work, "out"), join(work, "in", name)], cwd=work,
            capture_output=True, timeout=QL_SECONDS)
        page = join(work, "out", name + ".png")
        if not exists(page):
            return None
        with Image.open(page) as im:
            return im.convert("RGB")
    finally:
        for p in (join(work, "in", name), join(work, "out", name + ".png")):
            if exists(p):
                remove(p)


def save(img, folder: str) -> tuple:
    """(file name, sha256) of a picture saved as PNG by its hash."""
    buf = BytesIO()
    img.save(buf, "PNG")
    data = buf.getvalue()
    digest = sha256(data).hexdigest()
    name = f"{digest}.png"
    if not exists(join(folder, name)):
        with open(join(folder, name), "wb") as f:
            f.write(data)
    return name, digest


def build(root: str, log: Callable = print) -> int:
    root = dm.check_root(root)
    out = join(root, *OUT)
    pictures = join(out, "pictures")
    makedirs(pictures, exist_ok=True)
    boxes = dm.read_json(join(root, *BOXES), {}) or {}
    work = expanduser(WORK)
    rows = []
    for record, a in stored_documents(root):
        path, ext = join(root, a["file"]), a["file"].rsplit(".", 1)[-1].lower()
        base = {"message": record["id"], "channel_id": record.get("channel_id"), "doc": a["id"],
                "filename": a.get("filename")}
        if ext == "pdn":
            with open(path, "rb") as f:
                img = pdn_thumb(f.read())
            if img is not None:
                name, digest = save(img, pictures)
                rows.append({**base, "n": 1, "file": f"{'/'.join(OUT)}/pictures/{name}", "sha256": digest,
                             "w": img.width, "h": img.height, "how": "Paint.NET thumbnail"})
            continue
        box = boxes.get(a["id"])
        if not box:
            continue
        page = quicklook(path, work)
        if page is None:
            log(f"{a['id']}: QuickLook drew nothing")
            continue
        x0, y0, x1, y1 = box["area"]
        n = box["cells"]
        for i in range(n):
            if i + 1 in (box.get("skip") or ()):
                continue
            cell = page.crop((round(x0 + i * (x1 - x0) / n), y0, round(x0 + (i + 1) * (x1 - x0) / n), y1))
            name, digest = save(cell, pictures)
            rows.append({**base, "n": i + 1, "file": f"{'/'.join(OUT)}/pictures/{name}", "sha256": digest,
                         "w": cell.width, "h": cell.height, "how": "QuickLook page 1, a card cell"})
    keep = {basename(r["file"]) for r in rows}
    for name in [n for n in listdir(pictures) if n not in keep]:
        remove(join(pictures, name))
    dm.write_text(join(out, "pictures.jsonl"), "".join(dumps(r, ensure_ascii=False) + "\n" for r in rows))
    log(f"documents: {len(rows)} pictures from {len({r['doc'] for r in rows})} documents")
    return len(rows)


def document_pictures(root: str) -> list:
    """The documents' pictures (build's rows), for discord_match.load_posts."""
    path = join(root, *OUT, "pictures.jsonl")
    if not exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    parser = ArgumentParser(prog="python -m commands.discord_documents")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--root", default=dm.DEFAULT_ROOT)
    args = parser.parse_args(argv)
    build(expanduser(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
