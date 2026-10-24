"""
The one batch pass over the card repo's PNGs (owner, 2026-10-23; Photoshop's exports measured 4.8x too big): every card
PNG at a checkout's HEAD that oxipng makes more than png_export.RECOMPRESS_MIN (10%) and RECOMPRESS_BYTES (64 KiB) smaller
(our thresholds), losslessly (png_export.optimize, checked by same_png: the same header, pixels and chunks), is rewritten
in the working tree. From then on the push guards do the same for every PNG a push brings. Nothing is committed or
pushed: the landing commits the rewritten PNGs as AutoCroissant (the push guards skip the commit) with --message's text,
one Recompressed trailer per PNG (the website then shows no new image version). A file that is not HEAD's blob (edited,
or outside a sparse checkout) is skipped.

    python -m commands.png_recompress --clone <checkout> [--apply] [--workers N] [--report <file.json>] [--message <file>]

Without --apply nothing is written (the savings are measured). Needs pyoxipng (requirements-cards.txt skips it on 3.14:
`pip install pyoxipng==9.1.1` builds it there); no Discord, network or config.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from json import dump
from os import replace
from os.path import join
from subprocess import run
from typing import Optional

import commands.png_export as png_export
from commands.card_guards import RECOMPRESSED, card_png
from commands.github_api import blob_sha

# ========================
# Configuration
# ========================
WORKERS = 4                                     # processes (oxipng threads each one)
OUTCOMES = ("recompressed", "kept", "not the same", "skipped")


# ========================
# Helper Functions
# ========================
def card_pngs(clone: str) -> dict:
    """{path: blob sha} of the card PNGs at the checkout's HEAD (card_guards.card_png)."""
    raw = run(["git", "-C", clone, "ls-tree", "-r", "-z", "HEAD"], capture_output=True, check=True).stdout.decode("utf-8")
    out = {}
    for line in raw.split("\0"):
        meta, _, path = line.partition("\t")
        if path and meta.split(" ")[1] == "blob" and card_png(path):
            out[path] = meta.split(" ")[2]
    return out


def one(clone: str, path: str, blob: str, apply: bool) -> dict:
    """
    One PNG: {"path", "outcome" (OUTCOMES), "before", "after" (oxipng's size, written or not), "why"}; written in place
    (through a temporary file) when apply and recompressed.
    """
    row = {"path": path, "outcome": "skipped", "before": 0, "after": 0, "why": ""}
    full = join(clone, path)
    try:
        with open(full, "rb") as fh:
            data = fh.read()
    except OSError as error:
        return {**row, "why": f"unreadable: {error.strerror}"}
    if blob_sha(data) != blob:
        return {**row, "why": "not HEAD's blob"}
    out = png_export.optimize(data)
    row.update(before=len(data), after=len(data))
    if out is None:
        return {**row, "why": "oxipng gave nothing (not 8-bit, or no pyoxipng)"}
    row["after"] = len(out)
    if not png_export.same_png(data, out):
        return {**row, "outcome": "not the same"}
    if not png_export.worth(len(data), len(out)):
        return {**row, "outcome": "kept"}
    if apply:
        with open(full + ".recompress", "wb") as fh:
            fh.write(out)
        replace(full + ".recompress", full)
    return {**row, "outcome": "recompressed"}


def _one(item: tuple) -> dict:
    return one(*item)


def summary(rows: list) -> list:
    """The report's lines: per outcome the files and bytes, the total saved."""
    done = [r for r in rows if r["outcome"] == "recompressed"]
    before, after = sum(r["before"] for r in done), sum(r["after"] for r in done)
    every = sum(r["before"] for r in rows)
    lines = [f"{len(rows)} card PNGs, {every / 1e9:.2f} GB",
             f"recompressed: {len(done)} files, {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB "
             f"(saved {(before - after) / 1e9:.2f} GB, {100 * (before - after) / max(before, 1):.1f}% of them, "
             f"{100 * (before - after) / max(every, 1):.1f}% of all card PNGs)"]
    for outcome in OUTCOMES[1:]:
        found = [r for r in rows if r["outcome"] == outcome]
        if found:
            lines.append(f"{outcome}: {len(found)} files" + "".join(f"\n  {r['path']}: {r['why'] or r['after']}"
                                                                     for r in found[:20] if outcome != "kept"))
    return lines


def message(rows: list) -> str:
    """The batch commit's message: the subject with the count and sizes, one Recompressed trailer per PNG."""
    done = [r for r in rows if r["outcome"] == "recompressed"]
    before, after = sum(r["before"] for r in done), sum(r["after"] for r in done)
    subject = f"Card PNGs: {len(done)} recompressed losslessly ({before / 1e9:.2f} GB -> {after / 1e9:.2f} GB, the same pixels)"
    return subject + "\n\n" + "".join(f"{RECOMPRESSED}: {path}\n" for path in sorted(r["path"] for r in done))


# ========================
# Command line
# ========================
def main(args: Optional[list] = None) -> int:
    """The summary on stdout; exit 1 when oxipng's output of some PNG was not the same image (never written)."""
    cli = ArgumentParser(prog="python -m commands.png_recompress",
                         description="Recompress the card PNGs at a checkout's HEAD losslessly (oxipng) where that saves "
                                     "more than 10%; nothing committed.")
    cli.add_argument("--clone", required=True, metavar="DIR", help="a checkout of the card repo (its HEAD's PNGs)")
    cli.add_argument("--apply", action="store_true", help="rewrite the PNGs in the working tree (else measure only)")
    cli.add_argument("--workers", type=int, default=WORKERS, metavar="N", help=f"processes (default {WORKERS})")
    cli.add_argument("--report", default="", metavar="FILE", help="every PNG's outcome and sizes as JSON")
    cli.add_argument("--message", default="", metavar="FILE", help="the commit message for the recompressed PNGs")
    options = cli.parse_args(args)
    items = [(options.clone, path, blob, options.apply) for path, blob in sorted(card_pngs(options.clone).items())]
    with ProcessPoolExecutor(max(1, options.workers)) as pool:
        rows = list(pool.map(_one, items))
    if options.report:
        with open(options.report, "w", encoding="utf-8") as fh:
            dump(rows, fh, indent=1)
    if options.message:
        with open(options.message, "w", encoding="utf-8") as fh:
            fh.write(message(rows))
    print("\n".join(summary(rows)))
    return 1 if any(r["outcome"] == "not the same" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'card_pngs',
    'main',
    'message',
    'one',
    'summary',
]
