"""
A sweep: one kind of the guards' silent pixel fixes (card_standards.PIXEL_FIXES, in order: badge widths, apostrophes,
old glyphs, and any kind registered there, such as badges) on every card of a clone's main that needs it, through the
guards' own code: the fix (card_standards.fix, only that kind, gated), the PNG re-export (png_export.export over the
PNGs that show the PSD, numbered ones too) and the guards' checks on the result (the PNG against the new PSD: png_check;
the standards: no new issue). One PSD rewrite per card. The rewritten PSDs and PNGs go into a worktree at their repo
paths; nothing is committed or pushed. The plan (beside the worktree) records the blobs each card rested on, so the
landing can leave out a card whose PSD or PNG moved since (never over a newer PSD): `moved` lists them.

    python -m commands.card_sweep run --kind apostrophes --clone ~/Desktop/TTSCardMaker --out <worktree>
                                      [--ref main] [--cards <names>] [--plan <file>] [--workers 4]
    python -m commands.card_sweep run --kind apostrophes,badges --clone ~/Desktop/TTSCardMaker --out <folder>
    python -m commands.card_sweep run --kind badges --whole --cards <names> ...   (a layer the splice can't take redrawn
                                      whole, when the fork draws it as stored and its lines stay: fit_badges' whole;
                                      --kind tags too: tags_to_badges' whole)
    python -m commands.card_sweep run --kind badge-widths ...   (relay_badges: card_guards.earlier_fonts, the icon font's
                                      eras before the ref's, are the widths a raster the fork drew may show)
    python -m commands.card_sweep run --kind alt-template ... [--template <psd>]   (cards onto the template: its layer-comp bars in
                                      place of exposure digit rows; the template: the file given, else the ref's)
    python -m commands.card_sweep run --kind normalise ... [--batch 3/16 | --batch T] [--record <json>] [--held-ref <ref>]
                                      [--widen <json>: {PSD path: [layer id]}, the box layers K2 saw re-break]
                                      (the big pass, 2026-11-07: card_standards.normalise in one rewrite per PSD, the
                                      cards, then MDW/Unfinished and MDW/Templates (batch T); checked by the drawing
                                      check (K1: every picture drawn alike, else the PNGs re-exported), the parse (K3)
                                      and the link step's own check (K4); each PSD's look read off its stored image
                                      first (psd_edit.look_facts) into the plan, and into the render record with --record)
    python -m commands.card_sweep moved --plan <file> --clone <clone> [--ref origin/main]

A sparse worktree must hold the paths it writes (git sparse-checkout add), or git does not see them.

Several kinds run as passes in the order given, one commit each: pass n writes into <out>/<kind> (repo paths) and its
plan beside <out>, and reads the cards an earlier pass rewrote from that pass's files, so its base blobs are what the
earlier commit lands (`moved` against main after that commit).
Each pass's commit carries the trailer `Card-sweep: <kind>` (changelog.SWEEP_TRAILER): the art and text changes it
alone makes are silent (no Discord post, still on the Changes page; card_data.mark_sweeps), in any push.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from json import dump, load
from os import cpu_count, makedirs, replace
from dataclasses import replace as replaced
from os.path import abspath, basename, dirname, exists, join
from shutil import rmtree
from subprocess import CalledProcessError, run as run_process
from sys import stderr
from tempfile import NamedTemporaryFile, mkdtemp
from traceback import format_exc
from typing import Optional

import numpy as np

import commands.card_standards as standards
import commands.png_export as png_export
import commands.psd_edit as psd_edit
from commands.card_edits import CardAuthors
from commands.card_guards import Job, LinkContext, Source, build_context, card_name, earlier_fonts, whole_tree_jobs
from commands.github_api import blob_sha

# ========================
# Configuration
# ========================
WORKERS = 4                                     # process pool at most (memory: one PSD each)
EARLIER_KINDS = ("badge widths", "old glyphs")  # the kinds that read the icon font's earlier versions
NORMALISE = "normalise"
MDW_ROOTS = ("MDW/Unfinished/", "MDW/Templates/")   # normalise's scope past the cards, in this order (batch T)
BATCHES = 16                                    # --batch k: the k-th of 16 runs of the cards (catalogue order), ~48
DRAWN_TOL = 2                                   # K1: a pixel past this many levels is a change (owner: near identical)
_worker: dict = {}


# ========================
# One card
# ========================
def _init(clone: str, tree: dict, kind: str, work: str, chained: Optional[dict] = None, whole: bool = False,
          earlier: tuple = (), template: str = "", catalogue: Optional[dict] = None, held: str = "",
          reuse: Optional[dict] = None, widen: Optional[dict] = None) -> None:
    source = Source(clone)
    if chained:                                 # blobs an earlier pass wrote: {sha: file}
        stored = source.blob
        source.blob = lambda sha: _read(chained[sha]) if sha in chained else stored(sha)
    source.checked = tree
    context, vocab = build_context(source, tree)
    fonts = partial(source.fonts_for)
    fonts.earlier = earlier                     # card_standards.relay_badges, redraw_old_glyphs
    fonts.cards = context.cards                 # relay_badges' notes: a break inside a card's name
    if kind == NORMALISE:                       # the other lanes' pieces (big pass), each optional until it lands
        fonts.catalogue = catalogue             # card_standards.normalise's link step (psd_links.link)
        if catalogue:
            fonts.links = LinkContext({path: sha for path, (_, sha) in tree.items()}, catalogue, source.blob)
        if held:                                # H2: the held cards' bars lined up (psd_links.held_bars)
            fonts.held = partial(_held_bars, folder=held)
    if kind == "alt template":                      # the template's bars: the given file, else the ref's
        if not template and standards.ALT_TEMPLATE in tree:
            template = join(work, "alt-template.psd")
            with NamedTemporaryFile(dir=work, delete=False) as fh:     # whole, then in place: other workers read it
                fh.write(source.blob(tree[standards.ALT_TEMPLATE][1]))
            replace(fh.name, template)
        fonts.template = template
    _worker.update(source=source, context=context, parser=vocab.parser(), kind=kind, work=work, whole=whole, fonts=fonts,
                   reuse=reuse or {}, widen=widen or {})


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _held_bars(psd, rel: str, folder: str) -> list:
    """H2's hook: psd_links.held_bars(psd, rel, folder) -> the bars lined up; [] until it lands."""
    links = standards._links()
    return links.held_bars(psd, rel, folder) if links is not None and hasattr(links, "held_bars") else []


def _codes(issues: list) -> set:
    return {(issue.code, item) for issue in issues if not issue.waived for item in (issue.items or ("",))}


def sweep_card(job) -> dict:
    """
    One card through the kind's fix, the PNG export and the checks.

    Returns:
        {"card", "psd", "state": skip | fixed | refused | error, "why", "base": {path: blob}, "files": {path: bytes},
         "layers", "notes" (fixed: what a reader should look at), "changed_px"}
    """
    source, kind = _worker["source"], _worker["kind"]
    out = {"card": card_name(job.psd), "psd": job.psd, "state": "skip", "why": "", "files": {}, "layers": [],
           "base": getattr(job, "base", None) or {job.psd: job.psd_new, **({job.png: job.png_new} if job.png else {}),
                                                  **job.numbered}}
    if kind == NORMALISE:
        return _normalise_card(job, out)
    try:
        data = source.blob(job.psd_new)
        pushed = png_export.open_psd(data)
        if not standards.pixel_kinds(pushed, (kind,)):
            return out
        fixed = standards.fix(data, job.psd, [], _worker["work"], _worker["parser"], _worker["fonts"], only=kind,
                              whole=_worker.get("whole", False))
        if not fixed.redrawn:
            return {**out, "state": "refused" if fixed.pixel_left else "skip",
                    "why": "; ".join(why for _, why in fixed.pixel_left)}
        out["layers"], out["notes"] = list(fixed.redrawn[kind]), fixed.notes
        if not job.png:
            return {**out, "state": "refused", "why": "no PNG"}
        pngs = {path: png_export.open_png(source.blob(sha)) for path, sha in {job.png: job.png_new, **job.numbered}.items()}
        for path, image in pngs.items():        # the PNGs must show the PSD, as the guards' redraw asks
            if png_export.png_check(pushed, image).get("ok") is not True:
                return {**out, "state": "refused", "why": f"{basename(path)} does not show the PSD"}
        after = png_export.open_psd(fixed.data)
        found = png_export.export(pushed, after, pngs[job.png], {p: i for p, i in pngs.items() if p != job.png})
        if found.outcome == "refused" or any(not d for d, _ in found.numbered.values()):
            why = found.why or next((why for d, why in found.numbered.values() if not d), "")
            return {**out, "state": "refused", "why": f"PNG not made: {why}"}
        files = {job.psd: fixed.data}
        if found.outcome == "re-exported":
            files[job.png] = found.data
            files.update({path: d for path, (d, _) in found.numbered.items()})
            out["changed_px"] = found.changed_px
        for path in pngs:                       # the guards' checks on the rewritten card
            image = png_export.open_png(files[path]) if path in files else pngs[path]
            if png_export.png_check(after, image).get("ok") is not True:
                return {**out, "state": "refused", "why": f"{basename(path)} misses the new PSD (parity)"}
        context = _worker["context"]
        gained = _codes(standards.check(after, job.psd, context)) - _codes(standards.check(pushed, job.psd, context))
        if gained:
            return {**out, "state": "refused", "why": "new standards issues: " + ", ".join(sorted(c for c, _ in gained))}
        return {**out, "state": "fixed", "files": files}
    except Exception as error:                  # one card never stops the sweep
        print(format_exc(), file=stderr)
        return {**out, "state": "error", "why": f"{type(error).__name__}: {error}"}


def _drawings(psd) -> list:
    """K1's renders of every picture (psd_edit.variant): the fork's, no ICC, every text in the gamma, uint8 RGBA."""
    tags = sorted(psd_edit.variants_of(psd))
    out = []
    for n in tags if len(tags) > 1 else [None]:
        with psd_edit.variant(psd, n):
            drawn = psd.composite(ignore_preview=True, apply_icc=False, text_preview=False)
            out.append(np.asarray(drawn.convert("RGBA")))
    return out


def drawing_check(old, new, fonts=None) -> dict:
    """K1: old and new drawn alike, every picture: {"px": pixels past DRAWN_TOL, "max", "same": no pixel past it,
    every type layer's box as it was and no text moved ("text": the layers that did)}."""
    a, b = _drawings(old), _drawings(new)
    if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        return {"px": -1, "max": 255, "same": False}
    diffs = [np.abs(x.astype(np.int16) - y.astype(np.int16)).max(axis=2) for x, y in zip(a, b)]
    boxes = [{layer.layer_id: tuple(layer.bbox) for layer in psd.descendants() if layer.kind == "type"}
             for psd in (old, new)]
    px, top = sum(int((d > DRAWN_TOL).sum()) for d in diffs), max(int(d.max()) for d in diffs)
    moved = _moved_text(old, new, fonts)
    return {"px": px, "max": top, "same": not px and boxes[0] == boxes[1] and not moved,
            **({"text": moved} if moved else {})}


def _moved_text(old, new, fonts) -> list:
    """K1's text: the layers that held sub or superscript whose glyphs the fork lays out elsewhere in new."""
    after = {layer.layer_id: layer for layer in new.descendants() if layer.kind == "type"}
    out = []
    for layer in standards.script_layers(old):
        try:
            faces = standards._layout_fonts(fonts, layer)
            same = layer.layer_id in after and psd_edit.same_places(
                psd_edit.glyph_places(layer, faces), psd_edit.glyph_places(after[layer.layer_id], faces),
                psd_edit.GLYPH_TOL)
        except Exception:                       # no layout: moved
            same = False
        if not same:
            out.append(layer.name.strip())
    return out


def _picture(psd, path: str, main: str) -> Optional[int]:
    """The picture a tagged PSD's PNG shows (main: 1, NameN.png: N), else None (as saved)."""
    tags = psd_edit.variants_of(psd)
    n = 1 if path == main else png_export.picture_number(path)
    return n if len(tags) > 1 and n in tags else None


def _png_ok(psd, image, path: str, main: str) -> bool:
    """png_check (R10) of a PNG against the picture it shows."""
    with psd_edit.variant(psd, _picture(psd, path, main)):
        return png_export.png_check(psd, image).get("ok") is True


def _fonts_with(fonts, **attrs):
    """The worker's fonts with what normalise reads off them for one card (widen: layer ids; keep_text: why)."""
    kept = partial(fonts.func, *fonts.args, **fonts.keywords)
    kept.__dict__.update(fonts.__dict__, **attrs)
    return kept


def _normalise_card(job, out: dict) -> dict:
    """
    One PSD through the big pass (section 5.2): its look read off the stored image (psd_edit.look_facts), the rewrite
    (card_standards.normalise through fix: the parse gated, K3), the drawing check (K1); past it the PNGs re-exported
    (png_export.export, render-based) and checked (png_check); the link step's check (K4: psd_links.check). The PNG
    check against the stored image before is kept (png_check), never a refusal. Refused, it is done again with its text
    left as it is (5.2 step 9). Hand notes for what it left.
    """
    source = _worker["source"]
    try:
        data = source.blob(job.psd_new)
        pushed = png_export.open_psd(data)
        pushed.repo_path = job.psd              # the link step judges its links from here
        if not standards.needs_normalise(pushed):
            return out
        pngs = {path: png_export.open_png(source.blob(sha)) for path, sha in
                ({job.png: job.png_new, **job.numbered} if job.png else {}).items()}
        facts = psd_edit.look_facts(pushed, pngs.get(job.png))
        saved = psd_edit.saved_view(pushed)     # the stored image shows that picture alone
        out.update(facts=facts, png_check={basename(p): _png_ok(pushed, image, p, job.png) for p, image in pngs.items()
                                           if _picture(pushed, p, job.png) in (None, saved)})
        fonts = _worker["fonts"]
        if job.psd in _worker["widen"]:         # K2's redo: these box layers take the box-width rule
            fonts = _fonts_with(fonts, widen=_worker["widen"][job.psd])
        done = _normalised(job, dict(out), data, pushed, pngs, facts, fonts)
        if done["state"] == "refused" and standards.script_layers(pushed):
            again = _normalised(job, dict(out), data, pushed, pngs, facts,
                                _fonts_with(fonts, keep_text=f"left as it is ({done['why']})"))
            if again["state"] == "fixed":
                return again
        return done
    except Exception as error:                  # one card never stops the sweep
        print(format_exc(), file=stderr)
        return {**out, "state": "error", "why": f"{type(error).__name__}: {error}"}


def _normalised(job, out: dict, data: bytes, pushed, pngs: dict, facts: dict, fonts) -> dict:
    """_normalise_card's rewrite and checks with these fonts (the worker's, or _fonts_with's)."""
    source = _worker["source"]
    fixed = standards.fix(data, job.psd, [], _worker["work"], _worker["parser"], fonts, only=NORMALISE)
    out.update(left=list(fixed.left), notes=list(fixed.notes))
    if not fixed.normalised:
        return {**out, "state": "refused" if fixed.pixel_left or fixed.notes else "skip",
                "why": "; ".join([why for _, why in fixed.pixel_left] + fixed.left)}
    out["layers"], out["redraws"] = fixed.normalised, fixed.redrawn.get(NORMALISE, [])
    after = psd_edit.use_record(png_export.open_psd(fixed.data), job.psd, facts)
    psd_edit.use_record(pushed, job.psd, facts)
    out["out"] = blob_sha(fixed.data)
    seen = _worker["reuse"].get(job.psd, {})
    drawn = out["k1"] = seen["k1"] if seen.get("out") == out["out"] else drawing_check(pushed, after, fonts)
    if drawn.get("text"):
        return {**out, "state": "refused", "why": "text moved: " + ", ".join(drawn["text"])}
    files, made = {job.psd: fixed.data, **fixed.files}, []
    if not drawn["same"] and job.png:           # past near identical: the PNGs drawn again from the new PSD
        found = png_export.export(pushed, after, pngs[job.png], {p: i for p, i in pngs.items() if p != job.png},
                                  job.png, fonts)
        if found.outcome == "refused" or any(d == b"" for d, _ in found.numbered.values()):
            why = found.why or next((why for d, why in found.numbered.values() if d == b""), "")
            return {**out, "state": "refused", "why": f"PNG not made: {why}"}
        if found.outcome == "re-exported":
            files[job.png], out["changed_px"] = found.data, found.changed_px
            made.append(job.png)
        made += [path for path, (d, _) in found.numbered.items() if d]     # a picture redrawn alone too
        files.update({path: found.numbered[path][0] for path in made if path != job.png})
        for path in sorted(set(pngs) | set(made)):
            image = png_export.open_png(files[path]) if path in files else pngs[path]
            if not _png_ok(after, image, path, job.png):
                return {**out, "state": "refused", "why": f"{basename(path)} misses the new PSD (parity)"}
    elif not drawn["same"] and drawn["px"] < 0:
        return {**out, "state": "refused", "why": "the drawing changed size"}
    links = standards._links()
    problems = links.check(after, job.psd, files, source.checked) if links and hasattr(links, "check") else []
    if problems:
        return {**out, "state": "refused", "why": "links: " + "; ".join(problems)}
    out["pngs"] = sorted(made)                  # the card's PNGs written (the link step's files aside)
    return {**out, "state": "fixed", "files": files}


def normalise_jobs(tree: dict, names: Optional[set], batch: str = "") -> list:
    """The big pass's PSDs in catalogue order (the cards by repo path, then MDW_ROOTS'), names only (card keys) when
    given; batch "k/n": the k-th of n even runs of the cards ("k": of BATCHES), "T": the MDW PSDs alone."""
    cards = whole_tree_jobs(tree, "sweep", names or None)
    extra = [Job(path, "whole", "", tree[path][1], login="sweep") for root in MDW_ROOTS for path in sorted(tree)
             if path.startswith(root) and path.lower().endswith(".psd")
             and (not names or CardAuthors.key(card_name(path)) in names)]
    if not batch:
        return cards + extra
    if batch.upper() == "T":
        return extra
    k, n = (int(part) for part in (batch.split("/") if "/" in batch else (batch, BATCHES)))
    if not 1 <= k <= n:
        raise ValueError(f"--batch {batch}: k from 1 to n")
    size = -(-len(cards) // n)
    return cards[(k - 1) * size:k * size]


def write_record(plan: dict, path: str) -> int:
    """The plan's look facts merged into the render record at path (psd_edit.RENDER_RECORD's form); how many."""
    record = {}
    if exists(path):
        with open(path, encoding="utf-8") as fh:
            record = load(fh)
    found = {entry["psd"]: entry["facts"] for entry in plan["fixed"] + plan["refused"] if entry.get("facts")}
    record.update(found)
    with open(path, "w", encoding="utf-8") as fh:
        dump(dict(sorted(record.items())), fh, indent=0, sort_keys=True)
        fh.write("\n")
    return len(found)


# ========================
# The sweep
# ========================
def kinds_of(text: str) -> list:
    """The kinds named (comma-separated, - for space), in order; ValueError on an unknown one."""
    kinds = list(dict.fromkeys(k.strip().replace("-", " ") for k in text.split(",") if k.strip()))
    for kind in kinds:
        if kind not in standards.PIXEL_FIXES:
            raise ValueError(f"{kind!r}: not one of {', '.join(standards.PIXEL_FIXES)}")
    if not kinds:
        raise ValueError("no kind")
    return kinds


def run(clone: str, kind: str, out_dir: str, ref: str = "main", cards: str = "", plan_path: str = "",
        workers: int = WORKERS, whole: bool = False, template: str = "", batch: str = "", record: str = "",
        held_ref: str = "", held_base: str = "", held: str = "", catalogue: str = "", reuse: str = "",
        widen: str = "") -> list:
    """
    Sweep the clone's ref, one pass per kind: one kind's files written into out_dir, several kinds' into
    out_dir/<kind>, each later pass on the earlier passes' files; each plan to plan_path (one kind) or beside out_dir.
    normalise alone: batch (normalise_jobs), record (write_record), held_ref and held_base (a held card unmoved on ref
    since held_base is swept from held_ref's version; held: their bar documents' folder, H2), catalogue (the link
    step's file, psd_links.catalogue: built or checked again before the pool), reuse (an earlier plan, S-dry's: K1's
    verdict kept for a PSD whose output is the same blob, at the same clean AC and fork commits), widen (a file of
    {PSD path: [layer id]}: those box layers take the box-width rule, K2's redo).

    Returns:
        The plans: [{"kind", "ref", "commit", "fixed": [...], "refused": [...], "errors": [...], "skipped": n}]
    """
    kinds = kinds_of(kind)
    if whole and kinds not in (["badges"], ["tags"]):
        raise ValueError("--whole is for badges or tags alone")
    if (batch or record or held_ref or catalogue or reuse or widen) and kinds != [NORMALISE]:
        raise ValueError("--batch, --record, --held-ref, --catalogue, --reuse and --widen are for normalise alone")
    if bool(held_ref) != bool(held_base):       # without the base, a card pushed meanwhile would take the held blob
        raise ValueError("--held-ref and --held-base go together")
    source = Source(clone)
    commit = source.commit(ref)
    if not commit:
        raise ValueError(f"{ref}: not in {clone}")
    tree = dict(source.tree(commit))
    names = {CardAuthors.key(c) for c in cards.split(",") if c.strip()}
    chained: dict = {}                          # sha -> file, the earlier passes' writes
    plans = []
    for kind in kinds:
        slug = kind.replace(" ", "-")
        target = abspath(out_dir if len(kinds) == 1 else join(out_dir, slug))
        jobs = normalise_jobs(tree, names, batch) if kind == NORMALISE else whole_tree_jobs(tree, "sweep", names or None)
        work = mkdtemp(prefix="sweep-")
        plan = {"kind": kind, "ref": ref, "commit": commit, "fixed": [], "refused": [], "errors": [], "skipped": 0}
        try:
            earlier = tuple(earlier_fonts(source, commit, tree, work)) if kind in EARLIER_KINDS else ()
            found, seen, boxes = None, {}, {}
            if kind == NORMALISE:
                jobs, plan["held"] = _held_versions(source, tree, jobs, held_ref, held_base)
                links = standards._links()
                found = links.catalogue(source, tree, jobs, catalogue) if links and hasattr(links, "catalogue") else None
                plan["commits"] = code_commits()
                seen = _reused(reuse, plan["commits"])
                if widen:
                    with open(widen, encoding="utf-8") as fh:
                        boxes = load(fh)
            _sweep(plan, jobs, clone, tree, kind, work, target, workers, chained, whole, earlier, template, found, held,
                   seen, boxes)
        finally:
            rmtree(work, ignore_errors=True)
        for entry in plan["fixed"]:             # the next pass reads these
            for path, sha in entry["writes"].items():
                tree[path] = (tree.get(path, ("100644", ""))[0], sha)
                chained[sha] = join(target, path)
        path = (plan_path if len(kinds) == 1 else "") or join(dirname(abspath(out_dir)), f"card-sweep-{slug}.json")
        with open(path, "w", encoding="utf-8") as fh:
            dump(plan, fh, indent=1)
        plan["path"] = path
        if record and kind == NORMALISE:
            plan["recorded"] = write_record(plan, record)
        plans.append(plan)
    return plans


def _held_versions(source: Source, tree: dict, jobs: list, held_ref: str, held_base: str) -> tuple:
    """(jobs, held): a held card (its PSD differs on held_ref) whose PSD on the swept tree is still held_base's is swept
    from held_ref's PSD and PNGs (the plan's base stays the swept tree's); one that moved since keeps its own version
    (owner 11-07: rebuilt from the version someone pushed). held: {psd: "held" or "moved"}."""
    if not held_ref:
        return jobs, {}
    theirs = source.tree(source.commit(held_ref) or held_ref)
    start = source.tree(source.commit(held_base) or held_base)
    out, held = [], {}
    for job in jobs:
        mine = theirs.get(job.psd, ("", ""))[1]
        if not mine or mine == job.psd_new:
            out.append(job)
            continue
        if start.get(job.psd, ("", ""))[1] != job.psd_new:
            held[job.psd] = "moved"
            out.append(job)
            continue
        held[job.psd] = "held"
        numbered = {p: theirs.get(p, ("", sha))[1] for p, sha in job.numbered.items()}
        base = {job.psd: job.psd_new, **({job.png: job.png_new} if job.png else {}), **job.numbered}
        out.append(_with_base(job, base, psd_new=mine, numbered=numbered,
                               png_new=theirs.get(job.png, ("", job.png_new))[1] if job.png else ""))
    return out, held


def _with_base(job, base: dict, **changes):
    """A Job with changes, carrying the swept tree's blobs (base) for the plan."""
    found = replaced(job, **changes)
    found.base = base
    return found


def code_commits() -> dict:
    """{"ac", "fork"}: the commits this code runs at, "" for a tree with changes (S-dry's verdicts are reused only at
    clean, equal commits)."""
    import psd_tools
    out = {}
    for name, folder in (("ac", dirname(dirname(abspath(__file__)))), ("fork", dirname(abspath(psd_tools.__file__)))):
        try:
            head = run_process(["git", "-C", folder, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            dirty = run_process(["git", "-C", folder, "status", "--porcelain", "--", "."], capture_output=True, text=True,
                                check=True).stdout.strip()
            out[name] = "" if dirty else head.stdout.strip()
        except (OSError, CalledProcessError):
            out[name] = ""
    return out


def _reused(path: str, commits: dict) -> dict:
    """{psd: {"out", "k1"}} of an earlier normalise plan run at these clean commits, else {}."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        plan = load(fh)
    if not all(commits.values()) or plan.get("commits") != commits:
        return {}
    return {e["psd"]: {"out": e["out"], "k1": e["k1"]} for e in plan["fixed"] + plan["refused"]
            if e.get("out") and e.get("k1")}


def _sweep(plan: dict, jobs: list, clone: str, tree: dict, kind: str, work: str, out_dir: str, workers: int,
           chained: Optional[dict] = None, whole: bool = False, earlier: tuple = (), template: str = "",
           catalogue: Optional[dict] = None, held: str = "", reuse: Optional[dict] = None,
           widen: Optional[dict] = None) -> None:
    with ProcessPoolExecutor(max(1, min(workers, WORKERS, cpu_count() or 1)), initializer=_init,
                             initargs=(clone, tree, kind, work, chained, whole, earlier, template, catalogue,
                                       held, reuse, widen)) as pool:
        for done in pool.map(sweep_card, jobs, chunksize=4 if kind != NORMALISE else 1):
            state = done.pop("state")
            if state == "skip":
                plan["skipped"] += 1
                continue
            if state != "fixed":
                keep = [k for k in done if k not in ("files", "layers", "base")] if kind == NORMALISE \
                    else ["card", "psd", "why"]
                plan["refused" if state == "refused" else "errors"].append({k: done[k] for k in keep})
                continue
            files = done.pop("files")
            for path, data in files.items():
                if data is None:                # a deletion: the plan says it
                    continue
                target = join(out_dir, path)
                makedirs(dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(data)
            plan["fixed"].append({**{k: v for k, v in done.items() if k != "why"},
                                  "writes": {path: blob_sha(data) for path, data in files.items() if data is not None},
                                  **({"deletes": sorted(p for p, d in files.items() if d is None)}
                                     if any(d is None for d in files.values()) else {})})


def moved(plan_path: str, clone: str, ref: str = "origin/main") -> list:
    """The planned cards whose PSD or PNGs moved on ref since the sweep read them: leave them out of the landing."""
    with open(plan_path, encoding="utf-8") as fh:
        plan = load(fh)
    source = Source(clone)
    tree = source.tree(source.commit(ref) or ref)
    return [entry["psd"] for entry in plan["fixed"]
            if any(tree.get(path, ("", ""))[1] != blob for path, blob in entry["base"].items())]


def summary(plan: dict) -> str:
    lines = [f"{plan['kind']} sweep of {plan['ref']} ({plan['commit'][:8]}): {len(plan['fixed'])} cards fixed, "
             f"{len(plan['refused'])} refused, {len(plan['errors'])} errors, {plan['skipped']} not affected"]
    lines += [f"  fixed {e['psd']}: {', '.join(e['layers'])}" for e in plan["fixed"]]
    lines += [f"  look at {e['psd']}: {note}" for e in plan["fixed"] for note in e.get("notes", ())]
    lines += [f"  PNGs re-exported {e['psd']}: {e['k1']['px']:,} px past {DRAWN_TOL}" for e in plan["fixed"]
              if e.get("k1") and not e["k1"]["same"]]
    for e in plan["fixed"]:                     # 5.2 step 1: converted all the same, its PNG kept
        off = [name for name, ok in e.get("png_check", {}).items() if not ok]
        lines += [f"  PNG off its stored image, kept {e['psd']}: {', '.join(off)}"] if off else []
    lines += [f"  render record: {plan['recorded']} PSDs"] if "recorded" in plan else []
    lines += [f"  refused {e['psd']}: {e['why']}" for e in plan["refused"]]
    lines += [f"  error {e['psd']}: {e['why']}" for e in plan["errors"]]
    return "\n".join(lines + ([f"plan: {plan['path']}"] if plan.get("path") else []))


def main(args: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.card_sweep", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", help="sweep kinds of fix into a worktree, one pass each")
    one.add_argument("--kind", required=True, help="comma-separated, run in order: " +
                     ", ".join(k.replace(" ", "-") for k in standards.PIXEL_FIXES))
    one.add_argument("--clone", required=True)
    one.add_argument("--out", required=True, help="the worktree the files are written into")
    one.add_argument("--ref", default="main")
    one.add_argument("--cards", default="", help="comma-separated card names (default: every card)")
    one.add_argument("--plan", default="")
    one.add_argument("--workers", type=int, default=WORKERS)
    one.add_argument("--whole", action="store_true", help="badges, tags: redraw a layer whole when the splice can't")
    one.add_argument("--template", default="", help="alt-template: the template PSD (default: the ref's)")
    one.add_argument("--batch", default="", help="normalise: k/n (the k-th of n runs of the cards) or T (MDW)")
    one.add_argument("--record", default="", help="normalise: merge each PSD's look facts into this render record")
    one.add_argument("--held-ref", default="", help="normalise: the held cards' branch (work/cards-held)")
    one.add_argument("--held-base", default="", help="normalise: the commit the held branch started from")
    one.add_argument("--held", default="", help="normalise: the held cards' bar documents (H2)")
    one.add_argument("--catalogue", default="", help="normalise: the link step's catalogue file")
    one.add_argument("--reuse", default="", help="normalise: an earlier plan whose K1 verdicts hold for the same output")
    one.add_argument("--widen", default="", help="normalise: {PSD path: [layer id]} that take the box-width rule (K2)")
    held = sub.add_parser("moved", help="the planned cards whose files moved since")
    held.add_argument("--plan", required=True)
    held.add_argument("--clone", required=True)
    held.add_argument("--ref", default="origin/main")
    opts = parser.parse_args(args)
    if opts.command == "moved":
        for path in moved(opts.plan, opts.clone, opts.ref):
            print(path)
        return 0
    plans = run(opts.clone, opts.kind, opts.out, opts.ref, opts.cards, opts.plan, opts.workers, opts.whole, opts.template,
                opts.batch, opts.record, opts.held_ref, opts.held_base, opts.held, opts.catalogue, opts.reuse,
                opts.widen)
    print("\n".join(summary(plan) for plan in plans))
    return 1 if any(plan["errors"] for plan in plans) else 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'drawing_check',
    'kinds_of',
    'normalise_jobs',
    'moved',
    'run',
    'summary',
    'sweep_card',
    'write_record',
]
