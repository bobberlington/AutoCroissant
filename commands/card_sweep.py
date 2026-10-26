"""
A sweep: one kind of the guards' silent pixel fixes (card_standards.PIXEL_FIXES: apostrophes, old glyphs, and any kind
registered there, such as badges) on every card of a clone's main that needs it, through the guards' own code: the fix
(card_standards.fix, only that kind, gated), the PNG re-export (png_export.export over the PNGs that show the PSD,
numbered ones too) and the guards' checks on the result (the PNG against the new PSD: png_check; the standards: no new
issue). One PSD rewrite per card. The rewritten PSDs and PNGs go into a worktree at their repo paths; nothing is
committed or pushed. The plan (beside the worktree) records the blobs each card rested on, so the landing can leave out
a card whose PSD or PNG moved since (never over a newer PSD): `moved` lists them.

    python -m commands.card_sweep run --kind apostrophes --clone ~/Desktop/TTSCardMaker --out <worktree>
                                      [--ref main] [--cards <names>] [--plan <file>] [--workers 4]
    python -m commands.card_sweep run --kind apostrophes,badges --clone ~/Desktop/TTSCardMaker --out <folder>
    python -m commands.card_sweep run --kind badges --whole --cards <names> ...   (a layer the splice can't take redrawn
                                      whole, when the fork draws it as stored and its lines stay: fit_badges' whole)
    python -m commands.card_sweep moved --plan <file> --clone <clone> [--ref origin/main]

A sparse worktree must hold the paths it writes (git sparse-checkout add), or git does not see them.

Several kinds run as passes in the order given, one commit each: pass n writes into <out>/<kind> (repo paths) and its
plan beside <out>, and reads the cards an earlier pass rewrote from that pass's files, so its base blobs are what the
earlier commit lands (`moved` against main after that commit).
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from json import dump, load
from os import cpu_count, makedirs
from os.path import abspath, basename, dirname, join
from shutil import rmtree
from sys import stderr
from tempfile import mkdtemp
from traceback import format_exc
from typing import Optional

import commands.card_standards as standards
import commands.png_export as png_export
from commands.card_edits import CardAuthors
from commands.card_guards import Source, build_context, card_name, whole_tree_jobs
from commands.github_api import blob_sha

# ========================
# Configuration
# ========================
WORKERS = 4                                     # process pool at most (memory: one PSD each)
_worker: dict = {}


# ========================
# One card
# ========================
def _init(clone: str, tree: dict, kind: str, work: str, chained: Optional[dict] = None, whole: bool = False) -> None:
    source = Source(clone)
    if chained:                                 # blobs an earlier pass wrote: {sha: file}
        stored = source.blob
        source.blob = lambda sha: _read(chained[sha]) if sha in chained else stored(sha)
    source.checked = tree
    context, vocab = build_context(source, tree)
    _worker.update(source=source, context=context, parser=vocab.parser(), kind=kind, work=work, whole=whole)


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _codes(issues: list) -> set:
    return {(issue.code, item) for issue in issues if not issue.waived for item in (issue.items or ("",))}


def sweep_card(job) -> dict:
    """
    One card through the kind's fix, the PNG export and the checks.

    Returns:
        {"card", "psd", "state": skip | fixed | refused | error, "why", "base": {path: blob}, "files": {path: bytes},
         "layers", "changed_px"}
    """
    source, kind = _worker["source"], _worker["kind"]
    out = {"card": card_name(job.psd), "psd": job.psd, "state": "skip", "why": "", "files": {}, "layers": [],
           "base": {job.psd: job.psd_new, **({job.png: job.png_new} if job.png else {}), **job.numbered}}
    try:
        data = source.blob(job.psd_new)
        pushed = png_export.open_psd(data)
        if not standards.pixel_kinds(pushed, (kind,)):
            return out
        fixed = standards.fix(data, job.psd, [], _worker["work"], _worker["parser"], source.fonts_for, only=kind,
                              whole=_worker.get("whole", False))
        if not fixed.redrawn:
            return {**out, "state": "refused" if fixed.pixel_left else "skip",
                    "why": "; ".join(why for _, why in fixed.pixel_left)}
        out["layers"] = fixed.redrawn[kind]
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
        workers: int = WORKERS, whole: bool = False) -> list:
    """
    Sweep the clone's ref, one pass per kind: one kind's files written into out_dir, several kinds' into
    out_dir/<kind>, each later pass on the earlier passes' files; each plan to plan_path (one kind) or beside out_dir.

    Returns:
        The plans: [{"kind", "ref", "commit", "fixed": [...], "refused": [...], "errors": [...], "skipped": n}]
    """
    kinds = kinds_of(kind)
    if whole and kinds != ["badges"]:
        raise ValueError("--whole is for badges alone")
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
        jobs = whole_tree_jobs(tree, "sweep", names or None)
        work = mkdtemp(prefix="sweep-")
        plan = {"kind": kind, "ref": ref, "commit": commit, "fixed": [], "refused": [], "errors": [], "skipped": 0}
        try:
            _sweep(plan, jobs, clone, tree, kind, work, target, workers, chained, whole)
        finally:
            rmtree(work, ignore_errors=True)
        for entry in plan["fixed"]:             # the next pass reads these
            for path, sha in entry["writes"].items():
                tree[path] = (tree[path][0], sha)
                chained[sha] = join(target, path)
        path = (plan_path if len(kinds) == 1 else "") or join(dirname(abspath(out_dir)), f"card-sweep-{slug}.json")
        with open(path, "w", encoding="utf-8") as fh:
            dump(plan, fh, indent=1)
        plan["path"] = path
        plans.append(plan)
    return plans


def _sweep(plan: dict, jobs: list, clone: str, tree: dict, kind: str, work: str, out_dir: str, workers: int,
           chained: Optional[dict] = None, whole: bool = False) -> None:
    with ProcessPoolExecutor(max(1, min(workers, WORKERS, cpu_count() or 1)), initializer=_init,
                             initargs=(clone, tree, kind, work, chained, whole)) as pool:
        for done in pool.map(sweep_card, jobs, chunksize=4):
            state = done.pop("state")
            if state == "skip":
                plan["skipped"] += 1
                continue
            if state != "fixed":
                plan["refused" if state == "refused" else "errors"].append({k: done[k] for k in ("card", "psd", "why")})
                continue
            files = done.pop("files")
            for path, data in files.items():
                target = join(out_dir, path)
                makedirs(dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(data)
            plan["fixed"].append({**{k: v for k, v in done.items() if k != "why"},
                                  "writes": {path: blob_sha(data) for path, data in files.items()}})


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
    one.add_argument("--whole", action="store_true", help="badges: redraw a layer whole when the splice can't")
    held = sub.add_parser("moved", help="the planned cards whose files moved since")
    held.add_argument("--plan", required=True)
    held.add_argument("--clone", required=True)
    held.add_argument("--ref", default="origin/main")
    opts = parser.parse_args(args)
    if opts.command == "moved":
        for path in moved(opts.plan, opts.clone, opts.ref):
            print(path)
        return 0
    plans = run(opts.clone, opts.kind, opts.out, opts.ref, opts.cards, opts.plan, opts.workers, opts.whole)
    print("\n".join(summary(plan) for plan in plans))
    return 1 if any(plan["errors"] for plan in plans) else 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'kinds_of',
    'moved',
    'run',
    'summary',
    'sweep_card',
]
