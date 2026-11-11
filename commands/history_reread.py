"""
Card history read again: every stored old version's PSD opened at the commit that brought its blob (range requests,
RemotePSD), parsed with today's parser, and a table of stored against re-read ability, stars, types and stats. Where
the stats differ, that version's own PNG judges them (card_checks.read_picture, check_card's rules). A table only:
old_stats.json is never written. Run by hand on the Mac, from a scratch folder (about 3 GB from raw.githubusercontent
for every version; the clone gives only its git log and the vocabulary, no PSD):

    python -m commands.history_reread --clone ~/Desktop/TTSCardMaker --old-stats <db>/old_stats.json --out <dir>
        [--cards a,b] [--workers 2]

<dir>/reads.jsonl keeps each re-read (a second run reads only what is missing or failed); <dir>/history_reread.tsv is
the table, a row per version; the counts per field, stamp and verdict are printed. Verdicts: same; unknown (both
unknown); confirm (stored unknown, the re-read knows the same); fill (stored none, re-read known); lost (re-read
unknown or empty); format (ability: spacing or apostrophes); tokens (ability: [type] tokens); differs (an unknown
stored value too). Stats that fill or differ get a picture verdict: adopt (the PNG fits the re-read, not the stored),
keep (the other way), review (both, neither, or no say).
"""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from csv import writer
from io import BytesIO
from json import dumps, loads
from os import environ, makedirs
from os.path import expanduser, isfile, join, splitext
from re import sub
from subprocess import run
from threading import local
from time import perf_counter
from typing import Optional

from global_config import CARD_REPO, ICON_MANIFEST_PATH, TYPES_DIR
from commands.card_check import vocabulary
from commands.card_db import parse_old_stats
from commands.card_edits import as_parsed
from commands.card_history import GitIndex, unknown
from commands.card_models import RAID_STARS, CardInfo, CardStats, straight, straighten_card
from commands.psd_parser import PARSER_VERSION

# ========================
# Configuration
# ========================
FIELDS = ("ability", "stars", "types", "stats")
STAT_KEYS = ("hp", "defense", "attack", "speed")
PICTURE_STATS = ("hp", "def", "atk", "spd")             # read_picture's keys, in STAT_KEYS' order
TOKEN = r"\[([a-z0-9_ ]+)(?::[a-z]+)?\]"                # a [type] token (tinted or glowing too)
EMPTY = (None, "", [])                                  # a value that says nothing
LIST_LINES = 80                                         # stats disagreements printed at most
_local = local()


# ========================
# Values and verdicts
# ========================
def stats_of(card) -> list:
    """[[hp, defense, attack, speed]] per creature (one row for a single card); -1 unknown."""
    each = getattr(card, "creatures", None) or [getattr(card, "stats", None)]
    return [[getattr(s, k, -1) if s is not None else -1 for k in STAT_KEYS] for s in each]


def value(card: CardInfo, key: str):
    """A field as compared: ability "" is None, types sorted (their order is the layers')."""
    if key == "ability":
        return getattr(card, "ability", None) or None
    if key == "types":
        return sorted(getattr(card, "types", None) or [])
    if key == "stats":
        return stats_of(card)
    return getattr(card, key, None)


def _spaced(text: str) -> str:
    """text with straight apostrophes and single spaces."""
    return " ".join(straight(text or "").split())


def _known(card: CardInfo, key: str) -> bool:
    """Whether a reading knew the field: not unknown (card_history), no stars None."""
    return key not in unknown(card) and not (key == "stars" and getattr(card, "stars", None) is None)


def verdict(stored: CardInfo, reread: CardInfo, key: str) -> str:
    """same / unknown / confirm / fill / lost / format / tokens / differs (the module's docstring)."""
    a, b = value(stored, key), value(reread, key)
    if key == "stats":
        if a == b:
            return "same"
        pairs = [(x, y) for ra, rb in zip(a, b) for x, y in zip(ra, rb) if x != y] if len(a) == len(b) else [(0, 1)]
        return "fill" if all(x < 0 <= y for x, y in pairs) else "lost" if all(y < 0 <= x for x, y in pairs) else "differs"
    old, new = _known(stored, key), _known(reread, key)
    if a == b:
        return "same" if old else "confirm" if new else "unknown"
    if not new or (a not in EMPTY and b in EMPTY):
        return "lost"
    if not old and a in EMPTY:
        return "fill"
    if key == "ability" and a and b:
        if _spaced(a) == _spaced(b):
            return "format"
        if any(_spaced(sub(TOKEN, to, a)) == _spaced(sub(TOKEN, to, b)) for to in (r"\1", "")):
            return "tokens"                             # a word became its [type] token, or a token was added
    return "differs"


def picture(found: Optional[dict], stored: CardInfo, reread: CardInfo) -> str:
    """adopt / keep / review: the PNG's bars (bars()) on the stats the two readings differ in."""
    from commands.card_checks import _pairs
    reads = (found or {}).get("reads")
    if not reads or max(getattr(stored, "stars", 0) or 0, getattr(reread, "stars", 0) or 0) >= RAID_STARS:
        return "review"                                 # no bars read; a raid boss's run to 40
    if found.get("style") == "custom" and any(row["best"] is None for rows in reads.values() for row in rows):
        return "review"                                 # mismatches(): custom bars speak only when all read
    a, b = stats_of(stored), stats_of(reread)
    fits_old = fits_new = True
    spoke = False
    for n, stat in enumerate(PICTURE_STATS):
        old, new = [row[n] for row in a], [row[n] for row in b]
        if old == new:
            continue
        rows = reads.get(stat) or []
        pairs_old, pairs_new = _pairs(rows, old), _pairs(rows, new)
        if not pairs_old or not pairs_new or any(row["could"] is None for row, _ in pairs_old + pairs_new):
            return "review"
        spoke = True
        fits_old &= all(v in row["could"] for row, v in pairs_old)
        fits_new &= all(v in row["could"] for row, v in pairs_new)
    if not spoke or fits_old == fits_new:
        return "review"
    return "adopt" if fits_new else "keep"


# ========================
# Reading
# ========================
def tree_vocabulary(clone: str, head: str):
    """card_check's Vocabulary of the clone's tree at head (Types/ names and the icon manifest, which must be there)."""
    env = {**environ, "GIT_NO_LAZY_FETCH": "1"}
    paths = run(["git", "-C", clone, "ls-tree", "-r", "--name-only", head, "--", TYPES_DIR], capture_output=True,
                text=True, env=env, check=True).stdout.splitlines()
    manifest = run(["git", "-C", clone, "cat-file", "blob", f"{head}:{ICON_MANIFEST_PATH}"], capture_output=True,
                   env=env, check=True).stdout
    return vocabulary(paths, manifest)


def png_path(index: GitIndex, card: CardInfo) -> str:
    """The version's PNG path as the index spells it (a variant's: NameN.png)."""
    stem = splitext(card.path)[0]
    path = f"{stem}{card.variant}.png" if (getattr(card, "variant", None) or 1) > 1 else f"{stem}.png"
    if path in index.pngs:
        return path
    return next((p for p in index.pngs if p.lower() == path.lower()), path)


def bars(psd, image, creatures, variant: Optional[int]) -> dict:
    """check_card's reading of a picture: {"style", "reads"}; reads {} when its bars have no say (none, counted rows)."""
    from commands.card_checks import bar_style, read_picture
    from commands.psd_edit import counted_row
    from commands.psd_parser import variant_view
    from commands.psd_regions import stat_rows
    style = bar_style(psd, creatures)[0]
    rows = [group for found in stat_rows(psd).values() for group, _, _ in found]
    if not style or (rows and all(counted_row(group) for group in rows)):
        return {"style": style, "reads": {}}
    image.load()
    with variant_view(psd, variant or 1):
        return {"style": style, "reads": read_picture(psd, image, cells=style in ("new", "few"))}


def read_one(job: dict, repository: str, vocab) -> dict:
    """One PSD at its commit: today's parse; the PNG's bars only when a stored version's stats differ from it."""
    from PIL import Image
    from commands.github_api import RemotePSD, get_with_retry, raw_url
    from commands.psd_parser import variants_of
    if not hasattr(_local, "parser"):
        _local.parser = vocab.parser()
    out, remote, start = {"key": job["key"]}, None, perf_counter()
    try:
        remote = RemotePSD(repository, job["commit"], job["at"])
        variant = job["variant"] or (1 if len(variants_of(remote.psd)) > 1 else None)
        card = _local.parser.parse(remote.psd, job["path"], None, variant=variant)
        straighten_card(card)
        rows = stats_of(card)
        out["card"] = {"ability": card.ability, "stars": card.stars, "types": list(card.types or []),
                       "stats": [getattr(card.stats, k, -1) for k in STAT_KEYS],
                       "creatures": rows if card.creatures else None, "problems": list(card.problems or [])}
        out["pictures"] = {}
        for png, at in job["pngs"] if any(s != rows for s in job["stored"]) else ():
            try:
                reply = get_with_retry(raw_url(repository, at, png), timeout=60)
                reply.raise_for_status()
                out["png_bytes"] = out.get("png_bytes", 0) + len(reply.content)
                out["pictures"][f"{at}:{png}"] = bars(remote.psd, Image.open(BytesIO(reply.content)), card.creatures,
                                                      variant)
            except Exception as e:                  # no say
                out["pictures"][f"{at}:{png}"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        if remote.failed:
            out["error"] = "a range read failed"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    finally:
        if remote is not None:
            out["bytes"], out["requests"] = remote.fetched, remote.reader.requests
            remote.close()
    out["seconds"] = round(perf_counter() - start, 2)
    return out


def reread_card(stored: CardInfo, found: dict) -> CardInfo:
    """The re-read as a CardInfo beside the stored version (its name, type and path)."""
    card = found["card"]
    out = CardInfo(name=stored.name, card_type=stored.card_type, path=stored.path, ability=card["ability"],
                   stars=card["stars"], types=card["types"], stats=CardStats(*card["stats"]),
                   problems=card["problems"], parser_version=PARSER_VERSION)
    if card["creatures"]:
        out.creatures = [CardStats(*row) for row in card["creatures"]]
    return out


# ========================
# The table
# ========================
def _show(key: str, v) -> str:
    if key == "stats":
        return "; ".join("/".join("?" if x < 0 else str(x) for x in row) for row in v)
    if key == "types":
        return ",".join(v)
    return "" if v is None else str(v).replace("\t", " ").replace("\n", "\\n")


def jobs_of(old_stats: dict, index: GitIndex, names=()) -> tuple:
    """(rows, jobs): a row per version {name, n, card, key, commit, png}; a job per (blob, path, variant): where its
    blob came in, the stored versions' stats and their PNGs (path, commit) to judge by."""
    rows, jobs = [], {}
    for name in sorted(old_stats):
        if names and name not in names:
            continue
        for n, card in enumerate(old_stats[name]):
            blob = getattr(card, "blob_sha", None)
            intro = index.intro.get(blob) if blob else None
            row = {"name": name, "n": n, "card": card, "key": None, "commit": "", "png": ""}
            rows.append(row)
            if intro is None:
                continue
            key = f"{blob}:{card.path}:{getattr(card, 'variant', None) or ''}"
            row.update(key=key, commit=intro[0].sha)
            job = jobs.setdefault(key, {"key": key, "commit": intro[0].sha, "at": intro[1], "path": card.path,
                                        "variant": getattr(card, "variant", None), "pngs": [], "stored": []})
            job["stored"].append(stats_of(as_parsed(card)))
            at = (getattr(card, "version", None) or {}).get("png_commit")
            if at:
                png = png_path(index, card)
                row["png"] = f"{at}:{png}"
                if [png, at] not in job["pngs"]:
                    job["pngs"].append([png, at])
    return rows, jobs


def table(rows: list, found: dict) -> tuple:
    """(TSV lines, counts {(field, stamp, verdict): n}, picture counts, stats disagreement lines)."""
    head = ["name", "version", "stamp", "cause", "path", "blob", "read_at", "read"]
    for key in FIELDS:
        head += [f"{key}_stored", f"{key}_reread", f"{key}_verdict"]
    lines, counts, judged, listed = [head + ["picture"]], Counter(), Counter(), []
    for row in rows:
        card = row["card"]
        stamp = str(getattr(card, "parser_version", None) or "none")
        result = found.get(row["key"]) if row["key"] else None
        state = "no blob" if not row["key"] else "missing" if result is None else "error" if "error" in result else "ok"
        line = [row["name"], row["n"], stamp, (getattr(card, "version", None) or {}).get("cause", ""), card.path,
                (card.blob_sha or "")[:10], row["commit"][:10], state if state != "error" else result["error"]]
        counts[("read", stamp, state)] += 1
        mark = ""
        if state == "ok":
            stored, again = as_parsed(card), reread_card(card, result)
            for key in FIELDS:
                v = verdict(stored, again, key)
                counts[(key, stamp, v)] += 1
                line += [_show(key, value(stored, key)), _show(key, value(again, key)), v]
                if key == "stats" and v in ("fill", "differs"):
                    mark = picture((result.get("pictures") or {}).get(row["png"]), stored, again)
                    judged[(v, mark)] += 1
                    listed.append(f"{row['name']} v{row['n']} ({stamp}): {_show(key, value(stored, key))} -> "
                                  f"{_show(key, value(again, key))} [{v}, picture {mark}]")
        else:
            line += [""] * (3 * len(FIELDS))
        lines.append(line + [mark])
    return lines, counts, judged, listed


def summary(counts: Counter, judged: Counter, listed: list) -> str:
    """Markdown: a table per field (stamps down, verdicts across), the picture verdicts, the stats disagreements."""
    stamps = sorted({s for _, s, _ in counts}, key=lambda s: (s != "none", int(s) if s.isdigit() else 0))
    out = []
    for key in ("read", *FIELDS):
        verdicts = sorted({v for k, _, v in counts if k == key})
        out += [f"\n{key}", "| stamp | " + " | ".join(verdicts) + " |", "|---" * (len(verdicts) + 1) + "|"]
        out += [f"| {s} | " + " | ".join(str(counts[(key, s, v)]) for v in verdicts) + " |" for s in stamps
                if any(counts[(key, s, v)] for v in verdicts)]
    if judged:
        out += ["\nstats pictures: " + ", ".join(f"{v} {m}: {n}" for (v, m), n in sorted(judged.items()))]
        out += listed[:LIST_LINES] + ([f"... {len(listed) - LIST_LINES} more"] if len(listed) > LIST_LINES else [])
    return "\n".join(out)


# ========================
# Command line
# ========================
def main(args: Optional[list] = None) -> int:
    cli = ArgumentParser(prog="python -m commands.history_reread", description="Card history read again with today's "
                         "parser: a table of stored against re-read values (nothing written to the card data).")
    cli.add_argument("--clone", required=True, help="a full-history clone of the card repo (blobless is enough)")
    cli.add_argument("--old-stats", required=True, help="the card data's db/old_stats.json")
    cli.add_argument("--out", required=True, help="the folder for reads.jsonl and history_reread.tsv")
    cli.add_argument("--head", default="origin/main", help="the clone's ref to index and take the vocabulary from")
    cli.add_argument("--cards", default="", help="only these names (comma-separated)")
    cli.add_argument("--workers", type=int, default=2, help="threads (default %(default)s)")
    options = cli.parse_args(args)
    clone, out = expanduser(options.clone), expanduser(options.out)
    makedirs(out, exist_ok=True)
    with open(expanduser(options.old_stats), encoding="utf-8") as fh:
        old_stats = parse_old_stats(fh.read())
    index = GitIndex.from_git(clone, options.head)
    rows, jobs = jobs_of(old_stats, index, {n.strip() for n in options.cards.split(",") if n.strip()})
    cache, found = join(out, "reads.jsonl"), {}
    if isfile(cache):
        with open(cache, encoding="utf-8") as fh:
            for line in fh:
                one = loads(line)
                if "error" not in one:
                    found[one["key"]] = one
    todo = [job for key, job in jobs.items() if key not in found]
    print(f"{len(rows)} versions, {len(jobs)} PSD readings, {len(todo)} to read", flush=True)
    vocab, start, done = tree_vocabulary(clone, options.head), perf_counter(), 0
    with ThreadPoolExecutor(max(1, options.workers)) as pool, open(cache, "a", encoding="utf-8") as fh:
        for future in as_completed([pool.submit(read_one, job, CARD_REPO, vocab) for job in todo]):
            one = future.result()
            found[one["key"]] = one
            fh.write(dumps(one) + "\n")
            fh.flush()
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(todo)} read, {perf_counter() - start:.0f} s", flush=True)
    lines, counts, judged, listed = table(rows, found)
    with open(join(out, "history_reread.tsv"), "w", encoding="utf-8", newline="") as fh:
        writer(fh, delimiter="\t", lineterminator="\n").writerows(lines)
    reads = [found[k] for k in jobs if k in found]
    print(f"{sum(1 for r in reads if 'error' not in r)} read, {sum(1 for r in reads if 'error' in r)} failed; "
          f"{sum(r.get('bytes', 0) for r in reads) / 1e6:.1f} MB PSD in {sum(r.get('requests', 0) for r in reads)} "
          f"requests, {sum(r.get('png_bytes', 0) for r in reads) / 1e6:.1f} MB PNG; {perf_counter() - start:.0f} s")
    print(summary(counts, judged, listed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'main',
    'picture',
    'verdict',
]
