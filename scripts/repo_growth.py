"""
How the card repo grows: a by-hand script (no workflow, no schedule; owner 2026-11-11).

    python scripts/repo_growth.py CLONE [WEEKS]       raw bytes main's history added each week (the last WEEKS, 12), by
                                                      writer and file type
    python scripts/repo_growth.py CLONE BASE..HEAD    what a range adds (a sweep before its push): raw, and packed as git
                                                      stores it (each file a delta against its version before, window 10)

A writer is a commit's Card-sweep kind, the bot's commit kind (its subject before ':') or the author. Read-only, never
a fetch (GIT_NO_LAZY_FETCH): a blobless clone's missing sizes come from GitHub's GraphQL (gh), kept in
~/.cache/repo_growth/; the range needs its blobs (a sweep's are local).
"""
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory

BOT = "AutoCroissant"                   # git_commit.AUTHOR
SWEEP = "Card-sweep"                    # changelog.SWEEP_TRAILER
CACHE = Path.home() / ".cache" / "repo_growth"
ENV = {**environ, "GIT_NO_LAZY_FETCH": "1"}
FORMAT = f"C %H %ct %an%x00%s%x00%(trailers:key={SWEEP},valueonly,separator=%x2C)"
MB = 1e6


def git(clone: str, *args, data: str = None) -> str:
    return subprocess.run(["git", "-C", clone, *args], input=data, capture_output=True, text=True, check=True,
                          env=ENV).stdout


def kind(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    return ext if ext in ("png", "psd") else "other"


def writer(author: str, subject: str, sweep: str) -> str:
    if sweep.strip():
        return "sweep: " + sweep.strip()
    return f"bot: {subject.split(':')[0]}" if author == BOT else author


def versions(clone: str, *revs) -> list:
    """[(time, writer, path, old blob or None, new blob)]: every blob first-parent history added, once."""
    out, seen, at = [], set(), None
    for line in git(clone, "log", "--first-parent", "--reverse", "--raw", "--no-renames", "--no-abbrev",
                    "--format=" + FORMAT, *revs).splitlines():
        if line.startswith("C "):
            sha, when, rest = line[2:].split(" ", 2)
            author, subject, sweep = (rest.split("\0") + ["", ""])[:3]
            at = (int(when), writer(author, subject, sweep))
        elif line.startswith(":") and at:
            meta, path = line.split("\t", 1)
            _, mode, old, new, status = meta.split()
            if status != "D" and mode != "160000" and new not in seen:
                seen.add(new)
                out.append((*at, path, None if set(old) == {"0"} else old, new))
    return out


def sizes(clone: str, shas: list) -> dict:
    """{blob: bytes}: local ones by cat-file, the rest from the cache or GraphQL (None: unknown)."""
    found = {}
    for line in git(clone, "cat-file", "--batch-check=%(objectname) %(objectsize)", data="\n".join(shas) + "\n").splitlines():
        sha, size = line.split()[:2]
        if size.isdigit():
            found[sha] = int(size)
    missing = [sha for sha in shas if sha not in found]
    if not missing:
        return found
    url = git(clone, "remote", "get-url", "origin").strip()
    owner, name = re.search(r"github\.com[:/]([^/]+)/(.+?)(?:\.git)?$", url).groups()
    cache = CACHE / f"{owner}-{name}.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    ask = [sha for sha in missing if sha not in known]
    for n in range(0, len(ask), 100):
        part = ask[n:n + 100]
        query = f'query{{repository(owner:"{owner}",name:"{name}"){{' + "".join(
            f'o{i}:object(oid:"{sha}"){{...on Blob{{byteSize}}}}' for i, sha in enumerate(part)) + "}}"
        done = subprocess.run(["gh", "api", "graphql", "-f", "query=" + query], capture_output=True, text=True)
        if done.returncode:
            print(f"GraphQL stopped ({len(ask) - n} sizes unknown): {done.stderr.strip()[:200]}", file=sys.stderr)
            break
        got = json.loads(done.stdout)["data"]["repository"]
        known.update({sha: (got.get(f"o{i}") or {}).get("byteSize") for i, sha in enumerate(part)})
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known))
    return {**found, **{sha: known.get(sha) for sha in missing}}


def weekly(clone: str, weeks: int) -> None:
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=today.weekday(), weeks=weeks - 1)       # this week's Monday, WEEKS - 1 before
    found = [v for v in versions(clone, "origin/main") if datetime.fromtimestamp(v[0], timezone.utc).date() >= start]
    size = sizes(clone, [v[4] for v in found])
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    unknown = 0
    for when, who, path, _, new in found:
        day = datetime.fromtimestamp(when, timezone.utc).date()
        if size.get(new) is None:
            unknown += 1
            continue
        week = day - timedelta(days=day.weekday())
        for key in ((who, kind(path)), ("all", kind(path))):
            table[week][key][0] += 1
            table[week][key][1] += size[new]
    for week in sorted(table):
        rows = table[week]
        print(f"\nweek of {week}")
        for who in sorted({w for w, _ in rows}, key=lambda w: (w != "all", -sum(rows[(w, k)][1] for k in ("png", "psd", "other")))):
            cells = "".join(f"  {k} {rows[(who, k)][0]:5} {rows[(who, k)][1] / MB:9.1f} MB" for k in ("png", "psd", "other"))
            print(f"  {who[:28]:28}{cells}")
    if unknown:
        print(f"\n{unknown} versions of unknown size left out", file=sys.stderr)


def packed(clone: str, objects: list, tmp: str) -> dict:
    """{blob: bytes in a pack of objects ([(blob, path)], window 10)}."""
    git(clone, "-c", "pack.threads=4", "pack-objects", "-q", "--window=10", "--depth=50", f"{tmp}/p",
        data="".join(f"{sha} {path}\n" for sha, path in objects))
    index = next(Path(tmp).glob("p-*.idx"))
    out = {}
    for line in git(clone, "verify-pack", "-v", str(index)).splitlines():
        part = line.split()
        if len(part) >= 5 and part[1] == "blob":
            out[part[0]] = int(part[3])
    for file in Path(tmp).glob("p-*"):
        file.unlink()
    return out


def range_cost(clone: str, span: str) -> None:
    found = versions(clone, span)
    new = {v[4] for v in found}
    size = sizes(clone, sorted(new))
    if any(size.get(sha) is None for sha in new):
        sys.exit("the range's blobs are not all in the clone")
    base = {}                                   # a path's version before the range -> the path
    first = {}                                  # path -> its first new blob in the range
    for _, _, path, old, sha in found:
        first.setdefault(path, sha)
        if old and old not in new:
            base.setdefault(old, path)
    held = git(clone, "cat-file", "--batch-check=%(objectname) %(objecttype)", data="".join(f"{s}\n" for s in base))
    gone = {line.split()[0] for line in held.splitlines() if line.split()[1] == "missing"}
    if gone:                                    # a blobless clone: those files counted whole
        print(f"{len(gone)} versions before the range not in the clone: their files count whole", file=sys.stderr)
        base = {old: path for old, path in base.items() if old not in gone}
    with TemporaryDirectory() as tmp:
        alone = packed(clone, list(base.items()), tmp) if base else {}
        both = packed(clone, list(base.items()) + [(v[4], v[2]) for v in found], tmp)
    cost = {sha: both[sha] for sha in new}
    for old, path in base.items():              # the version before, made a delta of the new one: that too
        cost[first[path]] += both[old] - alone[old]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from commands.png_export import banded      # the cards venv
    table = defaultdict(lambda: [0, 0, 0, 0])
    for _, who, path, _, sha in found:
        bands = kind(path) == "png" and banded(git_bytes(clone, sha))
        for key in ((who, kind(path)), ("all", kind(path))):
            row = table[key]
            row[0], row[1], row[2], row[3] = row[0] + 1, row[1] + size[sha], row[2] + cost[sha], row[3] + bands
    print(f"{span}: versions, raw MB, packed MB (git's deltas), banded PNGs")
    for (who, k), (n, raw, packed_, bands) in sorted(table.items(), key=lambda item: (item[0][0] != "all", -item[1][2])):
        print(f"  {who[:28]:28} {k:5} {n:5} {raw / MB:9.1f} {packed_ / MB:9.1f}" + (f" {bands:5}" if k == "png" else ""))


def git_bytes(clone: str, sha: str) -> bytes:
    return subprocess.run(["git", "-C", clone, "cat-file", "blob", sha], capture_output=True, check=True, env=ENV).stdout


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if len(sys.argv) > 2 and ".." in sys.argv[2]:
        range_cost(sys.argv[1], sys.argv[2])
    else:
        weekly(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
