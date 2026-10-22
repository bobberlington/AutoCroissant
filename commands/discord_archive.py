"""
Discord archive miner: a guild's messages and image attachments, read through Discord's REST API v10 with GETs only
(never the gateway, never a write), so it can run beside the production bot on the same token.

    python -m commands.discord_archive --guild <id> [--channels <id>...] [--all] [--out <dir>]
                                       [--order oldest|newest] [--limit <n>] [--no-threads] [--no-download]
                                       [--list] [--retry-downloads] [--config <config.py>]

--channels are mined first (a category: its children), then --all adds every other channel; each channel's threads
(active, archived public, archived private: all with Manage Threads, else those the bot joined) follow it. A channel
the bot can't read is skipped with the reason (worked out from roles and overwrites before any request, else the HTTP
error). The out dir is ~/Documents/Tools/discord-archive or a folder in it, never inside a git repo (the messages
are other people's words); one run at a time holds it (.lock), and it writes through no symlink:
    channels.json                every channel and thread found: type, parent, tags, mined or why not
    messages/<id>.jsonl          one message a line, per channel or thread
    cursors/<id>.json            newest and oldest ids stored, whether the first message was reached
    attachments/<sha256>.<ext>   images: attachments (PSDs too) and Discord-hosted embed images; 25 MB cap, synced
    attachments.jsonl            every download try (the last line for an id wins; embeds: <message>-e<n>-<kind>)
    summary.json, runs.jsonl     this run (requests, rate limits, per channel counts and dates); one line per run
Resumable: a rerun fetches only messages after the newest stored, and before the oldest until the first is reached; a
torn last line is cut off. Attachment URLs are signed and expire, so images are fetched with their page;
--retry-downloads re-reads the messages of failed ones (found in the index and the lines) and patches their lines.
Rate limits are kept per Discord bucket; 1000 rejected requests (401/403/429) in 10 minutes stop the run. The token is
read from config.py with ast (never imported, printed or written) and redacted from all error text; the CDN never
gets it.
"""
from argparse import ArgumentParser
from ast import AnnAssign, Assign, Name, literal_eval, parse
from collections import deque
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from json import dump, dumps, load, loads
from fcntl import LOCK_EX, LOCK_NB, flock
from glob import glob
from math import isfinite
from os import (O_APPEND, O_CREAT, O_NOFOLLOW, O_RDWR, O_TRUNC, O_WRONLY, fdopen, fsync, ftruncate, makedirs,
                open as os_open, remove, replace)
from os.path import abspath, basename, dirname, exists, expanduser, getsize, islink, join, realpath, sep, splitext
from re import fullmatch
from signal import SIGTERM, signal
from sys import stderr
from time import monotonic, sleep
from traceback import format_exc
from typing import Callable, Optional
from urllib.parse import urljoin, urlsplit

from requests import Session
from requests.exceptions import RequestException

# ========================
# Configuration
# ========================
API = "https://discord.com/api/v10"
USER_AGENT = "DiscordBot (AutoCroissant, 1.0)"
CONFIG_PATH = join(dirname(dirname(abspath(__file__))), "config.py")
DEFAULT_OUT = "~/Documents/Tools/discord-archive"
PAGE = 100                                      # messages or threads a request: Discord's max
TIMEOUT = 30                                    # seconds a request
MAX_TRIES = 8                                   # a request: 429s, 5xx and network errors
BACKOFF = 1.0                                   # seconds after a 5xx or network error, doubled each try
RATE_MARGIN = 0.05                              # seconds added to Discord's waits
MIN_LIMIT_WAIT = 0.5                            # seconds at least after a 429
INVALID_CAP, INVALID_WINDOW = 1000, 600         # 401/403/429s that stop the run (Discord bans at 10,000 / 10 min)
MAJOR = {"channels", "guilds", "webhooks"}      # Discord's major route params: their ids pick the bucket
MIN_GAP = 1 / 45                                # seconds between API requests: under the 50/s global limit
MAX_DOWNLOAD = 25 * 1024 * 1024
CHUNK = 1 << 16
CDN_HOSTS = ("cdn.discordapp.com", "media.discordapp.net")   # attachment hosts; no token sent there
PROXY_HOST = r"images-ext-[0-9]{1,2}\.discordapp\.net"    # Discord's proxy for linked images
EMBED_MEDIA = ("image", "thumbnail")
IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "psd"}
EXT_ALIASES = {"jpeg": "jpg", "jpe": "jpg", "tif": "tiff"}
MAX_REDIRECTS = 3
EPOCH_MS = 1420070400000                        # Discord's snowflake epoch
REDACTED = "<redacted>"

CATEGORY, PRIVATE_THREAD = 4, 12
THREAD_TYPES = {10, 11, 12}
MESSAGE_TYPES = {0, 2, 5, 10, 11, 12, 13}       # text, voice chat, announcement, threads, stage chat
THREAD_PARENTS = {0, 5, 15, 16}                 # forum and media channels hold only threads
TYPE_NAMES = {0: "text", 2: "voice", 4: "category", 5: "announcement", 10: "announcement thread",
              11: "public thread", 12: "private thread", 13: "stage", 14: "directory", 15: "forum", 16: "media"}

ADMINISTRATOR, VIEW_CHANNEL, READ_HISTORY, MANAGE_THREADS = 1 << 3, 1 << 10, 1 << 16, 1 << 34
ALL_PERMISSIONS = (1 << 64) - 1


# ========================
# Helpers
# ========================
def is_snowflake(value) -> bool:
    return isinstance(value, str) and fullmatch(r"[0-9]{1,20}", value) is not None


def is_media_key(value) -> bool:
    """An attachment id, or an embed image's key."""
    return isinstance(value, str) and fullmatch(r"[0-9]{1,20}(-f[0-9]{1,3})?-e[0-9]{1,3}-(image|thumbnail)|[0-9]{1,20}",
                                                value) is not None


def is_cdn(host: str) -> bool:
    return host in CDN_HOSTS or fullmatch(PROXY_HOST, host) is not None


def host_of(url) -> str:
    """A URL's host, lowercase ('' if it won't parse: embed URLs are anyone's)."""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def seconds_in(value, default: float) -> float:
    """A finite float, else default."""
    try:
        found = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return found if isfinite(found) else default


def open_new(path: str, mode: str = "w"):
    """open() that won't follow a symlink: w truncates, a appends, r+ edits. A lone surrogate (valid JSON, not UTF-8)
    is written as its \\u escape, which loads() reads back."""
    flags = {"w": O_WRONLY | O_CREAT | O_TRUNC, "a": O_WRONLY | O_CREAT | O_APPEND, "r+": O_RDWR}[mode.rstrip("b")]
    return fdopen(os_open(path, flags | O_NOFOLLOW, 0o600), mode,
                  **({} if "b" in mode else {"encoding": "utf-8", "errors": "backslashreplace"}))


def snowflake_time(snowflake) -> Optional[str]:
    """The UTC time a Discord id was made (ISO)."""
    if snowflake is None:
        return None
    seconds = ((int(snowflake) >> 22) + EPOCH_MS) / 1000
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="seconds")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def header(headers, name: str) -> Optional[str]:
    """A response header, any case."""
    name = name.lower()
    return next((value for key, value in (headers or {}).items() if key.lower() == name), None)


def route_key(path: str) -> str:
    """A path's rate-limit bucket: ids kept only after channels, guilds, webhooks."""
    parts = path.split("/")
    return "/".join(":id" if is_snowflake(part) and i and parts[i - 1] not in MAJOR else part
                    for i, part in enumerate(parts))


def just_after(stamp: str) -> Optional[str]:
    """An ISO time 1 ms later (None if it won't parse)."""
    try:
        return (datetime.fromisoformat(stamp) + timedelta(milliseconds=1)).isoformat()
    except (TypeError, ValueError):
        return None


def is_image(attachment: dict) -> bool:
    kind = (attachment.get("content_type") or "").lower()
    return kind.startswith("image/") or splitext(attachment.get("filename") or "")[1][1:].lower() in IMAGE_EXTS


def extension(attachment: dict) -> str:
    """The cache file's extension: the filename's, else the content type's, else bin."""
    ext = splitext(attachment.get("filename") or "")[1][1:].lower()
    if not fullmatch(r"[a-z0-9]{1,8}", ext):
        ext = (attachment.get("content_type") or "").split(";")[0].split("/")[-1].lower()
    return EXT_ALIASES.get(ext, ext) if fullmatch(r"[a-z0-9]{1,8}", ext) else "bin"


def all_attachments(message: dict) -> list:
    """A message's attachments and those of the messages it forwards."""
    found = list(message.get("attachments") or [])
    for snapshot in message.get("message_snapshots") or []:
        found += (snapshot.get("message") or {}).get("attachments") or []
    return found


def embed_summary(embed: dict) -> dict:
    return {"type": embed.get("type"), "url": embed.get("url"), "title": embed.get("title"),
            "description": embed.get("description"), "image": (embed.get("image") or {}).get("url"),
            "thumbnail": (embed.get("thumbnail") or {}).get("url")}


def embed_key(message_id, i: int, forward: Optional[int] = None) -> str:
    """An embed's media key, less its kind: <message>[-f<forward>]-e<embed> (see is_media_key)."""
    return f"{message_id}{'' if forward is None else f'-f{forward}'}-e{i}"


def embed_media(embed: dict, key: str) -> list:
    """An embed's images as attachment-like dicts (Discord-hosted URL, the original first)."""
    found = []
    for kind in EMBED_MEDIA:
        media = embed.get(kind) or {}
        url = next((u for u in (media.get("url"), media.get("proxy_url"))
                    if isinstance(u, str) and is_cdn(host_of(u))), None)
        if url:
            found.append({"id": f"{key}-{kind}", "url": url, "filename": basename(urlsplit(url).path),
                          "width": media.get("width"), "height": media.get("height"), "kind": kind})
    return found


def stored_media(record: dict) -> list:
    """A stored line's attachment and embed image entries."""
    entries = list(record.get("attachments") or [])
    for embed in record.get("embeds") or []:
        entries += embed.get("media") or []
    for forwarded in record.get("forwarded") or []:
        entries += forwarded.get("attachments") or []
        for embed in forwarded.get("embeds") or []:
            entries += embed.get("media") or []
    return [entry for entry in entries if isinstance(entry, dict)]


def all_media(message: dict) -> list:
    """all_attachments plus the embeds' images, keyed as record() keys them."""
    found = all_attachments(message)
    for i, embed in enumerate(message.get("embeds") or []):
        found += embed_media(embed, embed_key(message.get("id"), i))
    for j, snapshot in enumerate(message.get("message_snapshots") or []):
        for i, embed in enumerate((snapshot.get("message") or {}).get("embeds") or []):
            found += embed_media(embed, embed_key(message.get("id"), i, j))
    return found


def reaction_summary(message: dict) -> list:
    return [{"emoji": (r.get("emoji") or {}).get("name"), "emoji_id": (r.get("emoji") or {}).get("id"),
             "count": r.get("count")} for r in message.get("reactions") or []]


def base_permissions(guild: dict, member: dict) -> int:
    """The member's guild-wide permissions (Discord's algorithm)."""
    if guild.get("owner_id") == (member.get("user") or {}).get("id"):
        return ALL_PERMISSIONS
    roles = {role["id"]: int(role.get("permissions") or 0) for role in guild.get("roles") or []}
    permissions = roles.get(guild["id"], 0)
    for role in member.get("roles") or []:
        permissions |= roles.get(role, 0)
    return ALL_PERMISSIONS if permissions & ADMINISTRATOR else permissions


def channel_permissions(base: int, channel: dict, guild_id: str, member: dict) -> int:
    """The member's permissions in a channel: @everyone, role, then member overwrites."""
    if base & ADMINISTRATOR:
        return ALL_PERMISSIONS
    overwrites = {o["id"]: (int(o.get("allow") or 0), int(o.get("deny") or 0))
                  for o in channel.get("permission_overwrites") or []}
    allow, deny = overwrites.get(guild_id, (0, 0))
    permissions = (base & ~deny) | allow
    allow = deny = 0
    for role in member.get("roles") or []:
        role_allow, role_deny = overwrites.get(role, (0, 0))
        allow, deny = allow | role_allow, deny | role_deny
    permissions = (permissions & ~deny) | allow
    allow, deny = overwrites.get((member.get("user") or {}).get("id"), (0, 0))
    return (permissions & ~deny) | allow


def load_token(path: str = CONFIG_PATH) -> str:
    """config.py's TOKEN, read with ast: config.py is never run, and errors never quote it."""
    try:
        with open(path, encoding="utf-8") as f:
            tree = parse(f.read(), path)
    except (OSError, SyntaxError, ValueError) as e:
        raise SystemExit(f"can't read the token from {path}: {type(e).__name__}") from None
    value = None
    for node in tree.body:                      # the last top-level binding: the one main.py imports
        targets = node.targets if isinstance(node, Assign) else [node.target] if isinstance(node, AnnAssign) else []
        if any(isinstance(t, Name) and t.id == "TOKEN" for t in targets) and node.value is not None:
            value = node.value
    if value is None:
        raise SystemExit(f"no TOKEN in {path}")
    try:
        token = literal_eval(value)
    except ValueError:
        raise SystemExit(f"TOKEN in {path} isn't a plain string") from None
    if not (isinstance(token, str) and token.strip()):
        raise SystemExit(f"no TOKEN in {path}")
    return token.strip()


def inside_git(path: str) -> Optional[str]:
    """The git work tree holding path, if any."""
    current = realpath(expanduser(path))
    while True:
        if exists(join(current, ".git")):
            return current
        parent = dirname(current)
        if parent == current:
            return None
        current = parent


# ========================
# HTTP (GETs only)
# ========================
class ApiError(Exception):
    """A failed GET (status 0: no answer); fatal stops the run."""
    def __init__(self, status: int, text: str = "", code=None, halt: bool = False):
        self.status, self.text, self.code, self.halt = status, text, code, halt
        super().__init__(self.reason())

    @property
    def fatal(self) -> bool:
        return self.status == 401 or self.halt

    def reason(self) -> str:
        return (f"HTTP {self.status}" if self.status else "no answer") + (f" {self.text}" if self.text else "") \
            + (f" ({self.code})" if self.code else "")


class TooBig(Exception):
    pass


class Rest:
    """GETs to the API (with the token) and CDN (without), rate limited."""
    def __init__(self, token: str, session=None, sleeper: Optional[Callable] = None, clock: Optional[Callable] = None):
        self._token = token
        self._session = session if session is not None else Session()
        self._sleeper, self._clocker = sleeper, clock
        self._ready: dict = {}                  # route key -> clock time it may be called again
        self._global_ready = 0.0
        self._invalid: deque = deque()          # clock times of recent 401/403/429s
        self.requests = self.limited = 0
        self.waited = 0.0

    def _sleep(self, seconds: float):
        (self._sleeper or sleep)(seconds)

    def _clock(self) -> float:
        return (self._clocker or monotonic)()

    def __repr__(self):
        return "Rest(token=<redacted>)"

    def redact(self, text) -> str:
        text = str(text)
        return text.replace(self._token, REDACTED) if self._token else text

    def _hold(self, route: str, seconds: float, everywhere: bool = False):
        until = self._clock() + max(0.0, seconds)
        if everywhere:
            self._global_ready = max(self._global_ready, until)
        else:
            self._ready[route] = max(self._ready.get(route, 0.0), until)

    def _wait(self, route: str):
        delay = max(self._global_ready, self._ready.get(route, 0.0)) - self._clock()
        if delay > 0:
            self.waited += delay
            self._sleep(delay)

    def _retry_after(self, response) -> tuple:
        """(seconds, global) from a 429: the body's retry_after, else Retry-After, else Reset-After."""
        try:
            body = response.json()
        except ValueError:
            body = None
        body = body if isinstance(body, dict) else {}
        seconds = body.get("retry_after") or header(response.headers, "Retry-After") \
            or header(response.headers, "X-RateLimit-Reset-After") or 1
        everywhere = bool(body.get("global")) or header(response.headers, "X-RateLimit-Global") == "true" \
            or header(response.headers, "X-RateLimit-Scope") == "global"
        return max(seconds_in(seconds, 1.0), MIN_LIMIT_WAIT) + RATE_MARGIN, everywhere

    def _count_invalid(self):
        """Stops the run well before Discord's invalid-request ban."""
        now = self._clock()
        self._invalid.append(now)
        while self._invalid and self._invalid[0] < now - INVALID_WINDOW:
            self._invalid.popleft()
        if len(self._invalid) >= INVALID_CAP:
            raise ApiError(429, f"{len(self._invalid)} rejected requests in {INVALID_WINDOW // 60} min: stopped",
                           halt=True)

    def _send(self, url: str, route: str, params, auth: bool, stream: bool):
        headers = {"User-Agent": USER_AGENT}
        if auth:
            headers["Authorization"] = f"Bot {self._token}"
        failure = ApiError(0, "no tries")
        for attempt in range(MAX_TRIES):
            self._wait(route)
            self.requests += 1
            try:
                response = self._session.get(url, params=params, headers=headers, timeout=TIMEOUT, stream=stream,
                                             allow_redirects=False)
            except RequestException as e:
                failure = ApiError(0, self.redact(f"{type(e).__name__}: {e}"))
                self._hold(route, BACKOFF * 2 ** attempt)
                continue
            if auth:
                self._hold("", MIN_GAP, everywhere=True)
                if response.status_code in (401, 403, 429):
                    try:
                        self._count_invalid()
                    except ApiError:
                        response.close()
                        raise
            if header(response.headers, "X-RateLimit-Remaining") == "0":
                self._hold(route, seconds_in(header(response.headers, "X-RateLimit-Reset-After"), 1.0) + RATE_MARGIN)
            if response.status_code == 429:
                self.limited += 1
                seconds, everywhere = self._retry_after(response)
                self._hold(route, seconds, everywhere)
                failure = ApiError(429, "rate limited")
                response.close()
                continue
            if response.status_code >= 500:
                failure = ApiError(response.status_code, "server error")
                response.close()
                self._hold(route, BACKOFF * 2 ** attempt)
                continue
            return response
        raise failure

    def _error(self, response) -> ApiError:
        try:
            body = response.json()
        except ValueError:
            body = None
        body = body if isinstance(body, dict) else {}
        text = self.redact(body.get("message") or response.text or "")[:200]
        return ApiError(response.status_code, text, body.get("code"))

    def get(self, path: str, params: Optional[dict] = None):
        """The JSON of an API GET; ApiError for anything but 2xx."""
        response = self._send(API + path, route_key(path), params, True, False)
        try:
            if response.status_code >= 300:
                raise self._error(response)
            try:
                return response.json()
            except ValueError:
                raise ApiError(response.status_code, "not JSON") from None
        finally:
            response.close()

    def fetch_file(self, url: str, dest: str, limit: int = MAX_DOWNLOAD) -> tuple:
        """Streams a CDN file to dest: (sha256, bytes); redirects only to the CDN."""
        for _ in range(MAX_REDIRECTS + 1):
            host = host_of(url)
            if not is_cdn(host):
                raise ApiError(0, f"not a Discord CDN host: {host or url[:40]}")
            response = self._send(url, host, None, False, True)
            location = header(response.headers, "Location")
            if response.status_code not in (301, 302, 303, 307, 308) or not location:
                break
            response.close()
            url = urljoin(url, location)
        else:
            raise ApiError(0, "too many redirects")
        try:
            if response.status_code >= 300:
                raise ApiError(response.status_code, "CDN")
            digest, size = sha256(), 0
            with open_new(dest, "wb") as out:
                for chunk in response.iter_content(CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise TooBig(size)
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                fsync(out.fileno())
            return digest.hexdigest(), size
        except RequestException as e:
            raise ApiError(0, self.redact(f"{type(e).__name__}: {e}")) from None
        finally:
            response.close()


# ========================
# Out dir
# ========================
class Store:
    """The out dir: messages/, cursors/, attachments/ and its index, channels.json, summary.json, runs.jsonl."""
    def __init__(self, root: str):
        self.root = root
        makedirs(root, exist_ok=True)
        self._lock = open_new(join(root, ".lock"), "a")
        try:
            flock(self._lock.fileno(), LOCK_EX | LOCK_NB)
        except OSError:
            self._lock.close()
            raise SystemExit(f"another run is using {root}") from None
        for sub in ("messages", "cursors", "attachments"):
            if islink(join(root, sub)):
                self.close()
                raise SystemExit(f"{join(root, sub)} is a symlink: refused")
            makedirs(join(root, sub), exist_ok=True)
        for part in glob(join(root, "attachments", ".part-*")):   # left by an interrupted download
            remove(part)
        self.attachments = self._load_index()   # media key -> its last try

    def close(self):
        """Frees the out dir for another run."""
        if not self._lock.closed:
            self._lock.close()

    def _path(self, kind: str, channel_id: str, suffix: str) -> str:
        if not is_snowflake(channel_id):
            raise ValueError(f"not a Discord id: {channel_id!r}")
        return join(self.root, kind, channel_id + suffix)

    def scan(self, channel_id: str) -> tuple:
        """(count, oldest id, newest id) stored for a channel; a torn last line is cut off."""
        path = self._path("messages", channel_id, ".jsonl")
        count, low, high, good = 0, None, None, 0
        if not exists(path):
            return 0, None, None
        with open(path, "rb") as f:
            for line in f:
                if not line.endswith(b"\n"):
                    break
                good += len(line)
                try:
                    message_id = int(loads(line)["id"])
                except (ValueError, KeyError, TypeError):
                    continue
                count += 1
                low = message_id if low is None else min(low, message_id)
                high = message_id if high is None else max(high, message_id)
        if good < getsize(path):
            with open_new(path, "r+b") as f:
                ftruncate(f.fileno(), good)
        return count, low, high

    def append(self, channel_id: str, records: list):
        with open_new(self._path("messages", channel_id, ".jsonl"), "a") as f:
            f.write("".join(dumps(record, ensure_ascii=False) + "\n" for record in records))
            f.flush()
            fsync(f.fileno())

    def cursor(self, channel_id: str) -> dict:
        try:
            with open(self._path("cursors", channel_id, ".json"), encoding="utf-8") as f:
                found = load(f)
        except (OSError, ValueError):
            return {}
        return found if isinstance(found, dict) else {}

    def save_cursor(self, channel_id: str, cursor: dict):
        self.write_json(self._path("cursors", channel_id, ".json"), cursor)

    def write_json(self, name: str, data):
        """Atomic: a temp file renamed over name."""
        path = join(self.root, name)
        with open_new(path + ".tmp") as f:
            dump(data, f, ensure_ascii=False, indent=1)
            f.flush()
            fsync(f.fileno())
        replace(path + ".tmp", path)

    def add_line(self, name: str, record: dict):
        with open_new(join(self.root, name), "a") as f:
            f.write(dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            fsync(f.fileno())

    def _load_index(self) -> dict:
        found, path = {}, join(self.root, "attachments.jsonl")
        try:
            with open(path, "rb") as f:
                data = f.read()
            if data and not data.endswith(b"\n"):
                with open_new(path, "r+b") as f:                            # a torn last line
                    ftruncate(f.fileno(), data.rfind(b"\n") + 1)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = loads(line)
                        found[entry["id"]] = entry
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            pass
        return found

    def has_file(self, name: str, size: Optional[int] = None) -> bool:
        """A stored attachment file, whole (its size known)."""
        path = join(self.root, name)
        return exists(path) and (size is None or getsize(path) == size)

    def file_for(self, digest: str, size: int) -> Optional[str]:
        """The stored file with these bytes under any extension (.jpg, .jpeg...)."""
        for path in sorted(glob(join(self.root, "attachments", digest + ".*"))):
            if getsize(path) == size:
                return "attachments/" + basename(path)
        return None

    def patch_attachments(self, channel_id: str, results: dict) -> int:
        """Puts retried downloads (media key -> result) into a channel's lines; the count."""
        path, patched = self._path("messages", channel_id, ".jsonl"), 0
        if not results or not exists(path):
            return 0
        self.scan(channel_id)                                                   # a torn last line goes first
        lines = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = loads(line)
                    entries = stored_media(record)
                except (ValueError, AttributeError, TypeError):
                    lines.append(line)
                    continue
                hits = [e for e in entries if isinstance(e, dict) and e.get("id") in results]
                for entry in hits:
                    entry.pop("download", None)
                    entry.update(results[entry["id"]])
                patched += len(hits)
                lines.append(dumps(record, ensure_ascii=False) + "\n" if hits else line)
        if patched:
            with open_new(path + ".tmp") as f:
                f.write("".join(lines))
                f.flush()
                fsync(f.fileno())
            replace(path + ".tmp", path)
        return patched

    def note_attachment(self, entry: dict):
        self.add_line("attachments.jsonl", entry)
        self.attachments[entry["id"]] = entry


# ========================
# Miner
# ========================
def _new_counts() -> dict:
    return {"new": 0, "images": 0, "download_failures": 0, "too_big": 0, "with_content": 0}


class Miner:
    """Lists a guild's channels and threads and stores their messages and images (see the module docstring)."""
    def __init__(self, rest: Rest, store: Store, guild_id: str, order: str = "oldest", limit: Optional[int] = None,
                 threads: bool = True, download: bool = True, log: Callable = print):
        if not is_snowflake(guild_id):
            raise ValueError(f"not a guild id: {guild_id!r}")
        self.rest, self.store, self.guild = rest, store, guild_id
        self.order, self.limit, self.threads, self.download, self.log = order, limit, threads, download, log
        self.base: Optional[int] = None         # the bot's guild permissions; None: unknown, every channel tried
        self.member: Optional[dict] = None
        self.notes: list = []
        self.raw: dict = {}                     # id -> channel object
        self.active: list = []
        self.inventory: dict = {}               # id -> channels.json entry
        self.results: list = []                 # summary.json rows
        self.totals = {"channels": 0, "threads": 0, "skipped": 0, "unchanged": 0, "errors": 0, "messages": 0,
                       **_new_counts()}
        self.started = now_iso()
        self._budget: Optional[int] = None

    # -- channels --
    def _note(self, channel: dict, **extra) -> dict:
        entry = self.inventory.setdefault(channel["id"], {"id": channel["id"]})
        meta = channel.get("thread_metadata") or {}
        found = {"name": channel.get("name"), "type": channel.get("type"),
                 "type_name": TYPE_NAMES.get(channel.get("type")), "parent_id": channel.get("parent_id"),
                 "position": channel.get("position"), "topic": channel.get("topic"),
                 "available_tags": [{"id": t.get("id"), "name": t.get("name")}
                                    for t in channel.get("available_tags") or []] or None,
                 "applied_tags": channel.get("applied_tags") or None, "owner_id": channel.get("owner_id"),
                 "message_count": channel.get("message_count"), "archived": meta.get("archived"),
                 "archive_timestamp": meta.get("archive_timestamp"), "locked": meta.get("locked"),
                 "created": meta.get("create_timestamp") or (snowflake_time(channel["id"])
                                                             if is_snowflake(channel["id"]) else None)}
        entry.update({key: value for key, value in found.items() if value is not None})
        entry.update(extra)
        return entry

    def _fetch_channel(self, channel_id: str) -> Optional[dict]:
        """A channel or thread the guild list lacks (a thread id given by hand)."""
        if not is_snowflake(channel_id):
            self._note({"id": channel_id}, status="skipped", reason="not a Discord id")
            return None
        try:
            channel = self.rest.get(f"/channels/{channel_id}")
        except ApiError as e:
            if e.fatal:
                raise
            self._note({"id": channel_id}, status="skipped", reason=e.reason())
            return None
        if channel.get("guild_id") != self.guild:
            self._note({"id": channel_id}, status="skipped", reason="not in this guild")
            return None
        self.raw[channel["id"]] = channel
        self._note(channel)
        return channel

    def discover(self, wanted: list, include_all: bool) -> list:
        """The channels to mine, in order: wanted first (categories expanded), then with include_all the rest."""
        me = self.rest.get("/users/@me")
        try:
            guild = self.rest.get(f"/guilds/{self.guild}")
        except ApiError as e:
            if e.status in (403, 404):
                raise SystemExit(f"can't read guild {self.guild} ({e.reason()}): the bot isn't a member; invite it "
                                 "with View Channel + Read Message History") from None
            raise
        try:
            self.member = self.rest.get(f"/guilds/{self.guild}/members/{me['id']}")
            self.base = base_permissions(guild, self.member)
            self.notes.append("permissions computed from roles and overwrites")
        except ApiError as e:
            if e.fatal:
                raise
            self.notes.append(f"permissions unknown ({e.reason()}): every channel tried")
        channels = self.rest.get(f"/guilds/{self.guild}/channels")
        try:
            self.active = self.rest.get(f"/guilds/{self.guild}/threads/active").get("threads") or []
        except ApiError as e:
            if e.fatal:
                raise
            self.notes.append(f"active threads unlisted: {e.reason()}")
        children: dict = {}
        for channel in channels:
            self.raw[channel["id"]] = channel
            self._note(channel)
            children.setdefault(channel.get("parent_id"), []).append(channel)
        for kids in children.values():
            kids.sort(key=lambda c: (c.get("position") or 0, int(c["id"])))
        for thread in self.active:
            if is_snowflake(thread.get("id")):
                self.raw.setdefault(thread["id"], thread)
                self._note(thread, thread_of=thread.get("parent_id"))
        plan, seen = [], set()

        def add(channel):
            if channel["id"] not in seen:
                seen.add(channel["id"])
                plan.append(channel)

        def add_tree(channel):
            if channel["type"] == CATEGORY:
                self._note(channel, status="category", reason="its channels are listed on their own")
                for kid in children.get(channel["id"], []):
                    add(kid)
            else:
                add(channel)

        for channel_id in wanted:
            channel = self.raw.get(channel_id) or self._fetch_channel(channel_id)
            if channel is not None:
                add_tree(channel)
        if include_all:
            top = children.get(None, [])
            for channel in [c for c in top if c["type"] != CATEGORY] + [c for c in top if c["type"] == CATEGORY]:
                add_tree(channel)
            for channel in sorted(channels, key=lambda c: (c.get("position") or 0, int(c["id"]))):
                if channel["type"] != CATEGORY:             # its category missing from the list
                    add(channel)
        return plan

    def _permissions(self, channel: dict) -> Optional[int]:
        if self.base is None or self.member is None:
            return None
        source = self.raw.get(channel.get("parent_id"), channel) if channel.get("type") in THREAD_TYPES else channel
        return channel_permissions(self.base, source, self.guild, self.member)

    def unreadable(self, channel: dict) -> str:
        """Why the bot can't read a channel's history ('' when it can or it can't tell)."""
        kind = channel.get("type")
        if kind not in MESSAGE_TYPES and kind not in THREAD_PARENTS:
            return f"{TYPE_NAMES.get(kind, kind)} channels hold no messages"
        permissions = self._permissions(channel)
        if permissions is None:
            return ""
        if not permissions & VIEW_CHANNEL:
            return "no View Channel"
        if not permissions & READ_HISTORY:
            return "no Read Message History"
        return ""

    def _archived(self, path: str, by_id: bool = False):
        """Pages archived threads, newest first; by_id: an id cursor."""
        before, seen = None, set()
        while True:
            body = self.rest.get(path, {"limit": PAGE, **({"before": before} if before else {})})
            page = [t for t in body.get("threads") or [] if is_snowflake(t.get("id"))]
            fresh = [t for t in page if t["id"] not in seen]
            seen.update(t["id"] for t in page)
            yield from fresh
            if not body.get("has_more") or not page:
                return
            if by_id:
                step = str(min(int(t["id"]) for t in page))
            else:                                       # 1 ms past: threads tied with the last come again
                last = (page[len(page) - 1].get("thread_metadata") or {}).get("archive_timestamp")
                step = just_after(last) if fresh else None
                if step is None or step == before:              # nothing new, or a tie filling pages: step past
                    step = last
            if not step or step == before:
                return
            before = step

    def threads_of(self, parent: dict) -> list:
        """A channel's threads, oldest first."""
        found = {t["id"]: t for t in self.active if t.get("parent_id") == parent["id"]}
        notes, permissions = [], self._permissions(parent)
        joined = ("joined private", f"/channels/{parent['id']}/users/@me/threads/archived/private", True)
        lists = [("public", f"/channels/{parent['id']}/threads/archived/public", False)]
        if permissions is not None and not permissions & MANAGE_THREADS:
            notes.append("archived private threads: only those the bot joined (no Manage Threads)")
            lists.append(joined)
        else:
            lists.append(("private", f"/channels/{parent['id']}/threads/archived/private", False))
        while lists:
            kind, path, by_id = lists.pop(0)
            try:
                for thread in self._archived(path, by_id):
                    found.setdefault(thread["id"], thread)
            except ApiError as e:
                if e.fatal:
                    raise
                notes.append(f"archived {kind} threads not listed: {e.reason()}")
                if kind == "private" and e.status == 403:       # permissions unknown: the joined ones still
                    lists.append(joined)
        threads = sorted((t for t in found.values() if is_snowflake(t.get("id"))), key=lambda t: int(t["id"]))
        for thread in threads:
            self.raw.setdefault(thread["id"], thread)
            self._note(thread, thread_of=parent["id"])
        self._note(parent, threads=len(threads), **({"thread_notes": notes} if notes else {}))
        return threads

    # -- messages --
    def _download(self, attachment: dict, container: str, message_id: str, counts: dict) -> dict:
        """Stores an image by sha256: {sha256, file} or {download: why not}."""
        known = self.store.attachments.get(attachment.get("id"))
        if known and known.get("sha256") and self.store.has_file(known["file"], known.get("bytes")):
            counts["images"] += 1
            return {"sha256": known["sha256"], "file": known["file"]}
        if not is_media_key(attachment.get("id")):
            counts["download_failures"] += 1
            return {"download": "failed: no attachment id"}
        part = join(self.store.root, "attachments", f".part-{attachment['id']}")
        try:
            if (attachment.get("size") or 0) > MAX_DOWNLOAD:
                raise TooBig(attachment["size"])
            digest, size = self.rest.fetch_file(attachment.get("url") or "", part)
            name = self.store.file_for(digest, size) or f"attachments/{digest}.{extension(attachment)}"
            if self.store.has_file(name, size):
                remove(part)
            else:
                replace(part, join(self.store.root, name))
            result = {"sha256": digest, "file": name, "bytes": size}
            counts["images"] += 1
        except TooBig:
            result = {"download": "too_big"}
            counts["too_big"] += 1
        except (ApiError, OSError) as e:
            result = {"download": f"failed: {e.reason() if isinstance(e, ApiError) else type(e).__name__}"}
            counts["download_failures"] += 1
        if exists(part):
            remove(part)
        self.store.note_attachment({"id": attachment["id"], "message_id": message_id, "container_id": container,
                                    "filename": attachment.get("filename"), "size": attachment.get("size"),
                                    "at": now_iso(), **result})
        return {key: value for key, value in result.items() if key != "bytes"}

    def _attachment(self, attachment: dict, container: str, message_id: str, counts: dict) -> dict:
        entry = {key: attachment.get(key) for key in ("id", "filename", "size", "width", "height", "content_type",
                                                       "url")}
        if self.download and is_image(attachment):
            entry.update(self._download(attachment, container, message_id, counts))
        return entry

    def _embeds(self, embeds: list, container: str, message_id: str, counts: dict,
                forward: Optional[int] = None) -> list:
        """Embed summaries; Discord-hosted images stored as media."""
        found = []
        for i, embed in enumerate(embeds or []):
            summary = embed_summary(embed)
            media = embed_media(embed, embed_key(message_id, i, forward)) if self.download else []
            if media:
                summary["media"] = [{**{k: m[k] for k in ("id", "kind", "url", "width", "height")},
                                     **self._download(m, container, message_id, counts)} for m in media]
            found.append(summary)
        return found

    def record(self, message: dict, channel: dict, parent: Optional[dict], counts: dict) -> dict:
        """One JSONL line: who, when, what, and where (thread messages carry their parent channel)."""
        container, owner = channel["id"], parent or channel
        author = message.get("author") or {}
        content = message.get("content") or ""
        counts["with_content"] += bool(content)
        record = {
            "guild_id": self.guild, "channel_id": owner["id"], "channel_name": owner.get("name"),
            "thread": {"id": channel["id"], "name": channel.get("name"), "type": channel.get("type")} if parent else None,
            "id": message["id"], "timestamp": message.get("timestamp"),
            "edited_timestamp": message.get("edited_timestamp"), "type": message.get("type"),
            "author": {"id": author.get("id"), "username": author.get("username"),
                       "global_name": author.get("global_name"), "bot": bool(author.get("bot"))},
            "content": content,
            "attachments": [self._attachment(a, container, message["id"], counts)
                            for a in message.get("attachments") or []],
            "embeds": self._embeds(message.get("embeds"), container, message["id"], counts),
            "mentions": [{"id": u.get("id"), "username": u.get("username"), "global_name": u.get("global_name")}
                         for u in message.get("mentions") or [] if isinstance(u, dict)],
            "referenced_message_id": (message.get("message_reference") or {}).get("message_id"),
            "reactions": reaction_summary(message), "pinned": bool(message.get("pinned")),
            "jump_url": f"https://discord.com/channels/{self.guild}/{container}/{message['id']}",
        }
        if message.get("webhook_id"):
            record["webhook_id"] = message["webhook_id"]
        forwarded = [snapshot.get("message") or {} for snapshot in message.get("message_snapshots") or []]
        if forwarded:
            record["forwarded"] = [{"content": f.get("content") or "", "timestamp": f.get("timestamp"),
                                    "attachments": [self._attachment(a, container, message["id"], counts)
                                                    for a in f.get("attachments") or []],
                                    "embeds": self._embeds(f.get("embeds"), container, message["id"], counts, j)}
                                   for j, f in enumerate(forwarded)]
        return record

    def _walk(self, channel: dict, parent: Optional[dict], direction: str, start: Optional[str], state: dict,
              counts: dict):
        """Pages after (forward) or before (back; start None: the newest) until done or out of budget."""
        cursor = start
        while self._budget is None or self._budget > 0:
            limit = PAGE if self._budget is None else min(PAGE, self._budget)
            page = self.rest.get(f"/channels/{channel['id']}/messages",
                                 {"limit": limit, **({direction: cursor} if cursor is not None else {})})
            if not isinstance(page, list):
                raise ApiError(200, "messages page isn't a list")
            short = len(page) < limit                   # counted before the cursor is dropped
            page = sorted((m for m in page if is_snowflake(m.get("id"))), key=lambda m: int(m["id"]),
                          reverse=direction == "before")   # file order stays complete if a write is cut short
            if cursor is not None:                      # only ids past the cursor: never a duplicate or a loop
                bound = int(cursor)
                page = [m for m in page if (int(m["id"]) > bound if direction == "after" else int(m["id"]) < bound)]
            if page:
                self.store.append(channel["id"], [self.record(m, channel, parent, counts) for m in page])
                ids = [int(m["id"]) for m in page] + [int(state[k]) for k in ("newest", "oldest") if state.get(k)]
                state.update(newest=str(max(ids)), oldest=str(min(ids)), count=state["count"] + len(page))
                counts["new"] += len(page)
                cursor = page[len(page) - 1]["id"]
                if self._budget is not None:
                    self._budget -= len(page)
            if short and direction == "before":
                state["start_reached"] = True
            if (short and direction == "after") or (direction == "before" and start is None):
                state["caught_up"] = True
            state["updated"] = now_iso()
            self.store.save_cursor(channel["id"], state)
            if short or not page:                       # a full page with nothing new: stuck, stop
                return

    def mine(self, channel: dict, parent: Optional[dict] = None) -> dict:
        """Stores a channel's or thread's new messages; a summary row."""
        channel_id = channel["id"]
        meta = channel.get("thread_metadata") or {}
        cursor = self.store.cursor(channel_id)
        count, low, high = self.store.scan(channel_id)
        row = {"id": channel_id, "name": channel.get("name"), "type_name": TYPE_NAMES.get(channel.get("type")),
               "parent_id": channel.get("parent_id"), **({"thread_of": parent["id"]} if parent else {})}
        counts = _new_counts()
        if (parent and meta.get("archived") and count and cursor.get("start_reached") and cursor.get("caught_up")
                and cursor.get("archive_timestamp") == meta.get("archive_timestamp")):
            status, reason = "unchanged", "archived thread, stored whole"
            state = cursor
        else:
            state = {"name": channel.get("name"), "newest": str(high) if high else None,
                     "oldest": str(low) if low else None, "count": count,
                     "start_reached": bool(cursor.get("start_reached")) and count > 0, "caught_up": False,
                     "archive_timestamp": meta.get("archive_timestamp")}
            status, reason = "mined", ""
            self._budget = self.limit
            try:
                if not count and self.order == "newest":
                    self._walk(channel, parent, "before", None, state, counts)
                else:
                    state["start_reached"] = state["start_reached"] or not count   # after=0 starts at the first
                    self._walk(channel, parent, "after", state["newest"] or "0", state, counts)
                    if not state["start_reached"]:
                        self._walk(channel, parent, "before", state["oldest"], state, counts)
            except ApiError as e:
                if e.fatal:
                    raise
                status = "skipped" if e.status in (403, 404) and not state["count"] else "error"
                reason = e.reason()
        row.update(status=status, **({"reason": reason} if reason else {}), messages=state.get("count", count),
                   first=snowflake_time(state.get("oldest")), last=snowflake_time(state.get("newest")),
                   start_reached=bool(state.get("start_reached")), **counts)
        self._note(channel, status=status, **({"reason": reason} if reason else {}), messages=row["messages"])
        self.results.append(row)
        for key in _new_counts():
            self.totals[key] += counts[key]
        self.totals["messages"] += row["messages"]
        self.totals["threads" if parent else "channels"] += status == "mined"
        self.totals["unchanged"] += status == "unchanged"
        self.totals["skipped"] += status == "skipped"
        self.totals["errors"] += status == "error"
        label = f"{parent.get('name')} > {channel.get('name')}" if parent else f"#{channel.get('name')}"
        self.log(f"{label}: {status}{f' ({reason})' if reason else ''}, +{counts['new']} messages "
                 f"({row['messages']} stored), {counts['images']} images")
        return row

    def _skip(self, channel: dict, reason: str):
        self._note(channel, status="skipped", reason=reason)
        self.results.append({"id": channel["id"], "name": channel.get("name"),
                             "type_name": TYPE_NAMES.get(channel.get("type")), "status": "skipped", "reason": reason})
        self.totals["skipped"] += 1
        self.log(f"#{channel.get('name')}: skipped ({reason})")

    def run(self, wanted: list, include_all: bool = False, list_only: bool = False) -> dict:
        """Discovers, then mines (or with list_only only lists) every planned channel and its threads."""
        status = "interrupted"
        try:
            plan = self.discover(wanted, include_all)
            self.log(f"{len(plan)} channels to {'list' if list_only else 'mine'}")
            done = set()                                # a thread is mined once, given by id or found with its channel
            for channel in plan:
                if channel["id"] in done:
                    continue
                reason = self.unreadable(channel)
                if reason:
                    self._skip(channel, reason)
                    continue
                done.add(channel["id"])
                if list_only:
                    self._note(channel, status="listed")
                elif channel["type"] in THREAD_TYPES:
                    self.mine(channel, self.raw.get(channel.get("parent_id")) or {"id": channel.get("parent_id")})
                elif channel["type"] in MESSAGE_TYPES:
                    self.mine(channel)
                else:
                    self._note(channel, status="threads only")
                if self.threads and channel["type"] in THREAD_PARENTS:
                    for thread in self.threads_of(channel):
                        if thread["id"] in done:
                            continue
                        done.add(thread["id"])
                        if list_only:
                            self._note(thread, status="listed")
                        else:
                            self.mine(thread, channel)
            status = "done"
        except SystemExit as e:
            status = f"stopped: {e}"
            raise
        except ApiError as e:
            status = f"stopped: {e.reason()}"
            raise
        finally:
            self.write_summary(status)
        return self.summary(status)

    def retry_downloads(self) -> dict:
        """Retries failed downloads with fresh URLs; patches the lines."""
        failed: dict = {}
        for entry in self.store.attachments.values():
            if str(entry.get("download", "")).startswith("failed"):
                failed.setdefault((entry.get("container_id"), entry.get("message_id")), set()).add(entry["id"])
        counts, stored = _new_counts(), {}
        for path in sorted(glob(join(self.store.root, "messages", "*.jsonl"))):   # failures the index lost
            container = basename(path)[:-len(".jsonl")]
            if not is_snowflake(container):
                continue
            self.store.scan(container)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        record = loads(line)
                        entries = stored_media(record)
                    except (ValueError, AttributeError, TypeError):
                        continue
                    for entry in entries:
                        if not str(entry.get("download", "")).startswith("failed"):
                            continue
                        known = self.store.attachments.get(entry.get("id")) or {}
                        if known.get("sha256") and self.store.has_file(known["file"], known.get("bytes")):
                            stored.setdefault(container, {})[entry["id"]] = {"sha256": known["sha256"],
                                                                              "file": known["file"]}
                        elif not str(known.get("download", "")).startswith("failed"):   # else listed above
                            failed.setdefault((container, record.get("id")), set()).add(entry.get("id"))
        try:
            for (container, message_id), ids in failed.items():
                if not (is_snowflake(container) and is_snowflake(message_id)):
                    continue
                try:
                    message = self.rest.get(f"/channels/{container}/messages/{message_id}")
                except ApiError as e:
                    if e.fatal:
                        raise
                    self.log(f"message {message_id}: {e.reason()}")
                    continue
                for attachment in all_media(message):
                    if attachment.get("id") in ids:
                        result = self._download(attachment, container, message_id, counts)
                        if result.get("sha256"):
                            stored.setdefault(container, {})[attachment["id"]] = result
        finally:
            for container, results in stored.items():
                self.store.patch_attachments(container, results)
        self.log(f"retried {sum(len(ids) for ids in failed.values())} downloads: {counts['images']} stored, "
                 f"{counts['download_failures']} failed")
        return counts

    # -- summary --
    def summary(self, status: str) -> dict:
        warnings = []
        if self.totals["new"] >= 50 and not self.totals["with_content"]:
            warnings.append("every new message came with empty content: is the Message Content intent off?")
        return {"guild_id": self.guild, "status": status, "started": self.started, "finished": now_iso(),
                "order": self.order, "limit": self.limit, "requests": self.rest.requests,
                "rate_limited": self.rest.limited, "waited_seconds": round(self.rest.waited, 1), "notes": self.notes,
                "warnings": warnings, "totals": self.totals, "channels": self.results}

    def write_summary(self, status: str):
        summary = self.summary(status)
        for entry in self.inventory.values():
            entry.setdefault("status", "not selected")
        self.store.write_json("summary.json", summary)
        self.store.write_json("channels.json", sorted(self.inventory.values(), key=lambda e: (
            e.get("thread_of") or e.get("parent_id") or "", e.get("position") or 0, e["id"])))
        self.store.add_line("runs.jsonl", {key: value for key, value in summary.items() if key != "channels"})


# ========================
# Command line
# ========================
def _interrupt(*_):
    raise KeyboardInterrupt


def main(argv: Optional[list] = None, session=None) -> int:
    parser = ArgumentParser(prog="python -m commands.discord_archive",
                            description="Mine a Discord guild with REST GETs only (resumable).")
    parser.add_argument("--guild", required=True)
    parser.add_argument("--channels", nargs="*", default=[], help="mined first; a category means its channels")
    parser.add_argument("--all", action="store_true", help="then every other channel")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"{DEFAULT_OUT} or a folder in it")
    parser.add_argument("--order", choices=("oldest", "newest"), default="oldest", help="a fresh channel's paging")
    parser.add_argument("--limit", type=int, help="messages a channel this run")
    parser.add_argument("--no-threads", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--list", action="store_true", help="channels and threads only, no messages")
    parser.add_argument("--retry-downloads", action="store_true")
    parser.add_argument("--config", default=CONFIG_PATH, help="the config.py holding TOKEN")
    args = parser.parse_args(argv)
    if not (args.channels or args.all or args.retry_downloads):
        parser.error("give --channels, --all or --retry-downloads")
    bad = [value for value in [args.guild, *args.channels] if not is_snowflake(value)]
    if bad:
        parser.error(f"not Discord ids: {' '.join(bad)}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    out, root = abspath(expanduser(args.out)), realpath(expanduser(DEFAULT_OUT))
    if realpath(out) != root and not realpath(out).startswith(root + sep):
        parser.error(f"--out must be {DEFAULT_OUT} or a folder in it")
    repo = inside_git(out)
    if repo:
        parser.error(f"--out is inside the git work tree {repo}: mined messages stay outside every repo")
    signal(SIGTERM, _interrupt)                     # kill: the summary still written (background jobs ignore ^C)
    rest = Rest(load_token(args.config), session)
    store = Store(out)
    miner = Miner(rest, store, args.guild, args.order, args.limit, not args.no_threads, not args.no_download)
    try:
        if args.retry_downloads:
            miner.retry_downloads()
        if args.channels or args.all:
            summary = miner.run(args.channels, args.all, args.list)
            print(f"done: {summary['totals']['new']} new messages, {summary['totals']['images']} images, "
                  f"{summary['requests']} requests ({summary['rate_limited']} rate limited); see {out}/summary.json")
    except ApiError as e:
        raise SystemExit(f"stopped: {e.reason()}") from None
    except KeyboardInterrupt:
        raise SystemExit("interrupted: rerun to resume") from None
    except Exception:
        print(rest.redact(format_exc()), file=stderr)
        return 1
    finally:
        store.close()
    return 0


# ========================
# Module Exports
# ========================
__all__ = [
    'ApiError',
    'Miner',
    'Rest',
    'Store',
    'channel_permissions',
    'base_permissions',
    'inside_git',
    'load_token',
    'main',
]


if __name__ == "__main__":
    raise SystemExit(main())
