"""A card PSD at a commit read by HTTP range requests (RemotePSD): the data build's and card_check's. Not the bot's:
it needs psd-tools (imported where used)."""
from re import compile as re_compile

import commands.github_api as github_api

CONTENT_RANGE = re_compile(r"bytes (\d+)-(\d+)/(\d+)")   # a 206 reply's Content-Range


class RangeError(OSError):
    """A range request answered with something else than its range (status is the HTTP code, 0 for a bad reply)."""
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class RemotePSD:
    """
    A card PSD at a commit, opened by HTTP range requests to raw.githubusercontent: psd-tools' lazy open fetches the
    header, resources and layer records (about 1% of a card), and a layer's pixels only when the parse reads them
    (stars, effect layers). Raw requests are not REST API requests.

    Args:
        repository: owner/name
        ref: A commit sha
        path: The file's repo path
        size: Its size from the tree, checked when given

    Raises:
        RangeError: no range reply (a 404: the commit is not on GitHub), another size, or a range other than asked;
        RequestException, or what psd-tools raises on the records. A read that fails after the open sets failed (the
        caller may have caught it)
    """
    def __init__(self, repository: str, ref: str, path: str, size=None, timeout: int = 60):
        from psd_tools import PSDImage
        from psd_tools.range_reader import RangeReader
        self.url, self.timeout = github_api.raw_url(repository, ref, path), timeout
        self.failed = False                     # a fetch raised (the parse may have caught it): read the PSD whole
        self.total = None                       # the file's size from the first reply
        self.reader = RangeReader(self._fetch)
        if size is not None and self.reader.size != size:
            raise RangeError(f"{self.reader.size} bytes at {ref[:8]}, the tree says {size}")
        self.psd = PSDImage.open(self.reader, lazy=True)

    def _fetch(self, start: int, stop: int) -> tuple:
        """Bytes [start, stop) and the file's size: one range request (get_with_retry's tries)."""
        try:
            reply = github_api.get_with_retry(self.url, stream=True, timeout=self.timeout,
                                              headers={**github_api.HEADERS, "Range": f"bytes={start}-{stop - 1}"})
            with reply:                         # a 200 would be the whole file: never read it here
                if reply.status_code != 206:
                    raise RangeError(f"HTTP {reply.status_code}, not a range", reply.status_code)
                found = CONTENT_RANGE.fullmatch(reply.headers.get("Content-Range", ""))
                first, last, total = (int(n) for n in found.groups()) if found else (-1, -1, -1)
                if first != start or last != min(stop, total) - 1:      # the reader's checks, here: failed is set
                    raise RangeError(f"Content-Range {reply.headers.get('Content-Range')!r} for bytes {start}-{stop - 1}")
                if self.total not in (None, total):
                    raise RangeError(f"file size changed: {self.total} -> {total}")
                data = reply.content
            if len(data) != last - start + 1:
                raise RangeError(f"{len(data)} bytes for bytes {start}-{last}")
            self.total = total
            return data, total
        except Exception:
            self.failed = True
            raise

    @property
    def fetched(self) -> int:
        """Bytes fetched so far."""
        return self.reader.fetched

    def close(self) -> None:
        self.reader.close()


# ========================
# Module Exports
# ========================
__all__ = [
    'CONTENT_RANGE',
    'RangeError',
    'RemotePSD',
]
