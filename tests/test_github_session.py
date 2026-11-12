"""github_api's Session per thread (keep-alive, never shared) and the blob retry, against fakes."""
from threading import Thread

import pytest
from requests import HTTPError

import commands.github_api as ga


def _in_thread(func):
    out = []
    worker = Thread(target=lambda: out.append(func()))
    worker.start()
    worker.join(30)
    return out[0]


# ========================
# Tests
# ========================
def test_session_is_per_thread():
    mine = ga.session()
    assert ga.session() is mine
    others = [_in_thread(ga.session) for _ in range(3)]
    assert len({id(s) for s in others + [mine]}) == 4


class Reply:
    def __init__(self, status):
        self.status_code, self.content, self.closed = status, b"psd", False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"{self.status_code} Server Error")

    def close(self):
        self.closed = True


@pytest.fixture
def replies(monkeypatch):
    queue, waits = [], []
    monkeypatch.setattr(ga, "get", lambda url, **kw: queue.pop(0))
    monkeypatch.setattr(ga, "sleep", waits.append)
    return queue, waits


def test_fetch_blob_retries_a_5xx(replies):
    queue, waits = replies
    tries = [Reply(502), Reply(503), Reply(200)]
    queue[:] = tries
    assert ga.fetch_blob("o/r", "abc", stream=True).content == b"psd" and waits == [1, 2] and queue == []
    assert [r.closed for r in tries] == [True, True, False]           # discarded replies free their connection

