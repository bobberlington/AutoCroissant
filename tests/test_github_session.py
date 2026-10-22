"""github_api's Session per thread (keep-alive, never shared) and the blob retry, against a loopback server and fakes."""
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import requests
from requests import HTTPError

import commands.github_api as ga

REAL_CONNECT, REAL_GETADDRINFO = socket.socket.connect, socket.getaddrinfo   # before conftest's offline guard


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"                                   # keep-alive

    def do_GET(self):
        self.server.seen.append((self.client_address[1], self.path, dict(self.headers)))
        body = b"ok " + self.path.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Set-Cookie", "tracker=1; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    """A loopback HTTP/1.1 server; only 127.0.0.1 may be reached (conftest refuses all other network)."""
    def connect(sock, address):
        assert address[0] == "127.0.0.1"
        return REAL_CONNECT(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        assert host == "127.0.0.1"
        return REAL_GETADDRINFO(host, *args, **kwargs)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.seen = []
    Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


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


def test_one_connection_per_thread(server):
    srv, base = server
    assert [ga.get(f"{base}/main{i}", timeout=5).text for i in range(4)] == [f"ok /main{i}" for i in range(4)]
    _in_thread(lambda: [ga.get(f"{base}/a{i}", timeout=5).text for i in range(3)])
    _in_thread(lambda: [ga.get(f"{base}/b{i}", timeout=5).text for i in range(3)])
    ports = {}
    for port, path, _ in srv.seen:
        ports.setdefault(path[1], set()).add(port)
    assert {k: len(v) for k, v in ports.items()} == {"m": 1, "a": 1, "b": 1}           # reused within a thread
    assert len(ports["m"] | ports["a"] | ports["b"]) == 3                                # never across threads


def test_same_request_as_requests_get_and_no_cookies(server):
    srv, base = server
    headers = {"Authorization": "token x", "Accept": ga.RAW_MEDIA}
    requests.get(f"{base}/plain", headers=headers, timeout=5)
    session = _in_thread(ga.session)                                # a fresh thread: its first request
    for path in ("/s1", "/s2"):                                     # the second would carry the cookie if kept
        session.get(f"{base}{path}", headers=headers, timeout=5)
    plain, s1, s2 = (h for _, _, h in srv.seen)
    assert plain == s1 == s2 and "Cookie" not in s2


def test_fetch_blob_keeps_timeout_headers_and_stream(monkeypatch):
    calls = []
    fake = type("S", (), {"get": lambda self, url, **kw: calls.append((url, kw)) or
                          type("R", (), {"status_code": 200, "raise_for_status": lambda self: None})()})()
    monkeypatch.setattr(ga, "session", lambda: fake)                # the real Session is never patched
    ga.fetch_blob("o/r", "abc", stream=True, timeout=120)
    url, kw = calls[0]
    assert url == f"{ga.API_BASE}o/r/git/blobs/abc"
    assert kw == {"stream": True, "timeout": 120, "headers": {**ga.HEADERS, "Accept": ga.RAW_MEDIA}}


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


def test_fetch_blob_gives_up_after_3_tries_and_closes_the_stream(replies):
    queue, waits = replies
    tries = [Reply(503), Reply(503), Reply(503)]
    queue[:] = list(tries)
    with pytest.raises(HTTPError):
        ga.fetch_blob("o/r", "abc", stream=True)
    assert waits == [1, 2] and all(r.closed for r in tries)


def test_fetch_blob_no_retry_on_404(replies):
    queue, waits = replies
    queue[:] = [Reply(404)]
    with pytest.raises(HTTPError):
        ga.fetch_blob("o/r", "abc")
    assert waits == []


def test_retry_reuses_one_connection(server, monkeypatch):
    """Two 503s then a 200 on a stream: the discarded replies are closed, so the 3 tries share one connection."""
    srv, base = server
    left = [2]
    real = Handler.do_GET

    def flaky(self):
        if left[0]:
            left[0] -= 1
            self.server.seen.append((self.client_address[1], self.path, dict(self.headers)))
            body = b"busy" * 1000
            self.send_response(503)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        real(self)
    monkeypatch.setattr(Handler, "do_GET", flaky)
    monkeypatch.setattr(ga, "sleep", lambda s: None)
    reply = _in_thread(lambda: ga.get_with_retry(f"{base}/blob", stream=True, timeout=5))
    assert reply.status_code == 200 and reply.text == "ok /blob"
    assert len(srv.seen) == 3 and len({port for port, _, _ in srv.seen}) == 1
