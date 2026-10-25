"""github_api's Session per thread (keep-alive, never shared) and the blob retry, against a loopback server and fakes."""
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
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

