"""setup.py's incremental build: a module rebuilds when its source, its .pxd or setup.py changes (not on a bare touch),
and a .so that is not a current INCLUDE_FILES build is deleted."""
import os

import pytest

pytest.importorskip("Cython")
import setup as st


def test_stale_and_orphans(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_py = tmp_path / "setup.py"
    setup_py.write_text("a\n")
    monkeypatch.setattr(st, "__file__", str(setup_py))
    (tmp_path / "commands").mkdir()
    src = tmp_path / "commands" / "a.py"
    src.write_text("x = 1\n")
    files = ["commands/a.py"]
    assert st.stale_files(files, {}) == files                     # no .so yet
    st.built_path(files[0]).write_bytes(b"")
    stamps = {files[0]: st.fingerprint(files[0])}
    assert st.stale_files(files, stamps) == []
    os.utime(src, (1, 1))                                         # a touch alone is not a change
    assert st.stale_files(files, stamps) == []
    src.write_text("x = 2\n")
    assert st.stale_files(files, stamps) == files
    stamps = {files[0]: st.fingerprint(files[0])}
    (tmp_path / "commands" / "a.pxd").write_text("cdef int x\n")
    assert st.stale_files(files, stamps) == files                 # a new .pxd
    stamps = {files[0]: st.fingerprint(files[0])}
    setup_py.write_text("b\n")
    assert st.stale_files(files, stamps) == files                 # a setup.py edit

    old = tmp_path / "commands" / "a.cpython-310-x86_64-linux-gnu.so"
    dropped = tmp_path / "commands" / "b.so"
    old.write_bytes(b""), dropped.write_bytes(b"")
    st.remove_orphans(files)
    assert not old.exists() and not dropped.exists() and st.built_path(files[0]).exists()
