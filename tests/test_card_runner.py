"""card_runner: the startup sweep of a cut-off run's temp PSDs under bake_lock."""
import os
import time

import commands.card_runner as cr


def test_startup_sweeps_stale_temp_psds(tmp_path, capsys):
    """A cut-off run's tmp*.psd (mkstemp's names) older than an hour go, in one line; young ones, other names and
    folders stay; nothing goes while update_stats or a font check holds bake_lock."""
    old = time.time() - 2 * cr.TEMP_AGE
    names = ("tmpab_c1234.psd", "tmpzz99yy88.psd", "tmpyoung123.psd", "tmpab_c1234.png", "card.psd", "tmpshort.psd")
    for name in names:
        (tmp_path / name).write_bytes(b"x" * 1000)
        if name != "tmpyoung123.psd":
            os.utime(tmp_path / name, (old, old))
    (tmp_path / "tmpdir12345.psd").mkdir()
    os.utime(tmp_path / "tmpdir12345.psd", (old, old))
    cr.bake_lock.acquire()
    try:
        assert cr.sweep_temp_psds(str(tmp_path)) == 0 and len(list(tmp_path.iterdir())) == 7
    finally:
        cr.bake_lock.release()
    assert cr.sweep_temp_psds(str(tmp_path)) == 2 and not cr.bake_lock.locked()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(set(names) - {"tmpab_c1234.psd", "tmpzz99yy88.psd"}
                                                                | {"tmpdir12345.psd"})
    assert capsys.readouterr().out == f"Removed 2 stale temp PSDs (0 MB) from {tmp_path}\n"
    assert cr.sweep_temp_psds(str(tmp_path)) == 0 and capsys.readouterr().out == ""
