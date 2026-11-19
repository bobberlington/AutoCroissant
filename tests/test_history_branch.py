"""scripts/history_branch.py (by hand): one commit of the preview's history folder for the history branch, an orphan
root first and on top of the branch's head after (fetched alone, shallow, into its own store), only that ref pushed and
never forced; a folder failing a check (review files, message keys, links, snowflakes, left-out pictures, wide or
orphan webps, a list copy no Morgue timeline names or too wide) is refused with nothing pushed."""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "history_branch.py"
PHOTO = "abcdef012345" + "6" * 52                                # a left-out channel's picture (sha256)


@pytest.fixture(scope="module")
def hb():
    spec = importlib.util.spec_from_file_location("history_branch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(where, *args) -> str:
    return subprocess.run(["git", "--git-dir", str(where), *args], capture_output=True, text=True, check=True).stdout


def archive(tmp_path, hb) -> Path:
    root = tmp_path / "archive"
    folder = root.joinpath(*hb.FOLDER)
    (folder / "a-card").mkdir(parents=True)
    (folder / "_morgue" / "lost").mkdir(parents=True)
    (root / "channels.json").write_text(json.dumps([{"id": str(n), "name": name} for n, name in enumerate(hb.dm.LEFT_OUT)]))
    (root / "attachments.jsonl").write_text(json.dumps({"container_id": "2", "sha256": PHOTO}) + "\n")
    for sub, name in (("a-card", "A Card"), ("_morgue/lost", "Lost")):
        Image.new("RGB", (100, 140), (200, 30, 30)).save(folder / sub / "P-111111111111.webp")
        entry = {"id": "D-P-111111111111", "kind": "discord", "date": "2020-07-24T19:31:07+00:00", "day": "2020-07-24",
                 "badge": "Discord", "channel": "ideas", "image": "P-111111111111.webp", "w": 100, "h": 140,
                 "fields": {"name": {"value": name}}, "changes": {}}
        t = {"card": sub, "key": name, "name": name, "entries": [entry]}
        if sub.startswith("_morgue"):                                  # its list copy (discord_preview list_copies)
            Image.new("RGB", (50, 70), (200, 30, 30)).save(folder / sub / f"P-111111111111{hb.LIST_SUFFIX}", "AVIF")
            t["list"] = f"P-111111111111{hb.LIST_SUFFIX}"
        (folder / sub / "timeline.json").write_text(json.dumps(t))
    index = {"version": 1, "cards": {"a-card": {"key": "A Card", "dir": "a-card"}}, "morgue": [{"dir": "_morgue/lost"}],
             "mdw": [], "created": {}}
    (folder / "index.json").write_text(json.dumps(index))
    return root


def test_one_commit_an_orphan_first_then_on_the_branch_head(tmp_path, hb, monkeypatch, capsys):
    for key in ("NAME", "EMAIL"):
        monkeypatch.setenv(f"GIT_AUTHOR_{key}", "t@x")
        monkeypatch.setenv(f"GIT_COMMITTER_{key}", "t@x")
    root, remote = archive(tmp_path, hb), tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    main = git(remote, "commit-tree", git(remote, "mktree").strip(), "-m", "main").strip()
    git(remote, "update-ref", "refs/heads/main", main)                     # never fetched, never pushed to
    run = ["--root", str(root), "--remote", f"file://{remote}"]
    assert hb.main([*run, "--store", str(tmp_path / "s1")]) == 0          # a dry run pushes nothing
    assert git(remote, "for-each-ref", "--format=%(refname)").split() == ["refs/heads/main"]
    assert hb.main([*run, "--store", str(tmp_path / "s1"), "--push"]) == 0
    first = git(remote, "rev-parse", "refs/heads/history").strip()
    assert git(remote, "for-each-ref", "--format=%(refname)").split() == ["refs/heads/history", "refs/heads/main"]
    assert git(remote, "log", "-1", "--format=%P|%an|%B", first).strip().endswith("Card-sweep: history")
    assert git(remote, "log", "-1", "--format=%P|%an", first).strip() == "|MichaelJSr"
    assert sorted(git(remote, "ls-tree", "-r", "--name-only", first).split()) == [
        "_morgue/lost/P-111111111111-list.avif", "_morgue/lost/P-111111111111.webp", "_morgue/lost/timeline.json",
        "a-card/P-111111111111.webp",
        "a-card/timeline.json", "index.json"]
    assert hb.main([*run, "--store", str(tmp_path / "s1"), "--push"]) == 0
    assert "unchanged" in capsys.readouterr().out
    timeline = root.joinpath(*hb.FOLDER, "a-card", "timeline.json")
    timeline.write_text(timeline.read_text().replace("ideas", "new-cards"))
    assert hb.main([*run, "--store", str(tmp_path / "s2"), "--push"]) == 0   # a new store: the head fetched shallow
    second = git(remote, "rev-parse", "refs/heads/history").strip()
    assert git(remote, "log", "-1", "--format=%P", second).strip() == first
    assert (tmp_path / "s2" / "shallow").read_text().split() == [first]
    assert git(remote, "rev-parse", "refs/heads/main").strip() == main
    assert subprocess.run(["git", "--git-dir", str(tmp_path / "s2"), "cat-file", "-e", main]).returncode != 0


@pytest.mark.parametrize("bad", ["review", "message", "link", "snowflake", "glued", "name", "photo", "wide", "orphan",
                                 "hidden", "key", "list-wide", "list-orphan", "list-card"])
def test_a_folder_failing_a_check_is_refused(tmp_path, hb, bad):
    root = archive(tmp_path, hb)
    folder = root.joinpath(*hb.FOLDER)
    timeline = folder / "a-card" / "timeline.json"
    t = json.loads(timeline.read_text())
    entry = t["entries"][0]
    if bad == "review":
        shutil.copy(timeline, folder / "items.json")
    elif bad == "message":
        entry["fields"]["name"]["message"] = "posted text"
    elif bad == "link":
        entry["subject"] = "see https://example.com"
    elif bad == "snowflake":
        entry["subject"] = "id 123456789012345678"
    elif bad == "glued":
        entry["id"] = "D-m123456789012345678"                               # no word boundary before it
    elif bad == "name":
        entry["image"] = "P-111111111111_123456789012345678.webp"
        (folder / "a-card" / "P-111111111111.webp").rename(folder / "a-card" / entry["image"])
    elif bad == "photo":
        Image.new("RGB", (10, 10)).save(folder / "a-card" / f"P-{PHOTO[:12]}.webp")
    elif bad == "wide":
        Image.new("RGB", (hb.WEBP_W + 1, 10)).save(folder / "a-card" / "P-111111111111.webp")
    elif bad == "orphan":
        Image.new("RGB", (10, 10)).save(folder / "a-card" / "P-222222222222.webp")
    elif bad == "hidden":
        (folder / "a-card" / ".DS_Store").write_bytes(b"x")
    elif bad == "key":
        entry["text"] = "a message"
    elif bad == "list-wide":
        Image.new("RGB", (hb.LIST_W + 1, 10)).save(folder / "_morgue" / "lost" / f"P-111111111111{hb.LIST_SUFFIX}", "AVIF")
    elif bad == "list-orphan":
        Image.new("RGB", (10, 10)).save(folder / "_morgue" / "lost" / f"P-222222222222{hb.LIST_SUFFIX}", "AVIF")
    elif bad == "list-card":                                            # list copies are the Morgue list's only
        Image.new("RGB", (10, 10)).save(folder / "a-card" / f"P-111111111111{hb.LIST_SUFFIX}", "AVIF")
        t["list"] = f"P-111111111111{hb.LIST_SUFFIX}"
    timeline.write_text(json.dumps(t))
    problems, _ = hb.check(str(folder), hb.left_out_ids(str(root)))
    assert problems
    if bad == "name":
        assert f"a-card/{entry['image']}: a link or snowflake in its name" in problems
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    assert hb.main(["--root", str(root), "--remote", f"file://{remote}", "--store", str(tmp_path / "s"), "--push"]) == 1
    assert git(remote, "for-each-ref") == "" and not (tmp_path / "s").exists()


def test_the_left_out_channels_are_found_by_name_or_refused(tmp_path, hb):
    root = archive(tmp_path, hb)
    assert hb.left_out_ids(str(root)) == {PHOTO[:12]}
    (root / "channels.json").write_text(json.dumps([{"id": "2", "name": "Old stuff thread"}]))
    with pytest.raises(SystemExit):
        hb.left_out_ids(str(root))
