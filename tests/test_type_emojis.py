"""type_emojis: icon selection from a tree, and sync_changed_emojis against a fake Discord client."""
import asyncio
from io import BytesIO

import pytest
from PIL import Image

import commands.type_emojis as te
from helpers import Client


def _png(color=(200, 30, 30, 255)) -> bytes:
    out = BytesIO()
    Image.new("RGBA", (16, 16), color).save(out, "PNG")
    return out.getvalue()


# ========================
# _icons
# ========================
def test_icons():
    shas = {"Types/Fire.png": "f1", "Types/fire.PNG": "f2", "Types/Stars/Dst Star.png": "s1",
            "Types/Fragile Equip.png": "e1", "Types/N.M.E.png": "n1", "Types/Sub/x.png": "x", "Types/readme.txt": "r",
            "Items/Fire.png": "i", "Types/Stars/Old/y.png": "y"}
    assert te._icons(shas) == {
        "fire": ("Types/Fire.png", "f1", False), "fire_no": ("Types/Fire.png", "f1", True),     # sorted: first wins
        "fragile_equip": ("Types/Fragile Equip.png", "e1", False),
        "fragile_equip_no": ("Types/Fragile Equip.png", "e1", True),
        "nme": ("Types/N.M.E.png", "n1", False), "nme_no": ("Types/N.M.E.png", "n1", True),
        "dst_star": ("Types/Stars/Dst Star.png", "s1", False),                                 # stars: no _no
    }


def test_long_names_fit_with_suffix():
    name = te.emoji_name("An Extremely Long Icon Name That Keeps Going")
    assert len(name) <= te.MAX_NAME_LEN and not name.endswith("_")
    assert len(te._no_name(name)) <= te.MAX_NAME_LEN and te._no_name(name).endswith("_no")


# ========================
# sync_changed_emojis
# ========================
@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(te, "EMOJI_JSON", str(tmp_path / "type_emojis.json"))
    monkeypatch.setattr(te, "_blobs", {})
    monkeypatch.setattr(te, "_saved", {})
    monkeypatch.setattr(te, "_blobs_loaded", True)
    monkeypatch.setattr(te, "type_emojis", {})
    monkeypatch.setattr(te, "_icon_png", lambda repository, path, sha: _png())
    monkeypatch.setattr(te, "_lock", asyncio.Lock())
    return te


def _sync(client, shas, failed):
    assert te._client is client
    asyncio.run(te.sync_changed_emojis(None, "owner/repo", shas, failed.append))


def test_sync_changed_emojis(state, monkeypatch, capsys):
    client = Client(["fire", "fire_no", "gone", "gone_no"])
    monkeypatch.setattr(te, "_client", client)
    te._blobs.update({"fire": "old", "fire_no": "old", "gone": "g", "gone_no": "g"})
    te.type_emojis.update({e.name: str(e) for e in client.emojis})
    failed = []
    _sync(client, {"Types/Fire.png": "new", "Types/Water.png": "w1", "Items/x.png": "i"}, failed)
    assert failed == []
    assert sorted(client.created) == ["fire", "fire_no", "water", "water_no"]
    assert sorted(client.deleted) == ["fire", "fire_no", "gone", "gone_no"]
    assert te._blobs == {"fire": "new", "fire_no": "new", "water": "w1", "water_no": "w1"}
    assert set(te.type_emojis) == {"fire", "fire_no", "water", "water_no"}
    assert capsys.readouterr().out.strip() == "Type emojis: 2 updated, 2 new, 2 removed"
    assert "water" in open(te.EMOJI_JSON).read()

    fetches = client.fetches
    _sync(client, {"Types/Fire.png": "new", "Types/Water.png": "w1"}, failed)
    assert client.fetches == fetches and failed == []                  # nothing changed: no Discord call


def test_sync_before_startup_sync(state, monkeypatch):
    monkeypatch.setattr(te, "_client", None)
    failed = []
    asyncio.run(te.sync_changed_emojis(None, "owner/repo", {"Types/Fire.png": "f"}, failed.append))
    assert failed == ["owner/repo"]                                    # next /update_stats retries


def test_sync_fetch_failure(state, monkeypatch, capsys):
    client = Client(fail_fetch=True)
    monkeypatch.setattr(te, "_client", client)
    failed = []
    _sync(client, {"Types/Fire.png": "f"}, failed)
    assert failed == ["owner/repo"] and client.created == []
    assert "Type emoji sync failed: 503 Service Unavailable" in capsys.readouterr().out


def test_render_type_tokens(monkeypatch):
    monkeypatch.setattr(te, "type_emojis", {"fire": "<:fire:1>", "fragile_equip": "<:fragile_equip:2>"})
    out = te.render_type_tokens("[fire] [Fragile Equip] [fire:no] [X] [Status] *b*", escape=True)
    assert out == "<:fire:1> <:fragile_equip:2> 🚫<:fire:1> [X] [Status] \\*b\\*"

