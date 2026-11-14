"""discord_psds.fields_of and header_types.ncc on made-up reads and bands."""
import numpy as np

import commands.discord_psds as dp
import commands.header_types as ht


def test_a_posted_psd_s_fields():
    """Name, ability, types as the PSD has them; stars never; an item's subtypes never; an unread ability layer
    to review; stats only when every printed row is sure; an art PSD has none."""
    bars = {"stats": {k: {"v": 3, "sure": True, "row": "label"} for k in ("hp", "def", "atk", "spd")}}
    row = {"read": {"name": "Bolt", "kind": "creature", "types": ["fire"], "ability": "Zap it", "stars_drawn": 5},
           "bars": bars}
    f = dp.fields_of(row)
    assert set(f) == {"name", "ability", "types", "stats"} and f["stats"]["value"]["hp"] == 3
    item = dp.fields_of({**row, "read": {**row["read"], "kind": "item"}, "ability_from": "text layers",
                         "ability_recall": 0.5})
    assert "types" not in item and item["ability"]["confidence"] == "review"
    assert dp.fields_of({"read": {"name": "Frog", "kind": "creature", "types": [], "ability": ""}}) == {}


def test_a_missing_icon_lowers_the_header_match():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 255, (52, 60, 3)).astype(np.float32)
    assert ht.ncc(a, a) == 1.0
    b = a.copy()
    b[:26, :30] = 20.0                                   # one icon cell blank
    assert ht.ncc(a, b) < 0.5 < ht._ncc(a, b)
