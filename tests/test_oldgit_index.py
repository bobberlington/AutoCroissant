"""scripts/oldgit_index.py (by hand): the first repo's authors and its cards joined to the old repo's."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "oldgit_index.py"


@pytest.fixture(scope="module")
def oi():
    spec = importlib.util.spec_from_file_location("oldgit_index", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authors_map_by_a_shared_email(oi):
    meta = {"a": {"author": "panda", "email": "p@x"}, "b": {"author": "Kitty", "email": "p@x"},
            "c": {"author": "Stranger", "email": "s@x"}}
    got = oi.mapped_authors(meta, {"_source": "owner", "panda": "Michael"})
    assert got == {"panda": "Michael", "Kitty": "Michael"}             # no e-mail match: unmapped


def test_a_first_repo_card_joins_the_old_card_of_its_last_name(oi):
    events = [{"lineage": "F1", "path": "Cards/Kirby/3 Stars/Magolor.png"},
              {"lineage": "F1", "path": "Creatures/Kirby/3 Stars/Magolor.png"},
              {"lineage": "F2", "path": "Items/Heal/2 Stars/Grapes2.png"},
              {"lineage": "F3", "path": "Items/Attack/2 Stars/Bomb.png"},
              {"lineage": "L1", "path": "Creatures/Kirby/3 Stars/Magolor.png"},
              {"lineage": "L2", "path": "Items/Heal/2 Stars/Grapes.png"},
              {"lineage": "L3", "path": "Items/Attack/2 Stars/Bomb.png"},
              {"lineage": "L4", "path": "Auxillary/Items/2 Stars/Bomb.png"}]
    got = oi.join_first({}, lambda x: x, events, {})
    assert got == {"F1": "L1", "F3": "L3"}                             # a name and its folder; never a near name
    meme = [{"lineage": "F9", "path": "Cards/Meme/3 Stars/Something.png"},
            {"lineage": "L9", "path": "Creatures/Omori/3 Stars/SOMETHING.png"}]
    assert oi.join_first({}, lambda x: x, meme, {}) == {}                # another folder: another card
