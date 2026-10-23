"""The psd_analyzer split: the facade's re-exports, the pickled class path, the shared helpers, the classifier guard."""
import pickle
from types import SimpleNamespace

import pytest

import commands.card_edits as ce
import commands.card_models as cm
import commands.github_api as ga
import commands.psd_analyzer as pa
import commands.psd_parser as pp
import commands.repo_history as rh
import commands.repo_traverser as rt
import commands.utils as utils
from conftest import REPO

# what main.py, query_card, the diagnostics scripts, the tests and the archive branch import from the facade
FACADE_NAMES = ("CardInfo CardStats CardType StatsDatabase stats_db get_stats_as_dict card_name CardAuthors CardEdits "
                "PSDParser CardClassifier CardValidator IconVocabulary MutableValue StatTrackers TYPE_REGION_RATIO PARSER_VERSION "
                "RepositoryTraverser update_stats UPDATE_RATE EXCLUDE_FOLDERS COMMITS_PAGE LOOKUP_WORKERS GIT_TOKEN "
                "ICON_MANIFEST_PATH AUTHORS_PATH AUTHORS_PATHS EDITS_PATH EDITS_PATHS card_repo init_psd list_orphans export_stats_to_file").split()


def test_facade_reexports_the_same_objects():
    assert [n for n in FACADE_NAMES if not hasattr(pa, n)] == []
    assert pa.CardInfo is cm.CardInfo and pa.stats_db is cm.stats_db is rt.stats_db
    assert pa.PSDParser is pp.PSDParser and pa.CardEdits is ce.CardEdits and pa.update_stats is rt.update_stats
    assert "get_old_stats_as_dict" not in pa.__all__ and {"stats_db", "update_stats", "init_psd"} <= set(pa.__all__)
    assert hasattr(pa.RepositoryTraverser, "_populate_types_from_local")            # parse_one.py, gap_trace.py


def test_pickled_class_path_is_the_facade():
    card = cm.CardInfo(name="A", card_type="item", stats=cm.CardStats(hp=1))
    data = pickle.dumps({"A": card})
    assert b"commands.psd_analyzer" in data and b"card_models" not in data
    assert pickle.loads(data)["A"] == card


def test_split_modules_are_compiled_and_github_api_is_not():
    text = (REPO / "setup.py").read_text(encoding="utf-8")
    assert all(f"'commands/{m}.py'" in text for m in ("psd_analyzer", "card_models", "card_edits", "psd_parser", "repo_traverser"))
    assert "github_api" not in text


def test_classify_shallow_paths_are_unknown():
    unknown = {"type": "unknown"}
    assert pp.CardClassifier.classify("Card.psd") == unknown
    assert pp.CardClassifier.classify("Items/Card.psd") == unknown            # read out of bounds when compiled
    assert pp.CardClassifier.classify("Creatures/Card.psd") == unknown
    assert pp.CardClassifier.classify("Items/Heal/2 Stars/B.psd") == {"type": "item", "subtype": "heal", "stars": 2}
    assert pp.CardClassifier.classify("N.M.E/Star_Jar.psd") == {"type": "nme"}
    assert pp.CardClassifier.classify("Auxiliary/Minions/Mini_Doomer.psd")["type"] == "minion"


@pytest.mark.parametrize("path", ["Field/ /X.psd", "Items/Weapons//X.psd", "Creatures/Doomers/ /X.psd",
                                  "Auxiliary/Status/Debuffs/\t/X.psd"])
def test_classify_blank_star_folder_raises(path):
    with pytest.raises(ValueError, match="no star count"):      # compiled, ''.split()[0] read out of bounds
        pp.CardClassifier.classify(path)
    assert pp.CardClassifier.classify("Auxiliary/Status/Debuffs/1 Star/X.psd") == {"type": "debuff", "stars": 1}


def test_card_name_and_card_key():
    assert cm.card_name("Items/Heal/2 Stars/A_Snack.psd") == "A Snack" == cm.card_name("Items/A_Snack.png")
    assert ce.CardAuthors.key("The_Lich  King") == "the lich king"


def test_pinned_ref_and_raw_url_agree(monkeypatch):
    monkeypatch.setattr(rh, "_paths", {"Items/A b.png": ("blob1", "c0ffee")})
    assert rh.pinned_ref(rh.REPOSITORY, "Items/A b.png", "blob1") == "c0ffee"
    assert rh.pinned_ref(rh.REPOSITORY, "Items/A b.png", "blob2") is None
    assert rh.raw_url(rh.REPOSITORY, "Items/A b.png", "blob1").endswith("/c0ffee/Items/A%20b.png")
    assert rh.raw_url("someone/else", "Items/A b.png", "blob1").endswith(f"/{ga.CARD_BRANCH}/Items/A%20b.png")


def test_fetch_tree_and_blob_requests(monkeypatch):
    calls = []
    def get(url, **kw):
        calls.append((url, kw))
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, content=b"png")
    monkeypatch.setattr(ga, "get", get)
    ga.fetch_tree("o/r", '"e1"', retry=False)
    ga.fetch_tree("o/r", "", retry=False)
    assert ga.fetch_blob("o/r", "abc").content == b"png"
    assert calls[0][0] == f"{ga.API_BASE}o/r/git/trees/{ga.CARD_BRANCH}?recursive=1"
    assert calls[0][1]["headers"].get("If-None-Match") == '"e1"' and "If-None-Match" not in calls[1][1]["headers"]
    assert calls[2][0].endswith("/git/blobs/abc") and calls[2][1]["headers"]["Accept"] == ga.RAW_MEDIA


def test_say_and_queue_long_message(capsys):
    utils.dispatch_queue.clear()
    utils.say(None, "to the terminal")
    assert capsys.readouterr().out.strip() == "to the terminal"
    utils.say(SimpleNamespace(channel_id=1), "x" * 4000, ephemeral=True)
    sent = [kw for _, kw in utils.dispatch_queue]
    utils.dispatch_queue.clear()
    assert len(sent) == 3 and all(kw["ephemeral"] and len(kw["content"]) <= utils.BREAK_LEN for kw in sent)


def test_lacks_admin_polarity(monkeypatch):
    monkeypatch.setattr(utils, "ADMINS", [1])
    assert not utils.lacks_admin(SimpleNamespace(user=SimpleNamespace(id=1)))
    assert utils.lacks_admin(SimpleNamespace(user=SimpleNamespace(id=2)))
