"""
The card page's text over time (TTSCardMaker site/assets/history.js, in node): the current text dated from when it
started (cards.json since, the bot's carried start), not from the last PSD commit, in the History block, the image
versions' "Text then" and the Recently updated feed; and every date the site shows in Pacific time, the bot's TIMEZONE,
whatever the viewer's zone. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone.
"""
import json
from os import environ
from subprocess import run as run_process

import pytest

from conftest import SITE_SRC

ASSETS = SITE_SRC / "site" / "assets"
HISTORY_JS = ASSETS / "history.js"
IMAGE_AT = 1792648620   # Beldam's 2026-10-21 22:57 PT image


def node(tmp_path, body: str, tz: str = "Europe/Berlin"):
    """Run body after importing history.js as h, in a viewer's time zone; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const h = await import({json.dumps(HISTORY_JS.as_uri())});\n{body}\n", encoding="utf-8")
    out = run_process(["node", str(script)], capture_output=True, text=True, check=True, env={**environ, "TZ": tz})
    return json.loads(out.stdout)


@pytest.mark.parametrize("tz", ["Europe/Berlin"])
def test_dates_are_pacific_for_every_viewer(tmp_path, tz):
    """10-21 22:57 PT is Oct 22 in Berlin and UTC: the site says Oct 21 in every zone, and its month is October."""
    out = node(tmp_path, f"""const pt = new Date({IMAGE_AT} * 1000).toLocaleDateString(undefined, {{ year: "numeric", month: "short", day: "numeric", timeZone: "America/Los_Angeles" }});
process.stdout.write(JSON.stringify([h.date({IMAGE_AT}), pt, h.month(1793509200)]));""", tz)
    assert out[0] == out[1] and "21" in out[0]
    assert "Oct" in out[2]                                              # 2026-10-31 22:00 PT: still October



def test_a_minion_label_beside_a_blank_is_no_ability_change(tmp_path):
    """Owner 11-17 (2): a box that only said 'Protective minion.' / 'Non-protective minion.' before a blank one is no
    ability change on the card page's compare too (history.js sameAbility), as the timelines say (discord_chain
    label_blank); a label to the other label (Whelp) or real text beside the label (Galacta Warrior) stays a change."""
    from commands.discord_chain import label_blank
    pairs = [("Non-Protective minion.", ""), ("Non-Protective Minion", "Non-Protective minion."), ("", "Protective minion."),
             (" protecting minion ", ""), ("Non protective minion", ""), ("Protective minion.", "Non-protective minion."),
             ("Protective minion. Summoned by Galacta Knight.", ""), ("Protective minion.", "Chomp Chomp"),
             ("Chomp Chomp", ""), ("Same text.", "Same text."), ("", ""), ("A minion.", "")]
    same = node(tmp_path, f"process.stdout.write(JSON.stringify({json.dumps(pairs)}.map(([x, y]) => h.sameAbility(x, y))));")
    assert same == [x == y or label_blank(x, y) for x, y in pairs]
    assert same == [True] * 5 + [False] * 4 + [True, True, False]


def test_compare_modes_like_github_s_image_diff(tmp_path):
    """Owner 11-17 ~20:35 (compare.js): a pinned base and a picked version in 2-up (the default), Swipe (a slider with
    its aria values, keys and pointer place), Onion skin (an opacity range) or Difference (CSS mix-blend-mode, no canvas
    reads); both fit the box of the taller one; each picture's own size only when both are known and differ; the last
    mode kept (on this page too). ~23:50 (1): an old version beside today's card (no pin) has the modes too, the old one
    picked over the current; a side with no picture stays 2-up with a note."""
    compare = ASSETS / "compare.js"
    out = node(tmp_path, f"""const m = await import({json.dumps(compare.as_uri())});
const a = {{ pic: '<img src="a.webp">', cap: "May 2, 2019", when: "May 2, 2019", base: true }};
const b = {{ pic: '<img src="b.webp">', cap: "Oct 29, 2023", when: "Oct 29, 2023" }};
const html = mode => m.compareView(mode, a, b, m.boxRatio([[588, 1172], [400, 612]]), {{ swipe: 30, onion: 70 }});
const old = {{ pic: '<img src="o.webp">', cap: "Jul 28, 2020", when: "Jul 28, 2020", own: [400, 612] }};
const cur = {{ pic: '<img src="c.avif">', cap: "Current", when: "current" }};
process.stdout.write(JSON.stringify({{ two: html(null), swipe: html("swipe"), onion: html("onion"), diff: html("diff"),
  now: m.compareView("onion", old, cur, "2 / 3"), sized: m.compareView("2up", old, {{ ...old, own: [399, 611] }}, "2 / 3"),
  lack: m.compareView("diff", old, {{ ...cur, pic: '<div class="empty">No picture</div>', none: true }}, "2 / 3"),
  modes: [m.modeOf("bad"), m.modeOf("diff")], ratio: [m.boxRatio([null, [400, 612]]), m.boxRatio([null, null])],
  keys: [m.slideKey(50, "ArrowLeft"), m.slideKey(50, "ArrowUp", true), m.slideKey(98, "ArrowRight"), m.slideKey(50, "Home"),
         m.slideKey(50, "End"), m.slideKey(50, "PageDown"), m.slideKey(50, "a")],
  at: [m.pctAt(150, 100, 200), m.pctAt(0, 100, 200), m.pctAt(400, 100, 200)],
  sizes: [m.sizesText([588, 1172], [400, 612]), m.sizesText([400, 612], [400, 612])], aria: m.swipeAria(40, "x", "y") }}));""")
    two, swipe = out["two"], out["swipe"]
    assert 'data-mode="2up"' in two and two.count('aria-pressed="true"') == 1 and 'data-mode="2up" aria-pressed="true"' in two
    assert 'role="group" aria-label="Compare view"' in two and [l for l in ("2-up", "Swipe", "Onion skin", "Difference") if f">{l}<" in two] == ["2-up", "Swipe", "Onion skin", "Difference"]
    assert "--ar:588 / 1172" in two and "--x:30%" in two and "--op:0.7" in two
    slider = swipe[swipe.index('role="slider"') - 30:]
    assert 'tabindex="0"' in slider and 'aria-valuemin="0" aria-valuemax="100" aria-valuenow="30"' in slider
    assert 'aria-valuetext="30%: the May 2, 2019 version left, the Oct 29, 2023 version right"' in slider
    assert 'tabindex="-1"' in two[two.index('role="slider"') - 30:] and '" hidden><span class="ver-grip"' in two
    assert 'type="range" class="ver-onion" min="0" max="100" step="1" value="70"' in out["onion"]
    assert 'aria-label="Opacity of the picked Oct 29, 2023 version over the base May 2, 2019 version"' in out["onion"]
    assert 'ver-pic ver-a ver-is-base' in out["diff"] and 'ver-pic ver-b ver-is-pick' in out["diff"]
    assert out["modes"] == ["2up", "diff"] and out["ratio"] == ["400 / 612", "2 / 3"]
    assert out["keys"] == [45, 51, 100, 0, 100, 25, None] and out["at"] == [25, 0, 100]
    assert out["sizes"] == [["588 × 1172 px", "400 × 612 px"], None]
    assert out["aria"] == {"aria-valuenow": "40", "aria-valuetext": "40%: the x version left, the y version right"}
    now = out["now"]                                     # no pin: the old version picked over today's card
    assert 'ver-pic ver-a ver-is-pick' in now and 'ver-pic ver-b ver-is-base' in now and 'class="ver-cmp" data-mode="onion"' in now
    assert 'aria-label="Opacity of the picked Jul 28, 2020 version over the current version"' in now
    assert '<span class="ver-size"></span>' in now and "400 × 612 px</span>" in out["sized"] and "399 × 611 px</span>" in out["sized"]
    assert 'class="ver-cmp ver-lack" data-mode="diff"' in out["lack"]
    css, app = (ASSETS / "style.css").read_text(encoding="utf-8"), (ASSETS / "app.js").read_text(encoding="utf-8")
    assert '.ver-cmp[data-mode="diff"]:not(.ver-lack) .ver-is-pick { mix-blend-mode: difference; }' in css
    assert ".ver-lack:not([data-mode=\"2up\"]) .ver-lack-note { display: inline;" in css
    assert "touch-action: pan-y" in css and "getImageData" not in compare.read_text(encoding="utf-8")
    assert app.count("compareView(cmpMode ?? pref(MODE_KEY)") == 2 and app.count("nowPair(") == 2 and "ver-pair" not in app
    assert "box.dataset.mode = cmpMode = mode" in app and "localStorage.setItem(MODE_KEY, mode)" in app


def test_compare_texts_load_with_the_versions_list():
    """history.json (the compare's "Text then" and changes list) starts once a card's Image versions list is drawn, not on
    the first pick."""
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    fill = app[app.index("const fillVersions = () => {"):app.index("const heavyPaths = ")]
    assert fill.index("versionsHtml(c, versionsOpen(c.id))") < fill.index("if (c.hv && !c.history) loadHistory().catch(() => {});")
