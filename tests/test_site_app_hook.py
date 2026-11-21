"""
The desktop card app's hook (owner 2026-11-20): openCard puts an empty .card-app slot with the card's repo files in the
picture column of every dialog (Morgue cards too) and fires one "card-dialog" event with them; a browser shows
nothing. pywebview builds the app's bridge with new Function, so a page CSP must allow 'unsafe-eval'. The app is
read-only (owner 2026-11-21): it sets window.cardApp = {readOnly: true} before the site's scripts, and the site then
shows no Sign in, Edit or editor. Reads $AC_SITE_SRC.
"""
import json
import re
from subprocess import run as run_process

from conftest import SITE_SRC

SITE = SITE_SRC / "site"
CSP = re.compile(r"""<meta[^>]+http-equiv=\\?["']?Content-Security-Policy[^>]*>""", re.I)


def frames(js, start, stop):
    """The template (`) and ${} frames (open braces) still open at js[stop], scanning from the backtick at js[start]."""
    stack, i = [], start
    while i < stop:
        ch = js[i]
        if stack and stack[-1] == "`":
            if ch == "\\":
                i += 1
            elif ch == "`":
                stack.pop()
            elif js.startswith("${", i):
                stack.append(0)
                i += 1
        elif ch in "'\"":
            i += 1
            while js[i] != ch:
                i += 2 if js[i] == "\\" else 1
        elif ch == "`":
            stack.append("`")
        elif ch == "{":
            stack[-1] += 1
        elif ch == "}":
            if stack[-1]:
                stack[-1] -= 1
            else:
                stack.pop()
        i += 1
    return stack


def test_card_dialog_hook_for_the_desktop_app(tmp_path):
    app = (SITE / "assets" / "app.js").read_text(encoding="utf-8")
    start = app.index("function openCard(")
    body = app[start:app.index("\n}\n", start)]
    wrap = body[body.index('<div class="image-wrap">'):body.index('<div id="versions-slot">')]   # the picture column
    assert re.search(r'<div class="actions card-app" data-id="\$\{esc\(c\.id\)\}" data-file-psd="\$\{esc\(c\.psd \|\| ""\)\}" '
                     r'data-file-png="\$\{esc\(c\.png \|\| ""\)\}"></div>', wrap)            # empty: no whitespace inside
    tpl, slot = body.index("modal.innerHTML = `") + len("modal.innerHTML = "), body.index('<div class="actions card-app"')
    assert frames(body, tpl, slot) == ["`", 0, "`"] and body[:slot].endswith('${c.psd || c.png ? `')   # in no other ${}: Morgue too
    assert body.rstrip().endswith("appHook(c);") and body.index("modal.hidden = false") < body.rindex("appHook(c)")
    css = re.sub(r"/\*.*?\*/", "", (SITE / "assets" / "style.css").read_text(encoding="utf-8"), flags=re.S)
    assert re.search(r"\.detail \.card-app:empty\s*\{\s*display:\s*none;?\s*\}", css)
    for f in (SITE / "index.html", SITE / "build" / "build_site.py"):
        assert all("'unsafe-eval'" in m for m in CSP.findall(f.read_text(encoding="utf-8"))), f
    at = app.index("function appHook(")
    script = tmp_path / "hook.mjs"
    script.write_text("""const modal = {}, slotEl = { slot: 1 }, seen = [];
let slotNow = slotEl;                      // then a card with no PSD or PNG (a lost Morgue entry): no slot
const $ = (sel, el) => (sel === ".card-app" && el === modal ? slotNow : null);
const document = new EventTarget();
document.addEventListener("card-dialog", ({ detail: d }) => seen.push({ id: d.id, psd: d.psd, png: d.png, slot: d.slot === slotEl }));
""" + app[at:app.index("\n}\n", at) + 3] + """
appHook({ id: "a-snack2", name: "A Snack", psd: "Items/Heal/2 Stars/A_Snack.psd", png: "Items/Heal/2 Stars/A_Snack2.png" });
slotNow = null; appHook({ id: "m-uru", name: "M'uru" });
process.stdout.write(JSON.stringify(seen));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [{"id": "a-snack2", "psd": "Items/Heal/2 Stars/A_Snack.psd",
                                        "png": "Items/Heal/2 Stars/A_Snack2.png", "slot": True}]


def test_read_only_in_the_desktop_app(tmp_path):
    """No Sign in, Edit, rulebook or deck editing, profile editor or remembered passphrase when window.cardApp.readOnly."""
    assets = SITE / "assets"
    app = (assets / "app.js").read_text(encoding="utf-8")
    assert "function appReadOnly() { return !!globalThis.cardApp?.readOnly; }" in app
    load = app[app.index("function loadEditor("):]
    assert load.split("\n")[1].strip() == 'if (appReadOnly()) return Promise.reject(new Error("read-only"));'
    act = app[app.index("function act("):]
    assert "else if (editor === IDLE && !appReadOnly() && !signInFirst(name, el))" in act[:act.index("\n}\n")]
    assert re.search(r'^if \(appReadOnly\(\)\) \$\("#account"\)\.remove\(\);', app, re.M)
    acct = app[app.index("function renderAccount("):]
    assert "if (!b) return;" in acct[:acct.index("\n}\n")]                      # the boot calls it: no button, no error
    for action in ("edit-card", "rb-new", "rb-edit", "rb-bans", "deck-publish"):
        at = [m.start() for m in re.finditer(f'data-action="{action}"', app)]
        assert at and all(re.search(r'\$\{appReadOnly\(\) \? "" : `<button[^`<]*$', app[:i]) for i in at), action
    user = app[app.index("function renderUser("):]
    assert user.index("if (r.edit && appReadOnly()) { location.replace(") < user.index("if (r.edit) {")
    assert "readOnly: appReadOnly() });" in app[app.index("function renderUsers("):] and "noRemember: appReadOnly() }" in app
    assert "decks.browserHtml(deckList(), f, { ...deckCtx(), error: S.decksData.error, readOnly: appReadOnly() });" in app
    assert 'tryOpen(pass, !!$("#deck-pass-remember")?.checked, false)' in app      # no Remember box in the app
    page = (SITE / "index.html").read_text(encoding="utf-8")                        # every page's head: hidden from the first paint
    head = page[:page.index("</head>")]
    assert re.search(r'<script>[^<]*\nif \(window\.cardApp\?\.readOnly\) document\.documentElement\.dataset\.app = "read-only";[^<]*</script>', head)
    css = re.sub(r"/\*.*?\*/", "", (assets / "style.css").read_text(encoding="utf-8"), flags=re.S)
    assert re.search(r'\[data-app="read-only"\] #account\s*\{\s*display:\s*none;?\s*\}', css)
    at = app.index("function appReadOnly(")
    probe = tmp_path / "ro.mjs"
    probe.write_text(app[at:app.index("\n", at)] + f"""
const Pf = await import({json.dumps((assets / "profile.js").as_uri())}), D = await import({json.dumps((assets / "decks.js").as_uri())});
const out = {{ browser: appReadOnly() }};
globalThis.cardApp = Object.freeze({{ readOnly: true }});
out.app = appReadOnly();
for (const ro of [false, true]) out[ro] = {{ people: Pf.peopleHtml([], {{ me: null, readOnly: ro, decks: [] }}).includes("Sign in"),
  remember: D.passFormHtml({{ owner: "a", slug: "b" }}, {{ noRemember: ro }}).includes("Remember on this device"),
  publish: D.browserHtml([], {{}}, {{ readOnly: ro }}).includes("Publish") }};
process.stdout.write(JSON.stringify(out));
""", encoding="utf-8")
    done = run_process(["node", str(probe)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"browser": False, "app": True, "false": {"people": True, "remember": True, "publish": True},
                                       "true": {"people": False, "remember": False, "publish": False}}
