import json, sys, os
SRC = "/Users/michaelsrouji/Desktop/TTSCardMaker/site/dist/data/rules.json"
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference")   # this skill's reference/
OUT = os.path.join(REF, "rulebook.md")
rules = {x["name"]: x for x in json.load(open(SRC))["rules"]}

ORDER = ["0000 Cover", "0000 Cover Book", "00 Table of Contents", "01 Getting Started", "02 Abbreviations",
 "03 Setting Up", "04 Table", "04 Table2", "04 Table3", "04 Table4", "04 Table5", "05 Shops",
 "06 Reading Cards", "06 Reading Cards2", "07 Stats1", "07 Stats2", "07 Stats3", "07 Stats4", "07 Stats5",
 "08 Global Turn", "09 TakingYourTurn", "10 Attacking", "10 Attacking2", "11 Abilities", "11 Abilities2",
 "11 Abilities3", "11 Abilities4", "11 Abilities5", "11 Abilities6", "11 Abilities7", "12 Reactions",
 "12 Reactions2", "12 Reactions3", "12 Reactions4", "12 Reactions5", "13 Items", "14 Item Types",
 "14 Item Types2", "15 Minions", "16 The Field", "16 The Field2", "17 Death", "18 Field Cards",
 "19 Status Effects", "19 Status Effects2", "20 Creature Types", "21 Damage Types", "22 Negation"]
assert sorted(ORDER) == sorted(rules), set(rules) ^ set(ORDER)

# (old, new) edits applied to the rules.json text. Each must match exactly once.
EDITS = {
 "00 Table of Contents": [("        If you see sleepy Kirby", "[icon: sleepy Kirby] If you see sleepy Kirby")],
 "08 Global Turn": [("SPD: 5: 3", "SPD: 5 [icon: ★]: 3"), ("SPD: 6: 4", "SPD: 6 [icon: ★]: 4"),
                    ("SPD: 3: 5", "SPD: 3 [icon: ★]: 5"), ("SPD: 5: 5", "SPD: 5 [icon: ★]: 5"),
                    ("SPD: 4: 4", "SPD: 4 [icon: ★]: 4"), ("SPD: 3: 3", "SPD: 3 [icon: ★]: 3")],
 "09 TakingYourTurn": [
   ("(See “ATTACKING” for more info)\nA creature can choose any creature on the playing field to ATTACK. Attacking removes HP from the chosen creature equal to the ATTACK VALUE of the attacking creature, minus the DEFENSE VALUE of the creature being attacked. (See “ATTACKING” for more info)\n",
    "(See “ATTACKING” for more info)\n[hidden duplicate layer omitted]\n"),
   ("This state is singified by turning the creature 90 degrees.\nA creature can use its action to go into DEFENSE MODE, doubling its defense while it is in this state. It goes out of defense mode at the beginning of its next turn. This state is singified by turning the creature 90 degrees.\n",
    "This state is singified by turning the creature 90 degrees.\n[hidden duplicate layer omitted]\n")],
 "11 Abilities": [
   ("\nPASSIVES\nPassive abilities are abilites that are in constant effect (WOF) “while the creature is on the field”. “(WOF)” is often only included when the ability effects other creatures as to signfy that it goes away when the creature extis the field.\n11",
    "\n[hidden layers omitted: a PASSIVES heading and paragraph, superseded by 11 Abilities7]\n11")],
 "11 Abilities6": [("ABILITIES 7\nABILITIES 6\n", "[off-canvas layer omitted: ABILITIES 7]\nABILITIES 6\n")],
 "12 Reactions": [("instead. of 5.", "instead. [overflow, not shown on page: of 5.]")],
 "14 Item Types": [("May not be used on [support].", "May not be used on [icon: machine]."),
                   ("various effects.  if you find this text dont delete it", "various effects. [overflow, not shown on page: if you find this text dont delete it]"),
                   ("\n14 [infliction]", "\n14")],
 "14 Item Types2": [("\n14.2 [fragile equip] [active] [deploy]", "\n14.2")],
 "15 Minions": [("TYPES:Offensive minions", "TYPES:\n[icon: offensive_minion]:Offensive minions"),
                ("(See: “FIELDS”):Protective Minions", "(See: “FIELDS”)\n[icon: defensive_minion]:Protective Minions"),
                ("\n15 [defensive_minion]", "\n15")],
 "16 The Field": [("designated by this symbol:", "designated by this symbol: [icon: defensive_minion]"),
                  ("\n16 [defensive_minion]", "\n16")],
 "19 Status Effects2": [("applied to a single creature.", "applied to a single creature. [icon: buff] [icon: debuff]"),
                        ("any number of themselves applied.", "any number of themselves applied. [icon: buffstack] [icon: debuffstack]"),
                        ("\nUPGRADABLE\n", "\nUPGRADABLE [icon: debuffupgrade]\n"),
                        ("\n19.2 [debuff]", "\n19.2")],
 "20 Creature Types": [("Gain 1 additional point (OD)\n\nThis may only be an innate type in special cases, but may still be added to creatures through effects.\n",
                        "Gain 1 additional point (OD)\n[overflow, not shown on page: This may only be an innate type in special cases, but may still be added to creatures through effects.]\n"),
                       ("\n20 [water] [electric] [undead] [light] [unknown] [girl]", "\n20")],
}

HIDDEN_2 = None  # filled below

def apply(name, text):
    for old, new in EDITS.get(name, []):
        n = text.count(old)
        if n != 1:
            sys.exit(f"{name}: edit matched {n} times: {old[:60]!r}")
        text = text.replace(old, new)
    return text

# 12 Reactions2 and 22 Negation: hidden blocks are removed by exact cut (long).
def cut(name, text, start, end_incl, label):
    i = text.index(start); j = text.index(end_incl, i) + len(end_incl)
    return text[:i] + label + text[j:], text[i:j]

NOTES = __import__("notes").NOTES  # per-page markdown annotations

def fmt(note):
    lines, res = note.split("\n"), []
    for ln in lines:
        prev = res[-1] if res else ""
        starts = ln.startswith(("Icons:", "Diagram:", "Notes:"))
        listy = ln.startswith(("- ", "1. ", "2. ")) and not prev.startswith(("- ", "1. ", "2. ")) and prev != ""
        endlist = prev.startswith(("- ", "1. ", "2. ", "3. ")) and not ln.startswith(("- ", "1. ", "2. ", "3. ", "|"))
        if res and (starts or listy or endlist) and prev != "":
            res.append("")
        res.append(ln)
    return "\n".join(res)

out = []
out.append(open(os.path.join(REF, "header.md")).read().rstrip() + "\n")
for name in ORDER:
    r = rules[name]
    t = r["ability"]
    removed = None
    if name == "12 Reactions2":
        t, removed = cut(name, t, "Although reactions happen in reverse order, anything that deals damage resolves last. Meaning all other effects take place before damage is dealt. Attacks also resolve forward",
                         "lastly the bomb deals 2.5 damage. So Toy Bonnie takes 3 damage total.\n", "[hidden layer omitted: an older draft of the 12 Reactions3 text; see Notes]\n")
    if name == "22 Negation":
        t, removed = cut(name, t, "- If an item is negated, it means it cannot be played or is taken out of play, and it returns to the hand,",
                         "this sends them back to the hand, requiring star points to be used to reactivate them.\n",
                         "[hidden layer omitted: an older draft of the ITEMS bullets; see Notes]\n")
    t = apply(name, t)
    pdir = r["psd"].rsplit("/", 1)[1][:-4]
    out.append(f"## {name}\n")
    meta = f"`{r['psd']}` (id `{r['id']}`"
    if r.get("types"):
        meta += ", rules.json types " + ", ".join(r["types"])
    if r.get("problems"):
        meta += ", problems " + "; ".join(r["problems"])
    meta += ")"
    out.append(meta + "\n")
    out.append("```text\n" + t.rstrip("\n") + "\n```\n")
    if removed and name == "12 Reactions2":
        out.append("Hidden layer text (not shown on the page), kept for reference:\n\n```text\n" + removed.strip("\n") + "\n```\n")
    if removed and name == "22 Negation":
        out.append("Hidden layer text (not shown on the page), kept for reference:\n\n```text\n" + removed.strip("\n") + "\n```\n")
    if name in NOTES:
        out.append(fmt(NOTES[name].strip()) + "\n")
out.append(open(os.path.join(REF, "footer.md")).read().rstrip() + "\n")
open(OUT, "w").write("\n".join(out))
print("wrote", OUT)
