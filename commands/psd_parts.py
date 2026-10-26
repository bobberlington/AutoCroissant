"""
A card's parts: one box per PSD layer the website edits (docs/Finished/session-2026-10-26/PARTS_EDITOR.md §2): the title layers, the ability
layers, a rule page's text blocks, the layers that follow them (copies, watermarks, a relaid hidden Ability), the
variant pictures and, for a card without an ability or a type icon, its kind's template. The selection is the parser's
(PSDParser.title_parts, ability_parts, blocks, under psd_parser.variant_view); this module only assembles the record.
No Discord or config; psd_edit and psd_details are imported where used (psd_edit imports this module).

    card_parts(psd, rel, name, parser, card=card, templates=Templates.from_dir(clone))
"""
from hashlib import sha1
from io import BytesIO
from os.path import isfile, join
from typing import Callable, Optional

from psd_tools import PSDImage

from commands.psd_parser import (EFFECT_LAYER_TAG, PSDParser, base_name, layer_tag, name_key,
                                 title_text, variant_view, variants_of)
from global_config import RULEBOOK_DIR

# ========================
# Configuration
# ========================
TEMPLATE_DIR = "MDW/Templates"
TEMPLATES = {"creature": "Creature_Template", "minion": "Creature_Template", "creatures": "Creature_Template",   # owner 10-25
             "item": "Item_Template", "aux item": "Item_Template", "nme": "Item_Template", "field": "Item_Template",
             "buff": "Status_Effect_Template", "debuff": "Status_Effect_Template"}
ACTIVE_TEMPLATE = "ActiveItem_Template"     # Items/Active


# ========================
# Helper Functions
# ========================
def raw_sha(layer) -> str:
    """The layer's raw text hash: what a part edit checks before it bakes (Photoshop changed it: skipped)."""
    return sha1((layer.text or "").encode("utf-8")).hexdigest()[:8]


def layer_path(layer) -> str:
    """"Group/Sub/Layer" by name, tags kept."""
    names = []
    while layer is not None and getattr(layer, "kind", "psdimage") != "psdimage":
        names.append(layer.name)
        layer = layer.parent
    return "/".join(reversed(names))


def lyid_index(psd) -> dict:
    """{layer id: layer}; raises psd_edit.Refused("integrity") when two layers share an id (-1, none stored, is left out)."""
    out = {}
    for layer in psd.descendants():
        if layer.layer_id < 0:
            continue
        if layer.layer_id in out:
            from commands.psd_edit import Refused
            raise Refused("integrity", f"two layers have the id {layer.layer_id}: {layer_path(out[layer.layer_id])!r} "
                                       f"and {layer_path(layer)!r}")
        out[layer.layer_id] = layer
    return out


def template_stem(card_type: str, subtype: Optional[str] = None) -> Optional[str]:
    """The MDW/Templates file stem a card of this kind takes a first ability or type icon from; None: no template."""
    if card_type == "item" and (subtype or "").lower() == "active":
        return ACTIVE_TEMPLATE
    return TEMPLATES.get(card_type)


class Templates:
    """
    The templates' facts, read once each: {"ability": {box, font, size} | None, "types": {spot} | None}.

    Args:
        open_rel: repo path -> the open PSD, or None when it can't be had
    """
    def __init__(self, open_rel: Callable[[str], Optional[PSDImage]]):
        self.open_rel = open_rel
        self.cache: dict = {}

    @classmethod
    def from_dir(cls, root: str) -> "Templates":
        """Templates from a checkout of the card repo."""
        return cls(lambda rel: PSDImage.open(join(root, rel)) if isfile(join(root, rel)) else None)

    @classmethod
    def from_blobs(cls, read: Callable[[str], Optional[bytes]]) -> "Templates":
        """Templates from a reader of repo paths' bytes (the build's blob store)."""
        def open_rel(rel):
            data = read(rel)
            return PSDImage.open(BytesIO(data)) if data else None
        return cls(open_rel)

    def facts(self, stem: str, vocab) -> Optional[dict]:
        if stem in self.cache:
            return self.cache[stem]
        rel = f"{TEMPLATE_DIR}/{stem}.psd"
        try:
            psd = self.open_rel(rel)
        except Exception as e:                  # an unreadable template: no facts, never a failed card
            print(f"{rel}: not read ({e})")
            psd = None
        facts = None
        if psd is not None:
            from commands.psd_details import _style, _type_header      # lazy: psd_details imports psd_edit
            layer = next((l for l in psd.descendants() if l.kind == "type" and base_name(l).strip().lower() == "ability"
                          and l.is_visible()), None)
            ability = None
            if layer is not None:
                font, size, _ = _style(layer)
                ability = {"box": list(layer.bbox), "font": font, "size": size}
            try:
                header = _type_header(psd, rel, stem, vocab)
            except Exception:                   # a header psd-tools can't read: no spot
                header = None
            facts = {"file": rel, "ability": ability,
                     "types": {"spot": header["row"][0]["box"]} if header and header.get("row") else None}
        self.cache[stem] = facts
        return facts


# ========================
# The record
# ========================
def _part(parser: PSDParser, layer, role: Optional[str] = None, named: Optional[bool] = None) -> dict:
    out = {"id": layer.layer_id, "layer": layer_path(layer)}
    if role is not None:
        out["role"] = role
    if named is not None:
        out["named"] = named
    out.update(text=parser.layer_text(layer), sha=raw_sha(layer), box=list(layer.bbox), variants=None)
    return out


def _shows(psd, layer) -> bool:
    from commands.psd_edit import _shows as shows      # lazy: psd_edit imports this module
    return shows(psd, layer)


def _select(psd, rel: str, name: str, parser: PSDParser) -> tuple:
    """One picture's ([title layers], [ability layers], [(block layer, role)], [(follower, [of], how)])."""
    texts = [layer for layer in psd.descendants() if layer.kind == "type" and EFFECT_LAYER_TAG not in layer.name]
    if RULEBOOK_DIR in rel:
        return [], [], parser.blocks(texts), []
    titles, copies = parser.title_parts(psd, texts, name)
    abilities, hidden = parser.ability_parts(texts)
    follow = [(layer, [twin], "copy") for layer, twin in copies]
    parts = titles + abilities
    title = name_key(" ".join(title_text(layer.text) for layer in titles))
    taken = {id(layer) for layer in parts} | {id(layer) for layer, _, _ in follow} | {id(layer) for layer in [hidden] if layer}
    for layer in texts:
        if id(layer) in taken or not layer.is_visible():
            continue
        own = parser.words(layer.text)
        twin = next((part for part in parts if own and parser.words(part.text) == own), None)
        if twin is not None and _shows(psd, layer):
            follow.append((layer, [twin], "copy"))
        elif base_name(layer).strip() == "Name" and titles and title and name_key(layer.text or "") == title:
            follow.append((layer, titles, "watermark"))
    if hidden is not None:
        follow.append((hidden, abilities, "relaid"))
    return titles, abilities, [], follow


def card_parts(psd, rel: str, name: str, parser: PSDParser, variant: Optional[int] = None, card=None,
               templates: Optional[Templates] = None) -> dict:
    """
    The details record's "parts" (PARTS_EDITOR.md §2.3).

    Args:
        psd: The open card
        rel: Its repo path
        name: Its card name (the file stem)
        parser: The build's PSDParser (its icon vocabulary reads the tokens)
        variant: None: every picture, each part's "variants" the pictures it shows in (null: all); N: only N's parts
        card: The parsed card (its kind and types decide the template); None: parsed here
        templates: Templates for a card with no ability part or type icon; None: "template" stays null

    Returns:
        {"title", "ability", "blocks", "follow", "variants", "template"}
    """
    pictures = variants_of(psd)
    tagged = len(pictures) > 1
    out = {"title": [], "ability": [], "blocks": [], "follow": [], "variants": None, "template": None}
    seen, shown, follows = {}, {}, {}
    for n in (pictures if tagged else [None]):
        with variant_view(psd, n):
            titles, abilities, blocks, follow = _select(psd, rel, name, parser)
            entries = [("title", layer, None, base_name(layer).strip() == "Name") for layer in titles]
            entries += [("ability", layer, None, base_name(layer).strip() == "Ability") for layer in abilities]
            entries += [("blocks", layer, role, True if role == "header" else base_name(layer).strip() == "Page Number"
                         if role == "page" else None) for layer, role in blocks]
            for key, layer, role, named in entries:
                if layer.layer_id not in seen:
                    seen[layer.layer_id] = _part(parser, layer, role, named)
                    out[key].append(seen[layer.layer_id])
                shown.setdefault(layer.layer_id, set()).add(n)
            for layer, of, how in follow:
                if layer.layer_id not in follows:
                    follows[layer.layer_id] = {"id": layer.layer_id, "layer": layer_path(layer),
                                               "of": [o.layer_id for o in of], "how": how, "variants": None}
                    out["follow"].append(follows[layer.layer_id])
                shown.setdefault(("follow", layer.layer_id), set()).add(n)
    if tagged:
        for key, entry in [(e["id"], e) for e in seen.values()] + [(("follow", e["id"]), e) for e in follows.values()]:
            found = sorted(shown[key])
            entry["variants"] = None if found == pictures else found
        stem = rel.rsplit("/", 1)[len(rel.rsplit("/", 1)) - 1][:-4]
        out["variants"] = {"count": len(pictures),
                           "pngs": {str(n): f"{stem}{'' if n == 1 else n}.png" for n in pictures},
                           "tagged": [{"id": layer.layer_id, "layer": layer_path(layer), "in": sorted(layer_tag(layer.name)[1])}
                                      for layer in psd.descendants() if layer_tag(layer.name)[1] is not None]}
    if RULEBOOK_DIR not in rel and templates is not None:
        if card is None:
            card = parser.parse(psd, rel, variant=1 if tagged else None)
        if not out["ability"] or not card.types:
            stem = template_stem(card.card_type, getattr(card, "subtype", None))
            facts = templates.facts(stem, parser.all_types) if stem else None
            if facts is not None:
                out["template"] = {"kind": card.card_type, "file": facts["file"],
                                   "ability": facts["ability"] if not out["ability"] else None,
                                   "types": facts["types"] if not card.types else None}
    if variant is not None and tagged:
        return {**out, **{key: [e for e in out[key] if e["variants"] is None or variant in e["variants"]]
                          for key in ("title", "ability", "blocks", "follow")}}
    return out


# ========================
# Module Exports
# ========================
__all__ = [
    'TEMPLATES',
    'Templates',
    'card_parts',
    'layer_path',
    'lyid_index',
    'raw_sha',
    'template_stem',
]
