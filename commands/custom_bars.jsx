#target photoshop
// custom_bars.py's Photoshop half. PLAN: {"log", "step": "build"|"finish", "margin", "cards": [plan.json, ...]}.
// build: each stat's Dark and Light rows (its row documents merged, cut to its band if it has one) become one smart
// object in the stat's group, the old layers deleted; inside it the Light group's mask, unlinked, moved per value
// with a layer comp "0".."10"; the comp set to the value. finish: each bar redrawn by Photoshop at its value, saved;
// every comp saved as a PNG copy.
var PLAN = /*PLAN*/;
var LOG = new File(PLAN.log); LOG.encoding = "UTF-8"; LOG.lineFeed = "Unix"; LOG.open("w");
var DD = app.displayDialogs; app.displayDialogs = DialogModes.NO;
var RU = app.preferences.rulerUnits; app.preferences.rulerUnits = Units.PIXELS;
function cT(s) { return charIDToTypeID(s); }
function sT(s) { return stringIDToTypeID(s); }
function q(s) { return '"' + String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/[\r\n]/g, " ") + '"'; }
function note(card, k, v) { LOG.writeln('{"card":' + q(card) + ',"' + k + '":' + q(v) + '}'); }
function px(u) { return Math.round(Number(u.as ? u.as("px") : u)); }
// groups by name all the way (layerSets), or a layer at the end (artLayers): as custom_bars.find
function find(doc, path, layer) {
  var l = doc, parts = path.split("/");
  for (var i = 0; i < parts.length; i++) l = (layer && i === parts.length - 1 ? l.artLayers : l.layerSets).getByName(parts[i]);
  return l;
}
function selectId(id, add) {
  var r = new ActionReference(); r.putIdentifier(cT('Lyr '), id);
  var d = new ActionDescriptor(); d.putReference(cT('null'), r);
  if (add) d.putEnumerated(sT('selectionModifier'), sT('selectionModifierType'), sT('addToSelection'));
  d.putBoolean(cT('MkVs'), false);
  executeAction(cT('slct'), d, DialogModes.NO);
}
function target() { var r = new ActionReference(); r.putEnumerated(cT('Lyr '), cT('Ordn'), cT('Trgt')); return r; }
function compIds() {
  var r = new ActionReference(); r.putProperty(sT('property'), sT('smartObjectMore'));
  r.putEnumerated(cT('Lyr '), cT('Ordn'), cT('Trgt'));
  var list = executeActionGet(r).getObjectValue(sT('smartObjectMore')).getObjectValue(sT('compInfo')).getList(sT('compList'));
  var ids = {};
  for (var i = 0; i < list.count; i++) { var c = list.getObjectValue(i); ids[c.getString(sT('name'))] = c.getInteger(sT('ID')); }
  return ids;
}
function placedAt() {
  var r = new ActionReference(); r.putProperty(sT('property'), sT('smartObjectMore'));
  r.putEnumerated(cT('Lyr '), cT('Ordn'), cT('Trgt'));
  var t = executeActionGet(r).getObjectValue(sT('smartObjectMore')).getList(sT('transform'));
  return [Math.round(t.getDouble(0)), Math.round(t.getDouble(1))];
}
function setComp(doc, layer, name) {
  doc.activeLayer = layer;
  var ids = compIds();
  if (ids[String(name)] === undefined) throw new Error(layer.name + " has no comp " + name);
  var d = new ActionDescriptor(); d.putReference(cT('null'), target()); d.putInteger(sT('compID'), ids[String(name)]);
  executeAction(sT('setPlacedLayerComp'), d, DialogModes.NO);
}
function moveMask(dx) {
  var s = new ActionDescriptor(), sr = new ActionReference(); sr.putEnumerated(cT('Chnl'), cT('Chnl'), cT('Msk '));
  s.putReference(cT('null'), sr); s.putBoolean(cT('MkVs'), false); executeAction(cT('slct'), s, DialogModes.NO);
  var m = new ActionDescriptor(); m.putReference(cT('null'), target());
  var o = new ActionDescriptor(); o.putUnitDouble(cT('Hrzn'), cT('#Pxl'), dx); o.putUnitDouble(cT('Vrtc'), cT('#Pxl'), 0);
  m.putObject(cT('T   '), cT('Ofst'), o);
  executeAction(cT('move'), m, DialogModes.NO);
}
function maskFromSelection() {
  var mk = new ActionDescriptor(); mk.putClass(cT('Nw  '), cT('Chnl'));
  var mr = new ActionReference(); mr.putEnumerated(cT('Chnl'), cT('Chnl'), cT('Msk ')); mk.putReference(cT('At  '), mr);
  mk.putEnumerated(cT('Usng'), cT('UsrM'), cT('RvlS'));
  executeAction(cT('Mk  '), mk, DialogModes.NO);
  var ul = new ActionDescriptor(); ul.putReference(cT('null'), target());
  var to = new ActionDescriptor(); to.putBoolean(sT('userMaskLinked'), false); ul.putObject(cT('T   '), cT('Lyr '), to);
  executeAction(cT('setd'), ul, DialogModes.NO);
}
function save(doc, path) {
  var o = new PhotoshopSaveOptions(); o.layers = true; o.embedColorProfile = true;
  doc.saveAs(new File(path), o, true, Extension.LOWERCASE);
}
function row(card, T, stat, state, s) {
  var R = app.open(new File(card.folder + "/R" + state + "_" + stat + ".psd"));
  R.activeLayer = find(R, s.group);            // Merge Visible needs a visible layer selected
  executeAction(sT('mergeVisible'), new ActionDescriptor(), DialogModes.NO);
  var m = R.activeLayer; m.name = state === "F" ? "Light" : "Dark";
  if (s.band) {                                // only the stat's band of the card
    var b = s.band;
    R.selection.select([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], SelectionType.REPLACE, 0, false);
    R.selection.invert(); R.selection.clear(); R.selection.deselect();
  }
  var dup = m.duplicate(T, ElementPlacement.PLACEATBEGINNING);
  R.close(SaveOptions.DONOTSAVECHANGES);
  app.activeDocument = T;
  return dup;
}
// the mask's offset at comp n, relative to comp 10's, and its edge (doc x per row) at comp 10
function edgeAt(s, n, X0, W, m) {
  var e = s.edge;
  if (e.kind === "straight") return n === 10 ? 0 : (n === 0 ? -(W + 2 * m + 1) : (e.x[String(n)] - X0) - W);
  return e.offsets[String(n)] - e.offsets["10"];
}
function bar(card, T, stat) {
  var s = card.stats[stat], G = find(T, s.group), old = [], i;
  for (i = 0; i < G.layers.length; i++) old.push(G.layers[i]);
  var light = row(card, T, stat, "F", s), dark = row(card, T, stat, "E", s);
  light.move(G, ElementPlacement.PLACEATEND);
  dark.move(G, ElementPlacement.PLACEATEND);
  for (i = 0; i < old.length; i++) old[i].remove();
  if (s.extra) find(T, s.extra, true).remove();
  selectId(light.id, false); selectId(dark.id, true);
  executeAction(sT('newPlacedLayer'), undefined, DialogModes.NO);
  var so = T.activeLayer; so.name = s.name;
  var b0 = so.bounds, at = placedAt(), X0 = at[0], Y0 = at[1];   // the transform's corner: bounds round outward
  executeAction(sT('placedLayerEditContents'), new ActionDescriptor(), DialogModes.NO);
  var E = app.activeDocument, W = px(E.width), H = px(E.height);
  var m = s.edge.kind === "shaped" ? s.edge.margin : 0;
  if (m) E.resizeCanvas(UnitValue(W + 2 * m, "px"), UnitValue(H, "px"), AnchorPosition.MIDDLECENTER);
  var Wd = W + 2 * m, L = E.layers.getByName("Light");
  var g = E.layerSets.add(); g.name = "Light"; L.move(g, ElementPlacement.INSIDE); E.activeLayer = g;
  var pts = [[0, 0]];
  if (s.edge.kind === "straight") pts = [[0, 0], [Wd, 0], [Wd, H], [0, H]];
  else {
    for (var r = 0; r < H; r++) {
      var k = Math.max(0, Math.min(s.edge.shape.length - 1, Y0 + r - s.edge.top));
      var x = Math.max(0, Math.min(Wd, s.edge.shape[k] + s.edge.offsets["10"] - X0 + m));
      pts.push([x, r]); pts.push([x, r + 1]);
    }
    pts.push([0, H]);
  }
  E.selection.select(pts, SelectionType.REPLACE, 0, false);
  maskFromSelection();
  E.selection.deselect();
  var cur = 0, offs = [];
  for (var n = 10; n >= 0; n--) {
    var t = edgeAt(s, n, X0, W, m);
    if (t !== cur) { moveMask(t - cur); cur = t; }
    offs.push(n + ":" + t);
    E.layerComps.add(String(n), "", false, true, false);
  }
  E.layerComps.getByName("10").apply();
  E.close(SaveOptions.SAVECHANGES);
  app.activeDocument = T;
  var b1 = so.bounds, dx = px(b0[0]) - px(b1[0]), dy = px(b0[1]) - px(b1[1]);
  if (dx || dy) { so.translate(dx, dy); note(card.stem, stat + " moved back", dx + "," + dy); }
  var at1 = placedAt();
  note(card.stem, stat + " placed", at1[0] + "," + at1[1] + " bounds " + px(b0[0]) + "," + px(b0[1]));
  setComp(T, so, s.value);
  note(card.stem, stat, "doc " + W + "x" + H + " margin " + m + " at " + X0 + "," + Y0 + " comps " + offs.join(" "));
}
function buildCard(card) {
  var T = app.open(new File(card.folder + "/T.psd"));
  try {
    var order = ["hp", "def", "atk", "spd"];
    for (var i = 0; i < order.length; i++) bar(card, T, order[i]);
    for (i = 0; i < card["delete"].length; i++) find(T, card["delete"][i]).remove();
    for (i = 0; i < card.empty.length; i++) { var e = find(T, card.empty[i]); if (e.layers.length === 0) e.remove(); else note(card.stem, "kept", card.empty[i]); }
    save(T, card.folder + "/built.psd");
    note(card.stem, "built", "ok");
  } finally { T.close(SaveOptions.DONOTSAVECHANGES); }
}
function finishCard(card) {
  var T = app.open(new File(card.folder + "/tagged.psd"));
  try {
    var order = ["hp", "def", "atk", "spd"], sos = [], i, n;
    for (i = 0; i < order.length; i++) {
      var s = card.stats[order[i]], so = find(T, s.group + "/" + s.name, true);
      sos.push(so);
      setComp(T, so, s.value === 10 ? 9 : 10);
      setComp(T, so, s.value);
    }
    save(T, card.folder + "/final.psd");
    note(card.stem, "final", "ok");
    var png = new PNGSaveOptions(); png.compression = 1; png.interlaced = false;
    T.saveAs(new File(card.folder + "/ps_saved.png"), png, true, Extension.LOWERCASE);
    for (n = 0; n <= 10; n++) {
      for (i = 0; i < sos.length; i++) setComp(T, sos[i], n);
      T.saveAs(new File(card.folder + "/ps_" + n + ".png"), png, true, Extension.LOWERCASE);
    }
    note(card.stem, "comps", "ok");
  } finally { T.close(SaveOptions.DONOTSAVECHANGES); }
}
for (var c = 0; c < PLAN.cards.length; c++) {
  var card = PLAN.cards[c], before = app.documents.length;
  try { if (PLAN.step === "build") buildCard(card); else finishCard(card); }
  catch (err) { note(card.stem, "err", err + " line " + err.line); }
  while (app.documents.length > before) {
    try { app.activeDocument.close(SaveOptions.DONOTSAVECHANGES); } catch (e) { note(card.stem, "closeerr", e); break; }
  }
}
app.preferences.rulerUnits = RU; app.displayDialogs = DD;
LOG.writeln('{"end":1,"docs":' + app.documents.length + '}'); LOG.close();
