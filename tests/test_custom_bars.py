"""custom_bars' mask edge: one straight cut where the cells part, one shape moved by the comps where they overlap."""
import numpy as np

from commands import custom_bars as cb


def _rows(width: int, slant: bool) -> dict:
    """Premultiplied rows 0-10 of ten cells (pitch 10): cell i lit up to n, drawn left to right (a later cell on top);
    slanted cells (12 wide, one px right every second row) overlap their neighbour by 2 px, upright ones (8) part."""
    height = 20
    out = {}
    for n in range(11):
        row = np.zeros((height, width, 4), np.float32)
        for i in range(1, 11):
            colour = (200.0, 150.0, 50.0, 255.0) if i <= n else (40.0, 40.0, 40.0, 255.0)
            for y in range(height):
                x = 10 * i + (y // 2 if slant else 0)
                row[y, x:x + (12 if slant else 8)] = colour
        out[n] = row
    return out


def _rebuilt(rows: dict, edge: dict, n: int, height: int) -> np.ndarray:
    out = rows[0].copy()
    for y in range(height):
        x = edge["x"][n] if edge["kind"] == "straight" else edge["shape"][y] + edge["offsets"][n]
        out[y, :x] = rows[10][y, :x]
    return out


def test_one_mask_edge_rebuilds_every_value():
    for slant, kind in ((False, "straight"), (True, "shaped")):
        rows = _rows(140, slant)
        edge = cb._profile(cb._cuts(rows), 20, 140, 4)
        assert edge["kind"] == kind and not edge.get("near")
        assert all(np.array_equal(_rebuilt(rows, edge, n, 20), rows[n]) for n in range(11)), kind
