"""Translate each main workpiece OBJ so its AABB center is at the origin,
and apply the SAME translation to its `_weld.obj` partner.

For each <name>.obj under assets/<category>/:
    main center c = (min + max) / 2 over its vertices
    main: v -> v - c
    weld: v -> v - c    (same offset, preserves relative pose)

Translation does not affect vn (directions) or vt (tex coords), so only
`v` lines are rewritten.

Originals are backed up to /tmp/genpose2_obj_backup_centered/<rel_path>.

Usage:
    python scripts/center_obj_to_origin.py            # dry run
    python scripts/center_obj_to_origin.py --apply    # write in place
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

ASSETS_DEFAULT = Path(__file__).resolve().parent.parent / "assets"
BACKUP_ROOT = Path("/tmp/genpose2_obj_backup_centered")


def parse_aabb(path: Path):
    mn = [math.inf, math.inf, math.inf]
    mx = [-math.inf, -math.inf, -math.inf]
    with path.open("r") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            if x < mn[0]: mn[0] = x
            if y < mn[1]: mn[1] = y
            if z < mn[2]: mn[2] = z
            if x > mx[0]: mx[0] = x
            if y > mx[1]: mx[1] = y
            if z > mx[2]: mx[2] = z
    return tuple(mn), tuple(mx)


def translate_obj(path: Path, offset: tuple[float, float, float]) -> str:
    ox, oy, oz = offset
    out: list[str] = []
    with path.open("r") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.startswith("v "):
                parts = stripped.split()
                x = float(parts[1]) - ox
                y = float(parts[2]) - oy
                z = float(parts[3]) - oz
                out.append(f"v {x:.6f} {y:.6f} {z:.6f}")
            else:
                out.append(stripped)
    return "\n".join(out) + "\n"


def backup(path: Path, assets_root: Path) -> Path:
    rel = path.resolve().relative_to(assets_root.resolve())
    dst = BACKUP_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(path, dst)
    return dst


def find_mains(assets_root: Path) -> list[Path]:
    return sorted(p for p in assets_root.rglob("*.obj") if not p.stem.endswith("_weld"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ASSETS_DEFAULT))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    assets_root = Path(args.assets).resolve()
    mains = find_mains(assets_root)
    print(f"Found {len(mains)} main workpiece OBJ(s) under {assets_root}")
    if not args.apply:
        print("(dry run — use --apply to commit)")

    for m in mains:
        weld = m.with_name(m.stem + "_weld.obj")
        mn, mx = parse_aabb(m)
        center = ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)
        rel = m.relative_to(assets_root)
        print(f"\n  {rel}")
        print(f"    AABB before:  min=({mn[0]:.4f},{mn[1]:.4f},{mn[2]:.4f}) "
              f"max=({mx[0]:.4f},{mx[1]:.4f},{mx[2]:.4f})")
        print(f"    center (offset) = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")

        targets = [m]
        if weld.exists():
            targets.append(weld)
        else:
            print(f"    (no _weld partner: {weld.name} not found)")

        for tgt in targets:
            tgt_rel = tgt.relative_to(assets_root)
            new_text = translate_obj(tgt, center)
            if args.apply:
                bdst = backup(tgt, assets_root)
                tgt.write_text(new_text)
                nmn, nmx = parse_aabb(tgt)
                ncenter = ((nmn[0] + nmx[0]) / 2, (nmn[1] + nmx[1]) / 2, (nmn[2] + nmx[2]) / 2)
                print(f"      wrote {tgt_rel}: new AABB min=({nmn[0]:.4f},{nmn[1]:.4f},{nmn[2]:.4f}) "
                      f"max=({nmx[0]:.4f},{nmx[1]:.4f},{nmx[2]:.4f}) "
                      f"new_center=({ncenter[0]:.4f},{ncenter[1]:.4f},{ncenter[2]:.4f}) backup={bdst}")
            else:
                pmn = (mn[0] - center[0], mn[1] - center[1], mn[2] - center[2]) if tgt is m else None
                # for weld in dry-run, recompute its own AABB then subtract offset
                tmn, tmx = parse_aabb(tgt)
                ncenter = ((tmn[0] + tmx[0]) / 2 - center[0],
                           (tmn[1] + tmx[1]) / 2 - center[1],
                           (tmn[2] + tmx[2]) / 2 - center[2])
                print(f"      would write {tgt_rel}: predicted new bbox center "
                      f"=({ncenter[0]:.4f},{ncenter[1]:.4f},{ncenter[2]:.4f})")


if __name__ == "__main__":
    main()
