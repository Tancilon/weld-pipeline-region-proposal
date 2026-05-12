"""Scale each main workpiece OBJ so its AABB has dims = 1, and apply the
SAME per-axis scale to its `_weld.obj` partner (preserving relative pose).

For each workpiece category dir under assets/:
    <name>.obj         -> reference; its AABB defines (sx, sy, sz) = (1/Lx, 1/Ly, 1/Lz)
    <name>_weld.obj    -> scaled by the SAME (sx, sy, sz) about the world origin

Vertices: v -> (sx*x, sy*y, sz*z)
Normals  (under non-uniform scale, normals transform by inverse-transpose):
    vn -> normalize((1/sx)*nx, (1/sy)*ny, (1/sz)*nz)
Tex coords (vt) and all other lines preserved verbatim.

Originals are backed up to /tmp/genpose2_obj_backup_scaled/<rel_path>.

Usage:
    python scripts/normalize_obj_to_unit_bbox.py            # dry run
    python scripts/normalize_obj_to_unit_bbox.py --apply    # write in place
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

ASSETS_DEFAULT = Path(__file__).resolve().parent.parent / "assets"
BACKUP_ROOT = Path("/tmp/genpose2_obj_backup_scaled")


def parse_main_aabb(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (min_xyz, max_xyz) of all vertex coordinates in the OBJ."""
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


def transform_obj(path: Path, scale: tuple[float, float, float]) -> str:
    """Return the rewritten OBJ text with v scaled and vn inverse-transpose-scaled."""
    sx, sy, sz = scale
    inv = (1.0 / sx, 1.0 / sy, 1.0 / sz)
    out: list[str] = []
    with path.open("r") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.startswith("v "):
                parts = stripped.split()
                x = float(parts[1]) * sx
                y = float(parts[2]) * sy
                z = float(parts[3]) * sz
                out.append(f"v {x:.6f} {y:.6f} {z:.6f}")
            elif stripped.startswith("vn "):
                parts = stripped.split()
                nx = float(parts[1]) * inv[0]
                ny = float(parts[2]) * inv[1]
                nz = float(parts[3]) * inv[2]
                norm = math.sqrt(nx * nx + ny * ny + nz * nz)
                if norm > 0:
                    nx /= norm; ny /= norm; nz /= norm
                out.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")
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


def find_main_objs(assets_root: Path) -> list[Path]:
    """Main workpiece OBJs = those without `_weld` in the stem."""
    out = []
    for p in assets_root.rglob("*.obj"):
        if p.stem.endswith("_weld"):
            continue
        out.append(p)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ASSETS_DEFAULT))
    ap.add_argument("--apply", action="store_true",
                    help="write changes in place; otherwise dry run")
    args = ap.parse_args()

    assets_root = Path(args.assets).resolve()
    mains = find_main_objs(assets_root)
    print(f"Found {len(mains)} main workpiece OBJ(s) under {assets_root}")
    if not args.apply:
        print("(dry run — use --apply to commit)")

    for main in mains:
        weld = main.with_name(main.stem + "_weld.obj")
        mn, mx = parse_main_aabb(main)
        Lx, Ly, Lz = mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]
        if min(Lx, Ly, Lz) <= 0:
            print(f"  SKIP {main}: degenerate AABB ({Lx},{Ly},{Lz})")
            continue
        sx, sy, sz = 1.0 / Lx, 1.0 / Ly, 1.0 / Lz
        rel = main.relative_to(assets_root)
        print(f"\n  {rel}")
        print(f"    AABB before:  L=({Lx:.4g}, {Ly:.4g}, {Lz:.4g})")
        print(f"    scale (sx,sy,sz) = ({sx:.6g}, {sy:.6g}, {sz:.6g})")

        targets = [main]
        if weld.exists():
            targets.append(weld)
        else:
            print(f"    (no _weld partner: {weld.name} not found)")

        for tgt in targets:
            new_text = transform_obj(tgt, (sx, sy, sz))
            tgt_rel = tgt.relative_to(assets_root)
            if args.apply:
                bdst = backup(tgt, assets_root)
                tgt.write_text(new_text)
                # verify new AABB for sanity
                nmn, nmx = parse_main_aabb(tgt)
                nL = (nmx[0] - nmn[0], nmx[1] - nmn[1], nmx[2] - nmn[2])
                print(f"      wrote {tgt_rel}: new AABB L=({nL[0]:.4f}, {nL[1]:.4f}, {nL[2]:.4f})  backup={bdst}")
            else:
                # dry-run: compute predicted new AABB without writing
                pmn, pmx = parse_main_aabb(tgt)
                pL = ((pmx[0] - pmn[0]) * sx, (pmx[1] - pmn[1]) * sy, (pmx[2] - pmn[2]) * sz)
                print(f"      would write {tgt_rel}: predicted AABB L=({pL[0]:.4f}, {pL[1]:.4f}, {pL[2]:.4f})")


if __name__ == "__main__":
    main()
