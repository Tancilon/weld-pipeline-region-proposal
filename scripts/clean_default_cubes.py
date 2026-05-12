"""Remove leftover Blender default-cube noise from OBJ assets.

Detects connected components by face-shared-vertex adjacency and drops any
component whose signature matches a Blender default cube:
    8 vertices, 12 triangular faces, all bbox dims <= CUBE_DIM_MAX.

Also drops any vertex not referenced by a surviving face (cleans up stray
orphan verts like `v 0 0 0`).

Preserves all other OBJ lines (mtllib, usemtl, o, g, s, vn, vt, comments)
and renumbers `v` indices in face statements to match the new vertex order.
Face statements with vt/vn references keep those indices unchanged because
vt/vn arrays are not pruned.

Run with --dry-run to preview, or --apply to write back. Originals are
backed up to /tmp/genpose2_obj_backup/<relative-path>.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ASSETS_DEFAULT = Path(__file__).resolve().parent.parent / "assets"
BACKUP_ROOT = Path("/tmp/genpose2_obj_backup")

CUBE_DIM_MAX = 5.0  # default cube is 2 units on a side; allow some slack


def parse_face_indices(token: str) -> tuple[int, str]:
    """Return (1-based vertex index, original token) for a face reference."""
    v_str = token.split("/", 1)[0]
    return int(v_str), token


def load_obj(path: Path):
    """Parse the OBJ into a list of (kind, payload) records preserving order."""
    records = []  # each: (kind, data)
    verts = []  # list of (x, y, z)
    faces = []  # list of list of 1-based v-indices
    face_record_idx = []  # index in `records` for each face
    vert_record_idx = []  # index in `records` for each v line

    with path.open("r") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.startswith("v "):
                parts = stripped.split()
                xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
                vert_record_idx.append(len(records))
                records.append(("v", (xyz, stripped)))
                verts.append(xyz)
            elif stripped.startswith("f "):
                tokens = stripped.split()[1:]
                v_idx = []
                for tok in tokens:
                    vi, _ = parse_face_indices(tok)
                    v_idx.append(vi)
                face_record_idx.append(len(records))
                records.append(("f", (tokens, stripped)))
                faces.append(v_idx)
            else:
                records.append(("other", stripped))
    return records, verts, faces, vert_record_idx, face_record_idx


def connected_components(num_verts: int, faces: list[list[int]]):
    """Union-find over vertices via face-shared adjacency. Returns parent map."""
    parent = list(range(num_verts))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for face in faces:
        # face indices are 1-based
        anchor = face[0] - 1
        for vi in face[1:]:
            union(anchor, vi - 1)
    return [find(i) for i in range(num_verts)]


def detect_noise_components(verts, faces, comp_root):
    """Return set of vertex indices (0-based) belonging to default-cube noise components."""
    comp_verts = defaultdict(list)  # root -> [vert_idx]
    for vi, root in enumerate(comp_root):
        comp_verts[root].append(vi)
    comp_faces = defaultdict(list)  # root -> [face_idx]
    for fi, face in enumerate(faces):
        root = comp_root[face[0] - 1]
        comp_faces[root].append(fi)

    noise_v = set()
    noise_components = []
    for root, vlist in comp_verts.items():
        flist = comp_faces.get(root, [])
        # Blender default cube = 8 verts (one per corner) + faces (6 quads or 12 tris)
        # connecting only those 8. Real workpiece components have many more verts.
        if len(vlist) != 8:
            continue
        xs = [verts[v][0] for v in vlist]
        ys = [verts[v][1] for v in vlist]
        zs = [verts[v][2] for v in vlist]
        dims = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        if max(dims) > CUBE_DIM_MAX:
            continue
        noise_v.update(vlist)
        cx = sum(xs) / 8
        cy = sum(ys) / 8
        cz = sum(zs) / 8
        noise_components.append({
            "n_verts": 8,
            "n_faces": len(flist),
            "dims": dims,
            "centroid": (cx, cy, cz),
        })
    return noise_v, noise_components


def clean_obj(path: Path):
    """Return (new_text, report) for the cleaned OBJ. Does not write."""
    records, verts, faces, vert_record_idx, face_record_idx = load_obj(path)
    if not faces:
        return None, {"path": path, "skipped": True, "reason": "no faces"}
    comp_root = connected_components(len(verts), faces)

    noise_v, noise_components = detect_noise_components(verts, faces, comp_root)

    # Drop faces that touch any noise vertex
    noise_face_idx = set()
    for fi, face in enumerate(faces):
        if any((vi - 1) in noise_v for vi in face):
            noise_face_idx.add(fi)

    kept_face_v_set = set()
    for fi, face in enumerate(faces):
        if fi in noise_face_idx:
            continue
        for vi in face:
            kept_face_v_set.add(vi - 1)

    # Any vertex not referenced by a kept face is orphan -> drop too
    drop_v = set(noise_v)
    for vi in range(len(verts)):
        if vi not in kept_face_v_set:
            drop_v.add(vi)

    # Build old-1based -> new-1based vertex remap (None if dropped)
    remap = {}
    new_idx = 0
    for vi in range(len(verts)):
        if vi in drop_v:
            continue
        new_idx += 1
        remap[vi + 1] = new_idx

    # Emit new file content
    out_lines: list[str] = []
    cur_v = 0
    cur_f = 0
    for kind, data in records:
        if kind == "v":
            # vertices are emitted in original order but skip dropped ones
            old_idx_1b = cur_v + 1
            cur_v += 1
            if (old_idx_1b - 1) in drop_v:
                continue
            _, original_line = data
            out_lines.append(original_line)
        elif kind == "f":
            old_idx = cur_f
            cur_f += 1
            if old_idx in noise_face_idx:
                continue
            tokens, _ = data
            new_tokens = []
            for tok in tokens:
                parts = tok.split("/")
                old_v = int(parts[0])
                new_v = remap[old_v]
                parts[0] = str(new_v)
                new_tokens.append("/".join(parts))
            out_lines.append("f " + " ".join(new_tokens))
        else:
            out_lines.append(data)

    new_text = "\n".join(out_lines) + "\n"

    report = {
        "path": path,
        "skipped": False,
        "n_verts_in": len(verts),
        "n_verts_out": len(verts) - len(drop_v),
        "n_faces_in": len(faces),
        "n_faces_out": len(faces) - len(noise_face_idx),
        "n_noise_cubes": len(noise_components),
        "n_orphan_verts": len(drop_v) - len(noise_v),
        "noise_components": noise_components,
    }
    return new_text, report


def fmt_report(r):
    if r.get("skipped"):
        return f"  SKIP {r['path']}: {r['reason']}"
    rel = r["path"]
    parts = [f"  {rel}"]
    parts.append(
        f"    verts {r['n_verts_in']}->{r['n_verts_out']}  "
        f"faces {r['n_faces_in']}->{r['n_faces_out']}  "
        f"cubes_removed={r['n_noise_cubes']}  "
        f"orphan_verts_removed={r['n_orphan_verts']}"
    )
    for nc in r["noise_components"]:
        cx, cy, cz = nc["centroid"]
        dx, dy, dz = nc["dims"]
        parts.append(
            f"      cube: dims=({dx:.2g},{dy:.2g},{dz:.2g}) centroid=({cx:.3g},{cy:.3g},{cz:.3g})"
        )
    return "\n".join(parts)


def backup(path: Path, assets_root: Path) -> Path:
    rel = path.resolve().relative_to(assets_root.resolve())
    dst = BACKUP_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(path, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ASSETS_DEFAULT), help="assets dir to scan")
    ap.add_argument("--apply", action="store_true", help="write changes in place (otherwise dry run)")
    args = ap.parse_args()

    assets_root = Path(args.assets).resolve()
    obj_paths = sorted(assets_root.rglob("*.obj"))

    print(f"Scanning {len(obj_paths)} OBJ files under {assets_root}")
    if not args.apply:
        print("(dry run — no files written; pass --apply to commit changes)")

    changed = 0
    for path in obj_paths:
        new_text, report = clean_obj(path)
        if report.get("skipped"):
            print(fmt_report(report))
            continue
        delta = (
            report["n_verts_in"] != report["n_verts_out"]
            or report["n_faces_in"] != report["n_faces_out"]
        )
        if not delta:
            print(f"  CLEAN {path.relative_to(assets_root)}")
            continue
        changed += 1
        print(fmt_report(report))
        if args.apply:
            bdst = backup(path, assets_root)
            path.write_text(new_text)
            print(f"    wrote (backup: {bdst})")

    print(f"\n{'Applied' if args.apply else 'Would change'}: {changed} file(s)")


if __name__ == "__main__":
    sys.exit(main())
