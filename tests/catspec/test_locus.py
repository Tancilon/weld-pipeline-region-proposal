import numpy as np
import trimesh

from catspec.locus import (
    OpenProfilePath,
    build_open_line_arc_line_arc_line_loci,
    build_closed_rounded_rect_locus,
    closed_path_gap,
    estimate_square_tube_profile,
    estimate_open_profile_paths,
    sample_locus_3d,
    symmetric_hausdorff,
    symmetric_rmse,
)


def _make_square_tube_like_mesh():
    y0 = -0.461722
    y1 = 0.5
    outer = 0.228986
    radius = 0.043545
    n = 36
    pts = []

    def add_round_rect(y):
        centers = [
            (outer - radius, outer - radius, 0.0, np.pi / 2),
            (outer - radius, -outer + radius, -np.pi / 2, 0.0),
            (-outer + radius, -outer + radius, np.pi, 3 * np.pi / 2),
            (-outer + radius, outer - radius, np.pi / 2, np.pi),
        ]
        for cx, cz, a0, a1 in centers:
            for angle in np.linspace(a0, a1, n):
                pts.append([cx + radius * np.cos(angle), y, cz + radius * np.sin(angle)])

    add_round_rect(y0)
    add_round_rect(y1)
    # Sparse bbox outliers mimic the existing square_tube.obj normalization vertices.
    pts.extend([
        [-0.5, y0, -0.5],
        [-0.5, y0, 0.5],
        [0.5, y0, -0.5],
        [0.5, y0, 0.5],
    ])
    return trimesh.Trimesh(vertices=np.array(pts), faces=np.empty((0, 3), dtype=int), process=False)


def test_estimate_square_tube_profile_uses_dense_end_face_not_sparse_bbox():
    mesh = _make_square_tube_like_mesh()

    profile = estimate_square_tube_profile(
        mesh,
        plane_axis="y",
        plane_side="min_dense",
        profile_axes=("x", "z"),
        profile_quantile=5.0,
    )

    assert abs(profile.plane_value - (-0.461722)) < 1e-6
    assert profile.min_uv[0] > -0.24
    assert profile.max_uv[0] < 0.24
    assert profile.min_uv[1] > -0.24
    assert profile.max_uv[1] < 0.24
    assert 0.025 < profile.corner_radius < 0.070


def test_build_closed_rounded_rect_locus_has_expected_topology():
    mesh = _make_square_tube_like_mesh()
    profile = estimate_square_tube_profile(mesh, "y", "min_dense", ("x", "z"), 5.0)

    locus = build_closed_rounded_rect_locus(profile)

    types = [seg["type"] for seg in locus["segments"]]
    assert types == ["line", "arc", "line", "arc", "line", "arc", "line", "arc"]
    assert locus["closed"] is True
    assert closed_path_gap(locus) < 1e-9


def test_sample_locus_3d_returns_closed_samples_in_original_axes():
    mesh = _make_square_tube_like_mesh()
    profile = estimate_square_tube_profile(mesh, "y", "min_dense", ("x", "z"), 5.0)
    locus = build_closed_rounded_rect_locus(profile)

    samples = sample_locus_3d(locus, points_per_segment=8)

    assert samples.shape[1] == 3
    assert np.allclose(samples[:, 1], profile.plane_value)
    assert np.linalg.norm(samples[0] - samples[-1]) < 1e-9


def _make_open_profile_mesh():
    pts = []
    for x in (-0.05, 0.05):
        pts.extend(
            [
                [x, 0.5, 0.43],
                [x, 0.5, -0.43],
                [x, 0.35, 0.43],
                [x, 0.35, -0.43],
                [x, -0.25, 0.30],
                [x, -0.32, 0.27],
                [x, -0.38, 0.16],
                [x, -0.38, -0.16],
                [x, -0.32, -0.27],
                [x, -0.25, -0.30],
            ]
        )
    pts.extend(
        [
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
        ]
    )
    return trimesh.Trimesh(vertices=np.array(pts), faces=np.empty((0, 3), dtype=int), process=False)


def test_estimate_open_profile_paths_uses_internal_planes():
    mesh = _make_open_profile_mesh()

    profiles = estimate_open_profile_paths(
        mesh,
        plane_axis="x",
        plane_values="dense_internal",
        profile_axes=("y", "z"),
        path_count=2,
    )

    assert [profile.plane_value for profile in profiles] == [-0.05, 0.05]
    assert all(profile.profile_axes == ("y", "z") for profile in profiles)
    assert all(profile.start_pos[0] < 0.4 for profile in profiles)


def test_build_open_line_arc_line_arc_line_loci_have_expected_topology():
    profiles = [
        OpenProfilePath(
            plane_axis="x",
            plane_value=-0.05,
            profile_axes=("y", "z"),
            start_pos=(0.35, 0.43),
            arc1_start=(-0.25, 0.30),
            arc1_mid=(-0.32, 0.27),
            arc1_end=(-0.38, 0.16),
            arc2_start=(-0.38, -0.16),
            arc2_mid=(-0.32, -0.27),
            arc2_end=(-0.25, -0.30),
            end_neg=(0.35, -0.43),
        )
    ]

    loci = build_open_line_arc_line_arc_line_loci(profiles)

    assert len(loci) == 1
    assert loci[0]["closed"] is False
    assert [segment["type"] for segment in loci[0]["segments"]] == ["line", "arc", "line", "arc", "line"]
    samples = sample_locus_3d(loci[0], points_per_segment=8)
    assert samples.shape[1] == 3
    assert np.allclose(samples[:, 0], -0.05)
    assert np.linalg.norm(samples[0] - samples[-1]) > 0.1


def test_symmetric_metrics_are_zero_for_identical_paths():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    assert symmetric_rmse(pts, pts) == 0.0
    assert symmetric_hausdorff(pts, pts) == 0.0
