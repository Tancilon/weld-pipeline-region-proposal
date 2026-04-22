"""Orchestration entry point for weld seam extraction."""

from __future__ import annotations

import json
import os

from weld.core import (
    build_json_output_multi,
    extract_model_name,
    load_weld_mesh,
    print_summary,
    visualize_multi,
)
from weld.strategies import get_strategy


def run_pipeline(workpiece_path, weld_path, output_path=None,
                 no_viz=False, category=None):
    model_name = extract_model_name(workpiece_path)
    if output_path is None:
        out_dir = os.path.dirname(workpiece_path) or "."
        output_path = os.path.join(out_dir, f"{model_name}_weld_seams.json")
    viz_path = output_path.replace(".json", "_fit.png")

    mesh = load_weld_mesh(weld_path)
    strategy = get_strategy(category)
    paths_data = strategy.process(mesh)

    result = build_json_output_multi(model_name, paths_data)
    print_summary(model_name, paths_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved to: {output_path}")

    if not no_viz:
        visualize_multi(paths_data, mesh, viz_path)
        print(f"Visualization saved to: {viz_path}")
