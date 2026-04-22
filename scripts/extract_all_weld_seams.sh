#!/usr/bin/env bash
set -e

OUT_DIR="assets/weld_seams_output"
mkdir -p "$OUT_DIR"

CATEGORIES=(bellmouth channel_steel H_beam square_tube cover_plate)

for cat in "${CATEGORIES[@]}"; do
    echo "=== Processing $cat ==="
    python -m weld \
        --workpiece "assets/$cat/$cat.obj" \
        --weld "assets/$cat/${cat}_weld.obj" \
        --output "$OUT_DIR/${cat}_weld_seams.json" \
        --category "$cat"
    echo
done

echo "All results saved to: $OUT_DIR"
