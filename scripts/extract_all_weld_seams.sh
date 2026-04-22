#!/usr/bin/env bash
set -e

OUT_DIR="assets/weld_seams_output"
mkdir -p "$OUT_DIR"

# Categories that do NOT need --force-close
OPEN_CATEGORIES=(bellmouth channel_steel H_beam)

# Categories that DO need --force-close
CLOSED_CATEGORIES=(square_tube cover_plate)

for cat in "${OPEN_CATEGORIES[@]}"; do
    echo "=== Processing $cat ==="
    python -m weld \
        --workpiece "assets/$cat/$cat.obj" \
        --weld "assets/$cat/${cat}_weld.obj" \
        --output "$OUT_DIR/${cat}_weld_seams.json"
    echo
done

for cat in "${CLOSED_CATEGORIES[@]}"; do
    echo "=== Processing $cat (force-close) ==="
    python -m weld \
        --workpiece "assets/$cat/$cat.obj" \
        --weld "assets/$cat/${cat}_weld.obj" \
        --output "$OUT_DIR/${cat}_weld_seams.json" \
        --force-close
    echo
done

echo "All results saved to: $OUT_DIR"
