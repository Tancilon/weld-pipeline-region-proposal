#!/bin/bash
# Legacy wrapper for the nuclear segmentation trainer.
#
# Use scripts/train_seg_single_agent.sh for the canonical entrypoint.

set -euo pipefail

echo "scripts/train_seg.sh is deprecated; use scripts/train_seg_single_agent.sh instead." >&2
exec bash "$(dirname "$0")/train_seg_single_agent.sh" "$@"
