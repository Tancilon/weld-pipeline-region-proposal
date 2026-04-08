import math
import itertools
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "scipy.optimize" not in sys.modules:
    scipy_module = types.ModuleType("scipy")
    optimize_module = types.ModuleType("scipy.optimize")

    def linear_sum_assignment(cost_matrix):
        rows = range(len(cost_matrix))
        best_perm = None
        best_cost = None
        for perm in itertools.permutations(range(cost_matrix.shape[1]), len(rows)):
            total = sum(float(cost_matrix[row, col]) for row, col in enumerate(perm))
            if best_cost is None or total < best_cost:
                best_cost = total
                best_perm = perm
        return list(rows), list(best_perm)

    optimize_module.linear_sum_assignment = linear_sum_assignment
    scipy_module.optimize = optimize_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.optimize"] = optimize_module

from networks.eomt_head import EoMTCriterion


def test_eomt_criterion_reports_mask_iou_and_mask_dice_for_matched_masks():
    criterion = EoMTCriterion(num_classes=2)

    class_logits = torch.tensor(
        [
            [
                [-8.0, 8.0, -8.0],
                [8.0, -8.0, -8.0],
            ]
        ],
        dtype=torch.float32,
    )
    mask_logits = torch.tensor(
        [
            [
                [[-10.0, -10.0], [10.0, 10.0]],
                [[10.0, 10.0], [-10.0, -10.0]],
            ]
        ],
        dtype=torch.float32,
    )
    gt_classes = [torch.tensor([0, 1], dtype=torch.long)]
    gt_masks = [
        torch.tensor(
            [
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
            ],
            dtype=torch.float32,
        )
    ]

    metrics = criterion.evaluate_batch(class_logits, mask_logits, gt_classes, gt_masks)

    assert math.isclose(metrics["mask_iou"].item(), 1.0)
    assert math.isclose(metrics["mask_dice"].item(), 1.0)
    assert math.isclose(metrics["cls_acc_matched"].item(), 1.0)
    assert math.isclose(metrics["matched_count"].item(), 2.0)
