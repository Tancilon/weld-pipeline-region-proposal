# Per-Category Weld Seam Fitting Strategies Design

Date: 2026-04-22

## Overview

当前焊缝提取pipeline对所有类别使用统一算法，导致3类工件拟合质量差：
- **bellmouth**：两根直线被错拟合成 line/arc 混合 + 错误标为 closed
- **square_tube**：单段arc无法表示完整圆（13.5mm误差）
- **channel_steel / H_beam**：10-37mm大误差

重构为**按类别的Strategy类**，每个类别完全自定义pipeline，便于后续针对性精细化调优。同时把代码从 `scripts/` 移到独立的 `weld/` Python包。

## 目录重组

**新建 `weld/` 包**（替代现 `scripts/extract_weld_seams.py`）：

```
weld/
├── __init__.py
├── __main__.py                # CLI 入口，支持 python -m weld
├── pipeline.py                # run_pipeline 主入口 + strategy 分发
├── core.py                    # 共享 helper
└── strategies/
    ├── __init__.py            # get_strategy(name) 注册表
    ├── base.py                # Strategy 基类 + GenericStrategy
    ├── bellmouth.py
    ├── channel_steel.py
    ├── h_beam.py
    ├── square_tube.py
    └── cover_plate.py
```

`scripts/` 仅保留 bash：
```
scripts/
└── extract_all_weld_seams.sh
```

测试从 `tests/scripts/` 迁到 `tests/weld/`：
```
tests/weld/
├── test_core.py                # mesh io, PCA, fitting基元, JSON输出
└── test_strategies.py          # 每个策略的集成测试
```

## `weld/core.py` — 共享helper

包含所有与类别无关的原语：
- **Mesh I/O**：`_parse_obj_objects`, `load_weld_mesh`, `extract_model_name`
- **PCA**：`pca_project`, `back_project`, `_fit_circle_center`, `_resample_by_arclength`
- **中心线通用算法**：`extract_centerline`, `_extract_centerline_by_angle`
- **曲率分段**：`compute_curvature`, `segment_by_curvature`
- **拟合基元**：`fit_line_error`, `fit_arc_error`, `fit_segment`, `_make_closing_segment`
- **闭合检测**：`detect_closed`
- **JSON 输出**：`build_json_output_multi`
- **可视化**：`_interpolate_arc_2d`, `_PATH_COLORS`, `_global_pca_plane`, `_project_to_plane`, `visualize_multi`
- **摘要打印**：`print_summary`
- **常量**：`MIN_COMPONENT_VERTICES = 10`

策略类按需从 `weld.core` 导入这些原语。

## Strategy 接口

`weld/strategies/base.py`：

```python
class Strategy:
    """Base class for per-category weld seam extraction strategies."""

    def process(self, mesh: trimesh.Trimesh) -> list[dict]:
        """Take a loaded weld mesh, return a list of paths_data dicts.

        Each dict has keys: centerline_2d, plane, fitted, closed.
        This shape is consumed by build_json_output_multi and visualize_multi.
        """
        raise NotImplementedError


class GenericStrategy(Strategy):
    """Default pipeline: split → filter → per-component extract+segment+fit.

    Wraps the current (pre-refactor) behavior.
    """
    force_close: bool = False

    def process(self, mesh):
        components = mesh.split(only_watertight=False)
        components = [c for c in components if len(c.vertices) >= MIN_COMPONENT_VERTICES]
        if not components:
            raise ValueError("No valid mesh components")
        return [_process_component(c, self.force_close) for c in components]
```

`_process_component` 就是当前同名helper，搬到 `weld.core`。

## Strategy 注册表

`weld/strategies/__init__.py`：

```python
from .base import GenericStrategy
from .bellmouth import BellmouthStrategy
from .channel_steel import ChannelSteelStrategy
from .h_beam import HBeamStrategy
from .square_tube import SquareTubeStrategy
from .cover_plate import CoverPlateStrategy

_REGISTRY = {
    "bellmouth": BellmouthStrategy,
    "channel_steel": ChannelSteelStrategy,
    "h_beam": HBeamStrategy,
    "H_beam": HBeamStrategy,       # alias for legacy dir name
    "square_tube": SquareTubeStrategy,
    "cover_plate": CoverPlateStrategy,
}


def get_strategy(name: str | None) -> Strategy:
    """Return a strategy instance by name, or GenericStrategy if name is None/unknown."""
    if name is None:
        return GenericStrategy()
    cls = _REGISTRY.get(name)
    if cls is None:
        return GenericStrategy()
    return cls()
```

未知 / 未指定的category默认fallback到 GenericStrategy（不抛异常，打印warning）。

## 各策略的初始实现

### BellmouthStrategy — 两根直线

```python
class BellmouthStrategy(Strategy):
    def process(self, mesh):
        components = mesh.split(only_watertight=False)
        components = [c for c in components if len(c.vertices) >= MIN_COMPONENT_VERTICES]
        return [self._fit_straight_line(c) for c in components]

    def _fit_straight_line(self, component):
        # 每根直线：PCA取主方向，投影所有顶点到PC1，取min/max作为line端点
        pts_2d, plane = pca_project(component.vertices)
        # PC1 方向在投影后是 x 轴
        x_min, x_max = pts_2d[:, 0].min(), pts_2d[:, 0].max()
        p0 = np.array([x_min, 0.0])
        p1 = np.array([x_max, 0.0])
        error = fit_line_error(pts_2d, p0, p1)
        fitted = [{"type": "line", "points_2d": [p0, p1],
                   "indices": (0, len(pts_2d)), "fitting_error_mm": round(error, 4)}]
        centerline = np.array([p0, p1])
        return {"centerline_2d": centerline, "plane": plane,
                "fitted": fitted, "closed": False}
```

预期效果：每根直线用 1 个 `line` 段精确表示，误差 <0.5mm。

### SquareTubeStrategy — 闭合圆

```python
class SquareTubeStrategy(Strategy):
    def process(self, mesh):
        components = mesh.split(only_watertight=False)
        components = [c for c in components if len(c.vertices) >= MIN_COMPONENT_VERTICES]
        return [self._fit_closed_circle(c) for c in components]

    def _fit_closed_circle(self, component):
        pts_2d, plane = pca_project(component.vertices)
        center, radius = _fit_circle_center(pts_2d)
        # 用4段90°arc拼成完整圆：θ=0°, 90°, 180°, 270°, 360°(=0°)
        thetas = np.linspace(0, 2 * np.pi, 5)  # 0, π/2, π, 3π/2, 2π
        pts = np.array([
            [center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]
            for t in thetas
        ])
        # 所有顶点到拟合圆的最大径向偏差即为 arc 段的拟合误差
        dists = np.linalg.norm(pts_2d - center, axis=1)
        error = float(np.max(np.abs(dists - radius)))
        fitted = []
        for i in range(4):
            p0, p_end = pts[i], pts[i + 1]
            t_mid = (thetas[i] + thetas[i + 1]) / 2
            pm = np.array([center[0] + radius * np.cos(t_mid),
                           center[1] + radius * np.sin(t_mid)])
            fitted.append({"type": "arc", "points_2d": [p0, pm, p_end],
                           "indices": (0, 0), "fitting_error_mm": round(error, 4)})
        return {"centerline_2d": pts, "plane": plane,
                "fitted": fitted, "closed": True}
```

预期效果：整个圆用4段90°arc描述，形状精确。

### CoverPlateStrategy — 跑道形（当前行为已OK）

```python
class CoverPlateStrategy(GenericStrategy):
    force_close = True
```

继承 GenericStrategy 并开启 force_close，保持当前 0.76mm 的好结果。

### ChannelSteelStrategy / HBeamStrategy — 先 stub

```python
class ChannelSteelStrategy(GenericStrategy):
    """Placeholder for per-category tuning."""
    pass

class HBeamStrategy(GenericStrategy):
    """Placeholder for per-category tuning."""
    pass
```

先落地框架，后续可独立精细化而不影响其他类别。

## Pipeline 入口

`weld/pipeline.py`：

```python
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
```

## CLI 接口

`weld/__main__.py`：

```bash
python -m weld \
  --workpiece assets/bellmouth/bellmouth.obj \
  --weld assets/bellmouth/bellmouth_weld.obj \
  --category bellmouth \
  [--output ...] [--no-viz]
```

移除 `--force-close` 参数（由 strategy 内部决定），`--category` 参数替代它。

## 批处理脚本更新

`scripts/extract_all_weld_seams.sh`：

```bash
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
done

echo "All results saved to: $OUT_DIR"
```

统一调用，`--force-close` 的分类逻辑搬进策略类。

## 测试

### 迁移

`tests/scripts/test_extract_weld_seams.py` → `tests/weld/test_core.py`（通用helper测试不变，仅 import 路径更新）

### 新增

`tests/weld/test_strategies.py`：

```python
def test_get_strategy_known():
    from weld.strategies import get_strategy, BellmouthStrategy
    assert isinstance(get_strategy("bellmouth"), BellmouthStrategy)

def test_get_strategy_unknown_fallbacks_to_generic():
    from weld.strategies import get_strategy, GenericStrategy
    assert isinstance(get_strategy("unknown"), GenericStrategy)
    assert isinstance(get_strategy(None), GenericStrategy)

def test_bellmouth_strategy_two_straight_lines(tmp_path):
    # 合成双直线OBJ
    mesh = _make_two_line_mesh()  # helper in test file
    from weld.strategies import BellmouthStrategy
    paths = BellmouthStrategy().process(mesh)
    assert len(paths) == 2
    for p in paths:
        assert p["closed"] is False
        assert len(p["fitted"]) == 1
        assert p["fitted"][0]["type"] == "line"

def test_square_tube_strategy_closed_circle():
    # 合成圆环mesh
    mesh = _make_circle_mesh(radius=50)
    from weld.strategies import SquareTubeStrategy
    paths = SquareTubeStrategy().process(mesh)
    assert len(paths) == 1
    assert paths[0]["closed"] is True
    assert len(paths[0]["fitted"]) == 4
    for seg in paths[0]["fitted"]:
        assert seg["type"] == "arc"

def test_cover_plate_strategy_inherits_generic_with_force_close():
    from weld.strategies import CoverPlateStrategy
    s = CoverPlateStrategy()
    assert s.force_close is True

def test_run_pipeline_uses_category(tmp_path):
    # End-to-end via run_pipeline with --category
    ...
```

## 迁移步骤（实现顺序提示）

1. 创建 `weld/core.py`，搬运 `extract_weld_seams.py` 的所有非入口函数
2. 创建 `weld/strategies/base.py` 与 `__init__.py`，定义 `Strategy`, `GenericStrategy`, `get_strategy`
3. 创建 `weld/pipeline.py` 与 `weld/__main__.py`（CLI）
4. 添加各具体策略类（空壳 / 简单实现）
5. 迁移测试到 `tests/weld/`
6. 更新 `scripts/extract_all_weld_seams.sh`
7. 删除 `scripts/extract_weld_seams.py`
8. 跑批处理验证5个类别的输出

## 技术决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 自定义粒度 | 完全自定义 (C) | 各类别几何差异大，共用中间层反而束手束脚 |
| 类别选择机制 | CLI `--category` + fallback | 显式可控，未指定则用通用策略 |
| 文件组织 | 每类别独立文件 (B) | 便于后续独立精细化调优 |
| 顶层目录 | `weld/` 独立包 | scripts/ 只保留shell，Python代码规范化 |
| 测试迁移 | 旧路径不保留 | 保持单一事实源，避免导入混乱 |
| Bellmouth策略 | 直接PCA line端点 | 2根直线最简单直接 |
| SquareTube策略 | 4段90° arc | 单arc无法表示完整圆，4段平衡精度与段数 |
| CoverPlate策略 | GenericStrategy + force_close | 当前0.76mm误差已足够好 |
| ChannelSteel / HBeam | 先 stub 继承 Generic | 框架先立住，精细化留待后续 |

## 非目标

- 自动根据mesh几何识别category（本spec只做显式 CLI 指定）
- ChannelSteel / HBeam 的精细化实现（本spec只立架子）
- 向后兼容旧 `extract_weld_seams.py` 导入路径（直接迁移）
- 多进程 / 并行处理（当前5个类别手动跑够快）
