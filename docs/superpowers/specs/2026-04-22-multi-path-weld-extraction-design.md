# Multi-Path Weld Seam Extraction Design

Date: 2026-04-22

## Overview

扩展 `scripts/extract_weld_seams.py` 使其支持**多条不连通路径**的焊缝。当前pipeline假设焊缝是单一连通路径，把多组件点云混合处理会导致几何失真（bellmouth的两根独立直线被错误拟合为弧线）。

5类工件的连通组件分布：

| 类别 | 组件数 | 几何 |
|---|---|---|
| bellmouth | 2 | 两根独立直线 |
| channel_steel | 2 | 两条独立焊缝 |
| H_beam | 2 | 两条独立焊缝 |
| square_tube | 1 | 单个闭合圆 |
| cover_plate | 1 | 单个跑道形 |

## 输入/输出

**输入**（不变）：
- 工件OBJ路径
- 焊缝OBJ路径（可能含多个不连通的mesh组件）

**输出**：
- `{model}_weld_seams.json` — 焊缝描述JSON（schema变更）
- `{model}_weld_seams_fit.png` — 可视化（多path叠加）

**新JSON schema**：

```json
{
  "model": "bellmouth",
  "coord_system": "raw",
  "weld_paths": [
    {
      "closed": false,
      "segments": [
        {"type": "line", "points": [[x,y,z], [x,y,z]], "fitting_error_mm": 0.15}
      ]
    },
    {
      "closed": false,
      "segments": [
        {"type": "arc", "points": [[x,y,z], [x,y,z], [x,y,z]], "fitting_error_mm": 0.32}
      ]
    }
  ]
}
```

与旧schema的区别：
- `weld_seams` → `weld_paths`（结构变成嵌套）
- 顶层 `closed` 字段删除，迁移到每条path内部
- 每条path有独立的 `segments` 数组

## 数据流

```
load_weld_mesh (选最大object，不变)
  → mesh.split(only_watertight=False)       # 新增：按连通组件拆分
  → 过滤组件：vertex count < 10 丢弃         # 新增
  → for each component:
      → pca_project (每个组件独立PCA)
      → extract_centerline
      → segment_by_curvature
      → fit_segment
      → [可选] force-close
  → build_json_output (多path聚合)
  → visualize (多path叠加)
```

## 模块变更

### 1. `run_pipeline` 重构

当前顺序处理一个mesh，改为循环处理各连通组件：

```python
mesh = load_weld_mesh(weld_path)
components = mesh.split(only_watertight=False)
components = [c for c in components if len(c.vertices) >= 10]

paths_data = []
for comp in components:
    pts_2d, plane = pca_project(comp.vertices)
    centerline = extract_centerline(comp, pts_2d)
    segments = segment_by_curvature(centerline)
    fitted = [fit_segment(s) for s in segments]
    if force_close and not detect_closed(centerline):
        fitted.append(_make_closing_segment(fitted, centerline))
    paths_data.append({
        "centerline_2d": centerline,
        "plane": plane,
        "fitted": fitted,
        "mesh_component": comp,
        "closed": detect_closed(centerline) or force_close,
    })

result = build_json_output_multi(model_name, paths_data)
visualize_multi(paths_data, mesh, output_path)
```

### 2. 组件过滤

- 过滤顶点数 < 10 的组件（避免mesh噪声产生的微小碎片被当成路径）
- 对5个现有类别无影响（所有实际焊缝组件顶点数 > 50）

### 3. `build_json_output_multi`

```python
def build_json_output_multi(model_name, paths_data):
    weld_paths = []
    for path in paths_data:
        segments_json = []
        for seg in path["fitted"]:
            pts_3d = back_project(np.array(seg["points_2d"]), path["plane"])
            segments_json.append({
                "type": seg["type"],
                "points": [[round(c, 6) for c in pt] for pt in pts_3d.tolist()],
                "fitting_error_mm": seg["fitting_error_mm"],
            })
        weld_paths.append({
            "closed": path["closed"],
            "segments": segments_json,
        })
    return {
        "model": model_name,
        "coord_system": "raw",
        "weld_paths": weld_paths,
    }
```

### 4. `visualize_multi`

**2D可视化的坐标系统一问题**：各path有独立PCA平面，直接绘制2D坐标会看起来位置错乱。

解决方案：
- 对所有组件合并顶点做**全局PCA**，得到 `global_plane`
- 各path的拟合结果（已知2D坐标+局部plane）先back_project到3D，再用 `global_plane` 重新投影到2D用于可视化
- 拟合计算仍在各path的**局部plane**完成（保持精度）

颜色方案（path索引循环）：
- Path 0: line=蓝, arc=红
- Path 1: line=紫, arc=橙
- Path 2: line=绿, arc=粉
- Path 3+: 从 matplotlib tab10 颜色循环取

3D面板配色同2D，原始mesh仍用半透明灰显示。

### 5. Force-close 行为

- `--force-close` 对所有**未自然闭合的path**生效
- 当前5个类别：只有 `cover_plate` 需要，`square_tube` 应被 `detect_closed` 自动识别（若未识别为已知限制，不在此spec解决）
- `_make_closing_segment` 函数不变

### 6. `print_summary` 更新

```
Weld seam analysis: bellmouth
  Paths: 2
  Path 0: Planarity 100.0% | 5 centerline pts | 1 segment (1 line, 0 arc) | closed: False | max err: 0.15 mm
  Path 1: Planarity 100.0% | 6 centerline pts | 1 segment (1 line, 0 arc) | closed: False | max err: 0.12 mm
  Total max fitting error: 0.15 mm
```

## 兼容性

- 旧JSON（`工件1_weld_seams.json` 等）**不保留向后兼容**，用新schema重新生成
- 现有单函数测试（pca_project、fit_segment等）不变
- 删除旧 `build_json_output`，保留 `_make_closing_segment`、`detect_closed` 等helper

## 测试

新增集成测试：
- `test_run_pipeline_multi_component`：合成一个含两根直线的OBJ文件，跑完整pipeline，断言 `weld_paths` 长度为2，每条path是1段line
- `test_component_filter`：含一个正常组件 + 一个5顶点的小碎片，断言只输出1条path

其他单元测试（pca_project、segment_by_curvature、fit_segment等）保持不变，因为这些函数仍然处理单组件数据。

## 技术决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 组件拆分接口 | `trimesh.Trimesh.split(only_watertight=False)` | 已有依赖，按face连通性拆分可靠 |
| 组件过滤阈值 | 10顶点 | 实际焊缝组件都>50，阈值10仅用于过滤噪声 |
| 各path独立PCA | 是 | 不同朝向焊缝共用平面会扭曲几何 |
| 2D可视化坐标系 | 全局PCA平面 | 避免多平面坐标错乱，仅视觉用途 |
| JSON schema | 嵌套（方案A） | 每path的closed/segments封装清晰 |
| Force-close粒度 | 全mesh级别 | 当前用例无需per-path控制 |
| 旧JSON兼容 | 不保留 | 只有少量测试数据，重跑即可 |

## CLI 接口

不变：

```bash
python scripts/extract_weld_seams.py \
  --workpiece assets/bellmouth/bellmouth.obj \
  --weld assets/bellmouth/bellmouth_weld.obj \
  [--output ...] [--no-viz] [--force-close]
```

批处理脚本 `scripts/extract_all_weld_seams.sh` 不变。

## 非目标

- 单个path内的多段arc精细拟合（square_tube圆形拟合精度问题另议）
- 跨mesh的焊缝对齐/合并（本spec只做拆分，不做合并）
- 路径顺序/起点的语义排序（输出顺序 = trimesh.split 的返回顺序，即face连通性决定）
