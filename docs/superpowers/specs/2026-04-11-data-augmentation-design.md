# AIWS5.2 数据集离线数据增强设计

## 背景

`data/aiws5.2-dataset-v1` 是类别均衡后的核工件实例分割数据集（train 292 张，val 104 张，共 396 张），4 类各约 25%。当前数据量偏小，需要通过离线数据增强将规模扩展到 3-4 倍（约 1200-1600 张）。

增强数据仅用于**实例分割和类别预测**训练，不涉及 6D 位姿估计，因此增强策略不受几何一致性约束。

## 增强策略

每张原图生成 **3 张**增强图（原始 396 + 增强 1188 = 共约 1584 张，~4 倍）。

### 几何变换（随机组合）

| 变换 | 参数 | 概率 |
|------|------|------|
| 水平翻转 | — | 0.5 |
| 随机旋转 | ±15° | 0.4 |
| 随机缩放裁剪 | scale 0.8x-1.2x，保持 16:9 比例 | 0.4 |
| 仿射变换 | 平移 ±5%，剪切 ±5° | 0.3 |

几何变换后的填充区域为黑色（`border_mode=CONSTANT, value=0`），掩码同步变换。

### 颜色/光照变换（随机组合）

| 变换 | 参数 | 概率 |
|------|------|------|
| 亮度/对比度 | ±0.2 | 0.5 |
| 色调/饱和度/明度 | H±10, S±20, V±20 | 0.4 |
| 高斯噪声 | std 0.02-0.05 | 0.3 |
| 高斯模糊 | kernel 3-7 | 0.2 |

颜色变换不影响掩码。

### 安全约束

- 变换后掩码面积 < 原始面积 10% 时，丢弃重新生成
- 每张图每个增强槽位最多重试 5 次，仍失败则跳过并记录日志

## 数据流与文件结构

### 输入

- `data/aiws5.2-dataset-v1/images/` — 原始 RGB（1920×1080）
- `data/aiws5.2-dataset-v1/annotations/train.json` / `val.json` — COCO 格式标注

### 输出

```
data/aiws5.2-dataset-v1-aug/
├── annotations/
│   ├── train.json          # 原始 + 增强标注合并
│   └── val.json            # 原始 + 增强标注合并
├── images/
│   ├── G90-v2-xxx.png      # 原始图片（复制）
│   ├── AUG_0000.png        # 增强图片
│   ├── AUG_0001.png
│   └── ...
```

### 命名与 ID 规则

- 增强图片文件名：`AUG_0000.png`、`AUG_0001.png` ... train 和 val 各自从 0000 开始递增
- `image_id`：原始图片保持原 ID，增强图片 ID 从 100000 开始递增
- `annotation id`：同理从 100000 开始递增
- 每条增强标注额外添加 `source_image_id` 字段，记录来源原图 ID

### 处理流程

1. 读取原始 COCO JSON
2. 遍历每张图片，`pycocotools.coco.annToMask()` 将多边形掩码转为二值位图
3. Albumentations Compose 同时对 image + mask 做变换，生成 3 次
4. `cv2.findContours()` 将增强后位图掩码转回 COCO 多边形格式
5. 过滤碎片轮廓（面积 < 50 像素），取最大连通区域
6. 重算 bbox 和 area
7. 安全检查（面积 < 原始 10% 则丢弃重试）
8. 保存增强图片，写入标注
9. 原始数据 + 增强数据合并输出最终 COCO JSON

## 实现细节

### Albumentations Compose

```python
transform = A.Compose([
    # 几何变换
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.4),
    A.RandomResizedCrop(size=(1080, 1920), scale=(0.8, 1.2), ratio=(16/9, 16/9), p=0.4),
    A.Affine(translate_percent=(-0.05, 0.05), shear=(-5, 5),
             border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3),
    # 颜色变换
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
    A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
])
```

### 掩码格式转换

- 多边形 -> 位图：`pycocotools.coco.annToMask()`
- 位图 -> 多边形：`cv2.findContours()`，过滤面积 < 50 的碎片，取最大连通区域

### 脚本接口

```bash
python scripts/augment_dataset.py \
    --input_dir data/aiws5.2-dataset-v1 \
    --output_dir data/aiws5.2-dataset-v1-aug \
    --num_aug 3 \
    --seed 42
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 输入数据集路径 | 必填 |
| `--output_dir` | 输出数据集路径 | 必填 |
| `--num_aug` | 每张原图生成增强图数量 | 3 |
| `--seed` | 随机种子，保证可复现 | 42 |

### 依赖

新增 `albumentations` 库，其余使用项目已有依赖（`opencv-python`、`pycocotools`、`numpy`）。

## 预期产出

| 指标 | Train | Val | 合计 |
|------|-------|-----|------|
| 原始 | 292 | 104 | 396 |
| 增强 | ~876 | ~312 | ~1188 |
| 总计 | ~1168 | ~416 | ~1584 |

各类别保持约 25% 均匀分布（每类等量增强）。
