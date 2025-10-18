# Vision Mamba 使用指南

## 📋 概述

本文档说明如何使用Vision Mamba版本进行SPIF回弹误差预测。Vision Mamba将原有的1D序列模型改造为基于Patch的2D图像模型，通过热力图表示提升特征提取能力。

---

## 🏗️ 架构说明

### Vision Mamba vs 原始Mamba

| 特性 | 原始Mamba | Vision Mamba |
|------|-----------|--------------|
| 输入格式 | 1D序列 (B, L, 1) | 2D图像 (B, 1, H, W) |
| 数据表示 | 直接数值 | 热力图 |
| 特征提取 | Linear映射 | Patch Embedding |
| 位置编码 | 1D正弦编码 | 2D正弦编码 |
| 扫描策略 | 单向 | 双向（可选） |
| 空间建模 | 序列依赖 | 2D空间结构 |

### 模型架构

```
图像输入(1×224×224) 
    ↓
Patch Embedding (14×14 patches)
    ↓
2D位置编码
    ↓
可选的Attention层（保留原有优势）
    ↓
Vision Mamba Blocks ×6
    ├─ LayerNorm
    ├─ 双向Mamba扫描
    │   ├─ Forward Mamba
    │   ├─ Backward Mamba
    │   └─ 特征融合
    ├─ 残差连接
    └─ FFN
    ↓
全局池化（使用CLS token）
    ↓
分类/回归头
    ↓
输出预测值
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装新增的依赖
pip install albumentations>=1.3.0
pip install timm>=0.9.0
pip install einops>=0.7.0

# 或者直接安装全部依赖
pip install -r requirements.txt
```

### 2. 准备数据

#### 方式A：自动从文本生成热力图（推荐用于快速测试）

```bash
python train_vision_mamba.py \
    --project_root /your/project/path \
    --grid 15 \
    --data_mode from_text \
    --image_size 224 \
    --patch_size 16 \
    --d_model 128
```

#### 方式B：预先生成热力图（推荐用于大规模训练）

```bash
# 步骤1: 批量生成热力图
python generate_heatmaps.py \
    --project_root /your/project/path \
    --grid_size 15 \
    --image_size 224 \
    --output_dir heatmaps \
    --cmap viridis \
    --interpolation bilinear

# 步骤2: 使用预生成的热力图训练
python train_vision_mamba.py \
    --data_mode from_heatmap \
    --heatmap_path heatmaps/15mm \
    --grid 15 \
    --image_size 224 \
    --patch_size 16 \
    --d_model 128
```

### 3. 训练模型

#### 基础训练（仅Mamba）

```bash
python train_vision_mamba.py \
    --project_root /your/project/path \
    --grid 15 \
    --data_mode from_text \
    --image_size 224 \
    --patch_size 16 \
    --d_model 128 \
    --num_mamba_layers 6 \
    --learning_rate 0.00001 \
    --epochs 1500 \
    --batch_size 64
```

#### 混合架构训练（Attention + Mamba）

```bash
python train_vision_mamba.py \
    --project_root /your/project/path \
    --grid 15 \
    --data_mode from_text \
    --image_size 224 \
    --patch_size 16 \
    --d_model 128 \
    --num_mamba_layers 6 \
    --use_attention \
    --num_attention_heads 4 \
    --num_attention_layers 6 \
    --learning_rate 0.00001 \
    --epochs 1500
```

### 4. 评估模型

```bash
python evaluation_vision_mamba.py \
    --project_root /your/project/path \
    --grid 15 \
    --data_mode from_text \
    --image_size 224 \
    --patch_size 16 \
    --d_model 128 \
    --load_model trained_models/vision_mamba/mamba_vision_15mm_d128_patch16_final.pth \
    --eval_train \
    --show_predictions \
    --save_results
```

---

## 🎛️ 主要参数说明

### 图像参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image_size` | 224 | 输入图像尺寸（必须能被patch_size整除） |
| `--patch_size` | 16 | Patch大小（常用值：8, 16, 32） |
| `--in_channels` | 1 | 输入通道数（1=灰度，3=RGB） |

### 数据加载

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_mode` | from_text | 数据加载模式（from_text/from_heatmap/from_images） |
| `--heatmap_path` | '' | 预生成热力图目录 |
| `--heatmap_cmap` | viridis | 热力图colormap |
| `--heatmap_interpolation` | bilinear | 插值方法 |

### 模型架构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--d_model` | 64 | 模型嵌入维度 |
| `--num_mamba_layers` | 6 | Mamba块数量 |
| `--use_attention` | False | 是否使用Attention层 |
| `--num_attention_heads` | 4 | Attention头数 |
| `--num_attention_layers` | 6 | Attention层数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning_rate` | 0.00001 | 学习率 |
| `--epochs` | 1500 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--use_augmentation` | False | 是否使用数据增强 |

---

## 📊 性能对比

### 不同Patch尺寸对比

```bash
# Patch 8×8 (更多patches，更细粒度)
python train_vision_mamba.py --patch_size 8 --image_size 224

# Patch 16×16 (平衡)
python train_vision_mamba.py --patch_size 16 --image_size 224

# Patch 32×32 (更少patches，更高效)
python train_vision_mamba.py --patch_size 32 --image_size 224
```

### 不同图像尺寸对比

```bash
# 小图像（快速训练）
python train_vision_mamba.py --image_size 128 --patch_size 16

# 标准图像
python train_vision_mamba.py --image_size 224 --patch_size 16

# 大图像（高精度）
python train_vision_mamba.py --image_size 384 --patch_size 16
```

---

## 🔧 高级用法

### 1. 自定义热力图生成

```python
from utils.heatmap_generator import HeatmapGenerator

# 创建生成器
generator = HeatmapGenerator(
    grid_size=(3, 3),
    image_size=(224, 224),
    cmap='plasma',  # 尝试不同的colormap
    interpolation='bicubic',
    normalize=True
)

# 生成热力图
data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
heatmap = generator.generate(data)

# 可视化对比
generator.visualize_comparison(data, save_path='my_heatmap.png')
```

### 2. 使用数据增强

```bash
python train_vision_mamba.py \
    --use_augmentation \
    --data_mode from_text \
    --grid 15
```

### 3. 提取中间特征

```python
from models.vision_mamba_model import VisionMambaModel
import torch

# 加载模型
model = VisionMambaModel(...)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 提取特征
image = torch.randn(1, 1, 224, 224)
features = model.get_feature_maps(image)

# 访问不同层的特征
patch_features = features['patch_embed']
attention_features = features['attention']
mamba_features = features['mamba_block_0']
```

### 4. 可视化注意力权重

在训练时添加可视化：

```python
# 在models/vision_mamba_model.py中
# 设置vis=True查看attention权重
x = self.attention_blocks(x, vis=True)
```

---

## 📁 文件结构

```
Mamba-back/
├── models/
│   ├── patch_embedding.py          # Patch嵌入和位置编码
│   ├── vision_mamba_model.py       # Vision Mamba模型
│   ├── layers.py                   # Attention层（复用）
│   └── position_encoding.py        # 位置编码（复用）
├── utils/
│   ├── heatmap_generator.py        # 热力图生成器
│   ├── image_data_loader.py        # 图像数据加载
│   └── cli.py                      # 命令行参数（已扩展）
├── train_vision_mamba.py           # Vision Mamba训练脚本
├── evaluation_vision_mamba.py      # Vision Mamba评估脚本
├── generate_heatmaps.py            # 热力图批量生成工具
├── requirements.txt                # 依赖包（已更新）
└── VISION_MAMBA_README.md          # 本文档
```

---

## 🐛 常见问题

### Q1: 内存不足怎么办？

**解决方案**：
- 减小batch_size
- 减小image_size
- 增大patch_size
- 使用gradient checkpointing

```bash
python train_vision_mamba.py \
    --batch_size 32 \
    --image_size 128 \
    --patch_size 16
```

### Q2: 训练速度慢？

**解决方案**：
- 预先生成热力图（避免实时生成）
- 使用混合精度训练
- 减少Mamba层数
- 关闭Attention层

### Q3: 如何选择Patch尺寸？

**建议**：
- 小Patch (8×8): 更细粒度，参数多，慢但准确
- 中Patch (16×16): 平衡，推荐
- 大Patch (32×32): 粗粒度，快但可能丢失细节

### Q4: 是否使用Attention层？

**建议**：
- 数据量大：使用Attention，性能更好
- 数据量小：不用Attention，防止过拟合
- 计算资源足：使用，充分利用混合架构优势

---

## 📈 实验建议

### 消融实验

1. **Patch尺寸消融**
   ```bash
   for patch in 8 16 32; do
       python train_vision_mamba.py --patch_size $patch --grid 15
   done
   ```

2. **模型维度消融**
   ```bash
   for dim in 64 128 256; do
       python train_vision_mamba.py --d_model $dim --grid 15
   done
   ```

3. **Attention有效性**
   ```bash
   # 无Attention
   python train_vision_mamba.py --grid 15
   
   # 有Attention
   python train_vision_mamba.py --grid 15 --use_attention
   ```

4. **不同网格尺寸**
   ```bash
   for grid in 5 10 15 20; do
       python train_vision_mamba.py --grid $grid --d_model 128
   done
   ```

---

## 🎯 最佳实践

### 推荐配置（平衡性能与效率）

```bash
python train_vision_mamba.py \
    --project_root /your/path \
    --grid 15 \
    --data_mode from_text \
    --image_size 224 \
    --patch_size 16 \
    --d_model 128 \
    --num_mamba_layers 6 \
    --use_attention \
    --num_attention_heads 4 \
    --num_attention_layers 6 \
    --learning_rate 0.00001 \
    --epochs 1500 \
    --batch_size 64
```

### 快速验证配置

```bash
python train_vision_mamba.py \
    --grid 15 \
    --image_size 128 \
    --patch_size 16 \
    --d_model 64 \
    --num_mamba_layers 4 \
    --epochs 100 \
    --batch_size 128
```

### 高精度配置

```bash
python train_vision_mamba.py \
    --grid 5 \
    --image_size 384 \
    --patch_size 16 \
    --d_model 256 \
    --num_mamba_layers 8 \
    --use_attention \
    --epochs 2000 \
    --batch_size 32 \
    --learning_rate 0.000005
```

---

## 🔍 可视化与调试

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir runs/

# 访问 http://localhost:6006
```

### 查看训练日志

```bash
# 查看loss曲线
# 查看学习率变化
# 查看评估指标
```

### 生成热力图示例

```bash
python generate_heatmaps.py \
    --grid_size 15 \
    --output_dir heatmaps \
    --image_size 224

# 查看生成的示例对比图
open heatmaps/15mm/example_comparison.png
```

---

## 📞 技术支持

如有问题，请检查：
1. 数据路径是否正确
2. 图像尺寸能否被patch_size整除
3. GPU内存是否充足
4. 依赖包是否正确安装

---

## 🎉 结语

Vision Mamba为SPIF回弹预测提供了全新的视角，通过热力图表示和2D空间建模，有望获得更好的性能。

**接口已预留好**，您可以：
- 直接使用文本数据训练（自动生成热力图）
- 预先生成热力图加速训练
- 自定义图像输入

祝训练顺利！🚀

