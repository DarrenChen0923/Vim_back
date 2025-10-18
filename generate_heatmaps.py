#!/usr/bin/env python3
"""
热力图批量生成工具
用于预先将文本数据转换为热力图，加速训练过程
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from utils.heatmap_generator import HeatmapGenerator
import random


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量生成热力图')
    
    parser.add_argument(
        '--project_root',
        type=str,
        default='/Users/darren/资料',
        help='项目根目录'
    )
    
    parser.add_argument(
        '--grid_size',
        type=int,
        default=15,
        choices=[5, 10, 15, 20],
        help='网格尺寸（mm）'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='输出图像尺寸'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='heatmaps',
        help='输出目录'
    )
    
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help='Colormap名称'
    )
    
    parser.add_argument(
        '--interpolation',
        type=str,
        default='bilinear',
        choices=['nearest', 'bilinear', 'bicubic'],
        help='插值方法'
    )
    
    parser.add_argument(
        '--save_rgb',
        action='store_true',
        help='保存RGB彩色图（默认保存灰度图）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=2,
        help='随机种子'
    )
    
    return parser.parse_args()


def load_text_data(project_root, grid_size, seed=2):
    """
    从文本文件加载数据
    
    Args:
        project_root: 项目根目录
        grid_size: 网格尺寸
        seed: 随机种子
    
    Returns:
        data_list: 数据列表
        labels: 标签列表
    """
    random.seed(seed)
    
    filenums = [1, 2, 3]
    data_list = []
    labels = []
    
    print(f"从文本文件加载数据 (网格尺寸: {grid_size}mm)...")
    
    for filenum in filenums:
        file_path = f'{project_root}/Mamba-back/data/{grid_size}mm_file/outfile{filenum}/trainingfile_{grid_size}mm_overlapping_3.txt'
        
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 {file_path}")
            continue
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            
            for line in lines:
                line = line.strip("\n")
                x = line.split("|")[0].split(",")
                y = line.split("|")[1]
                
                # NaN处理
                x = [0.0 if value == 'NaN' else float(value) for value in x]
                y = 0.0 if y == 'NaN' else float(y)
                
                data_list.append(x)
                labels.append(y)
    
    print(f"加载完成! 总样本数: {len(data_list)}")
    return np.array(data_list, dtype=np.float32), np.array(labels, dtype=np.float32)


def generate_heatmaps(args):
    """批量生成热力图"""
    
    # 加载数据
    data_list, labels = load_text_data(args.project_root, args.grid_size, args.seed)
    
    # 创建热力图生成器
    grid_h = grid_w = 3  # 3x3网格
    
    generator = HeatmapGenerator(
        grid_size=(grid_h, grid_w),
        image_size=(args.image_size, args.image_size),
        cmap=args.cmap,
        interpolation=args.interpolation,
        normalize=True
    )
    
    # 创建输出目录
    output_dir = f'{args.output_dir}/{args.grid_size}mm'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n生成热力图...")
    print(f"输出目录: {output_dir}")
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    print(f"Colormap: {args.cmap}")
    print(f"插值方法: {args.interpolation}")
    print(f"格式: {'RGB' if args.save_rgb else 'Grayscale'}")
    
    # 批量生成并保存
    for i, (data, label) in enumerate(tqdm(zip(data_list, labels), total=len(data_list))):
        # 生成热力图
        heatmap = generator.generate(data, return_rgb=args.save_rgb)
        
        # 文件名格式: {label}_{index}.png
        filename = f'{label:.6f}_{i:05d}.png'
        filepath = os.path.join(output_dir, filename)
        
        # 保存
        generator.save(heatmap, filepath, show_colorbar=False)
    
    print(f"\n完成! 已生成 {len(data_list)} 张热力图")
    print(f"保存位置: {output_dir}")
    
    # 生成元数据文件
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Grid Size: {args.grid_size}mm\n")
        f.write(f"Image Size: {args.image_size}x{args.image_size}\n")
        f.write(f"Colormap: {args.cmap}\n")
        f.write(f"Interpolation: {args.interpolation}\n")
        f.write(f"Total Samples: {len(data_list)}\n")
        f.write(f"Format: {'RGB' if args.save_rgb else 'Grayscale'}\n")
        f.write(f"Random Seed: {args.seed}\n")
    
    print(f"元数据已保存: {metadata_path}")
    
    # 生成示例对比图
    if len(data_list) > 0:
        print("\n生成示例对比图...")
        sample_idx = 0
        comparison_path = os.path.join(output_dir, 'example_comparison.png')
        generator.visualize_comparison(data_list[sample_idx], save_path=comparison_path)
        print(f"示例对比图已保存: {comparison_path}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("热力图批量生成工具")
    print("=" * 80)
    
    # 生成热力图
    generate_heatmaps(args)
    
    print("\n" + "=" * 80)
    print("全部完成!")
    print("=" * 80)
    print("\n使用生成的热力图进行训练:")
    print(f"  python train_vision_mamba.py \\")
    print(f"    --data_mode from_heatmap \\")
    print(f"    --heatmap_path {args.output_dir}/{args.grid_size}mm \\")
    print(f"    --grid {args.grid_size} \\")
    print(f"    --image_size {args.image_size} \\")
    print(f"    --d_model 128")


if __name__ == '__main__':
    main()

