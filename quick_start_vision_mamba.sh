#!/bin/bash
# Vision Mamba 快速开始脚本

echo "=========================================="
echo "Vision Mamba 快速开始"
echo "=========================================="

# 设置项目根目录（请修改为您的实际路径）
PROJECT_ROOT="/Users/darren/资料"
GRID_SIZE=15
IMAGE_SIZE=224
PATCH_SIZE=16
D_MODEL=128

echo ""
echo "配置参数:"
echo "  项目根目录: $PROJECT_ROOT"
echo "  网格尺寸: ${GRID_SIZE}mm"
echo "  图像尺寸: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Patch尺寸: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "  模型维度: $D_MODEL"
echo ""

# 选项1: 直接训练（自动生成热力图）
echo "=========================================="
echo "选项1: 直接训练（推荐用于快速测试）"
echo "=========================================="
echo ""
echo "运行命令:"
echo "python train_vision_mamba.py \\"
echo "    --project_root $PROJECT_ROOT \\"
echo "    --grid $GRID_SIZE \\"
echo "    --data_mode from_text \\"
echo "    --image_size $IMAGE_SIZE \\"
echo "    --patch_size $PATCH_SIZE \\"
echo "    --d_model $D_MODEL \\"
echo "    --num_mamba_layers 6 \\"
echo "    --use_attention \\"
echo "    --epochs 100 \\"
echo "    --batch_size 64"
echo ""

read -p "是否执行训练? (y/n): " choice
if [ "$choice" == "y" ]; then
    python train_vision_mamba.py \
        --project_root $PROJECT_ROOT \
        --grid $GRID_SIZE \
        --data_mode from_text \
        --image_size $IMAGE_SIZE \
        --patch_size $PATCH_SIZE \
        --d_model $D_MODEL \
        --num_mamba_layers 6 \
        --use_attention \
        --epochs 100 \
        --batch_size 64
fi

# 选项2: 预生成热力图 + 训练
echo ""
echo "=========================================="
echo "选项2: 预生成热力图（推荐用于大规模训练）"
echo "=========================================="
echo ""
echo "步骤1: 生成热力图"
echo "python generate_heatmaps.py \\"
echo "    --project_root $PROJECT_ROOT \\"
echo "    --grid_size $GRID_SIZE \\"
echo "    --image_size $IMAGE_SIZE \\"
echo "    --output_dir heatmaps"
echo ""

read -p "是否生成热力图? (y/n): " choice
if [ "$choice" == "y" ]; then
    python generate_heatmaps.py \
        --project_root $PROJECT_ROOT \
        --grid_size $GRID_SIZE \
        --image_size $IMAGE_SIZE \
        --output_dir heatmaps
    
    echo ""
    echo "步骤2: 使用预生成热力图训练"
    echo "python train_vision_mamba.py \\"
    echo "    --data_mode from_heatmap \\"
    echo "    --heatmap_path heatmaps/${GRID_SIZE}mm \\"
    echo "    --grid $GRID_SIZE \\"
    echo "    --image_size $IMAGE_SIZE \\"
    echo "    --patch_size $PATCH_SIZE \\"
    echo "    --d_model $D_MODEL \\"
    echo "    --epochs 1500"
    echo ""
    
    read -p "是否开始训练? (y/n): " choice
    if [ "$choice" == "y" ]; then
        python train_vision_mamba.py \
            --data_mode from_heatmap \
            --heatmap_path heatmaps/${GRID_SIZE}mm \
            --grid $GRID_SIZE \
            --image_size $IMAGE_SIZE \
            --patch_size $PATCH_SIZE \
            --d_model $D_MODEL \
            --epochs 1500
    fi
fi

# 评估模型
echo ""
echo "=========================================="
echo "评估训练好的模型"
echo "=========================================="
echo ""
MODEL_PATH="trained_models/vision_mamba/mamba_vision_${GRID_SIZE}mm_d${D_MODEL}_patch${PATCH_SIZE}_final.pth"
echo "模型路径: $MODEL_PATH"
echo ""
echo "python evaluation_vision_mamba.py \\"
echo "    --project_root $PROJECT_ROOT \\"
echo "    --grid $GRID_SIZE \\"
echo "    --data_mode from_text \\"
echo "    --image_size $IMAGE_SIZE \\"
echo "    --patch_size $PATCH_SIZE \\"
echo "    --d_model $D_MODEL \\"
echo "    --load_model $MODEL_PATH \\"
echo "    --eval_train \\"
echo "    --show_predictions \\"
echo "    --save_results"
echo ""

read -p "是否评估模型? (y/n): " choice
if [ "$choice" == "y" ]; then
    python evaluation_vision_mamba.py \
        --project_root $PROJECT_ROOT \
        --grid $GRID_SIZE \
        --data_mode from_text \
        --image_size $IMAGE_SIZE \
        --patch_size $PATCH_SIZE \
        --d_model $D_MODEL \
        --load_model $MODEL_PATH \
        --eval_train \
        --show_predictions \
        --save_results
fi

echo ""
echo "=========================================="
echo "完成!"
echo "=========================================="
echo ""
echo "查看TensorBoard日志:"
echo "  tensorboard --logdir runs/"
echo ""
echo "查看评估结果:"
echo "  cat evaluation_results/vision_mamba_${GRID_SIZE}mm_d${D_MODEL}_patch${PATCH_SIZE}_results.txt"
echo ""

