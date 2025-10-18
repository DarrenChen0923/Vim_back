import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # === 原有参数 ===
    parser.add_argument(
        "--project_root",
        default="/Users/darren/资料",
        type=str,
        help="The project root to folder SPIF_DU, e.g. /example/works/du"
    )

    parser.add_argument(
        "--grid",
        default=15,
        type=int,
        help="Training with which grid size. Availiable options: (5,10,15,20)"
    )

    parser.add_argument(
        "--load_model",
        default="",
        type=str,
        help="Name of the model to be evaluated in trained_models folder."
    )

    parser.add_argument(
        "--d_model",
        default=64,
        type=int,
        help="Mamba Model dimension d_model"
    )
    
    # === Vision Mamba 图像参数 ===
    parser.add_argument(
        "--image_size",
        default=224,
        type=int,
        help="Input image size for Vision Mamba (default: 224)"
    )
    
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="Patch size for Vision Mamba (default: 16)"
    )
    
    parser.add_argument(
        "--in_channels",
        default=1,
        type=int,
        help="Number of input channels (1=grayscale, 3=RGB)"
    )
    
    # === 数据加载模式 ===
    parser.add_argument(
        "--data_mode",
        default='from_text',
        type=str,
        choices=['from_text', 'from_heatmap', 'from_images'],
        help="Data loading mode: from_text (generate heatmaps), from_heatmap (load pre-generated), from_images (custom images)"
    )
    
    parser.add_argument(
        "--heatmap_path",
        default='',
        type=str,
        help="Path to pre-generated heatmap directory (for data_mode='from_heatmap')"
    )
    
    # === Vision Mamba 架构参数 ===
    parser.add_argument(
        "--num_mamba_layers",
        default=6,
        type=int,
        help="Number of Vision Mamba blocks (default: 6)"
    )
    
    parser.add_argument(
        "--use_attention",
        action='store_true',
        help="Use attention layers before Mamba blocks (combines strengths of both)"
    )
    
    parser.add_argument(
        "--num_attention_heads",
        default=4,
        type=int,
        help="Number of attention heads if use_attention is True (default: 4)"
    )
    
    parser.add_argument(
        "--num_attention_layers",
        default=6,
        type=int,
        help="Number of attention layers if use_attention is True (default: 6)"
    )
    
    # === 训练参数 ===
    parser.add_argument(
        "--learning_rate",
        default=0.00001,
        type=float,
        help="Learning rate for optimizer (default: 1e-5)"
    )
    
    parser.add_argument(
        "--epochs",
        default=1500,
        type=int,
        help="Number of training epochs (default: 1500)"
    )
    
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Training batch size (default: 64)"
    )
    
    # === 数据增强参数 ===
    parser.add_argument(
        "--use_augmentation",
        action='store_true',
        help="Use data augmentation during training"
    )
    
    # === 评估参数 ===
    parser.add_argument(
        "--eval_train",
        action='store_true',
        help="Evaluate on training set as well"
    )
    
    parser.add_argument(
        "--show_predictions",
        action='store_true',
        help="Show prediction examples during evaluation"
    )
    
    parser.add_argument(
        "--save_results",
        action='store_true',
        help="Save evaluation results to file"
    )
    
    # === 热力图生成参数 ===
    parser.add_argument(
        "--heatmap_cmap",
        default='viridis',
        type=str,
        help="Colormap for heatmap generation (default: viridis)"
    )
    
    parser.add_argument(
        "--heatmap_interpolation",
        default='bilinear',
        type=str,
        choices=['nearest', 'bilinear', 'bicubic'],
        help="Interpolation method for heatmap generation (default: bilinear)"
    )
    
    return parser