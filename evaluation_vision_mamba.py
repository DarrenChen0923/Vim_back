import torch
from models.vision_mamba_model import VisionMambaModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from utils.image_data_loader import get_image_data, normalize_image
from utils.cli import get_parser
from time import time
import os

# 解析命令行参数
parser = get_parser()
args = parser.parse_args()

# 参数配置
gsize = args.grid
d_model = args.d_model
image_size = args.image_size
patch_size = args.patch_size
data_mode = args.data_mode
model_path = args.load_model

print("=" * 80)
print("Vision Mamba 模型评估")
print("=" * 80)

# 验证模型路径
if not model_path:
    model_path = f'trained_models/vision_mamba/mamba_vision_{gsize}mm_d{d_model}_patch{patch_size}_final.pth'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}")

print(f"加载模型: {model_path}")
print(f"网格尺寸: {gsize}mm")
print(f"模型维度: {d_model}")
print(f"图像尺寸: {image_size}x{image_size}")
print(f"Patch尺寸: {patch_size}x{patch_size}")

# 加载数据
print(f"\n加载数据 (模式: {data_mode})...")
time0 = time()

x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = get_image_data(
    mode=data_mode,
    grid_size=gsize,
    image_size=image_size,
    heatmap_dir=args.heatmap_path if data_mode == 'from_heatmap' else None,
    train_ratio=0.8,
    seed=2
)

print(f"训练集: {x_train_tensor.shape}")
print(f"测试集: {x_test_tensor.shape}")

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型
print("\n创建模型...")
model = VisionMambaModel(
    img_size=image_size,
    patch_size=patch_size,
    in_channels=1,
    d_model=d_model,
    d_state=16,
    d_conv=4,
    expand=2,
    num_mamba_layers=args.num_mamba_layers,
    num_classes=1,
    use_attention=args.use_attention,
    num_attention_heads=4,
    num_attention_layers=6,
    use_bidirectional_mamba=True,
    dropout=0.1,
    pos_encoding_type='sinusoidal'
).to(device=device)

# 加载模型权重
print(f"加载模型权重...")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

total_params = model.get_num_params()
print(f"模型参数量: {total_params:,}")

# 归一化数据
x_test_normalized = normalize_image(x_test_tensor)
x_train_normalized = normalize_image(x_train_tensor)

# 在测试集上评估
print("\n" + "=" * 80)
print("在测试集上评估...")
print("=" * 80)

eval_start = time()

with torch.no_grad():
    pre_test = model(x_test_normalized)
    pre_test_numpy = pre_test.cpu().detach().numpy()
    y_test_numpy = y_test_tensor.cpu().detach().numpy()

eval_time = time() - eval_start

# 计算指标
test_mae = mean_absolute_error(y_test_numpy, pre_test_numpy)
test_mse = mean_squared_error(y_test_numpy, pre_test_numpy)
test_rmse = mean_squared_error(y_test_numpy, pre_test_numpy, squared=False)
test_r2 = r2_score(y_test_numpy, pre_test_numpy)

print("\n测试集性能:")
print(f"  MAE:  {test_mae:.6f}")
print(f"  MSE:  {test_mse:.6f}")
print(f"  RMSE: {test_rmse:.6f}")
print(f"  R²:   {test_r2:.6f}")
print(f"\n评估时间: {eval_time:.4f} 秒")
print(f"样本数量: {len(y_test_numpy)}")
print(f"平均推理时间: {eval_time / len(y_test_numpy) * 1000:.2f} ms/样本")

# 可选：在训练集上评估
if args.eval_train:
    print("\n" + "=" * 80)
    print("在训练集上评估...")
    print("=" * 80)
    
    with torch.no_grad():
        pre_train = model(x_train_normalized)
        pre_train_numpy = pre_train.cpu().detach().numpy()
        y_train_numpy = y_train_tensor.cpu().detach().numpy()
    
    train_mae = mean_absolute_error(y_train_numpy, pre_train_numpy)
    train_mse = mean_squared_error(y_train_numpy, pre_train_numpy)
    train_rmse = mean_squared_error(y_train_numpy, pre_train_numpy, squared=False)
    train_r2 = r2_score(y_train_numpy, pre_train_numpy)
    
    print("\n训练集性能:")
    print(f"  MAE:  {train_mae:.6f}")
    print(f"  MSE:  {train_mse:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  R²:   {train_r2:.6f}")

# 预测示例（显示前10个）
if args.show_predictions:
    print("\n" + "=" * 80)
    print("预测示例 (前10个测试样本):")
    print("=" * 80)
    print(f"{'Index':<8} {'True':<12} {'Predicted':<12} {'Error':<12}")
    print("-" * 50)
    
    for i in range(min(10, len(y_test_numpy))):
        true_val = y_test_numpy[i, 0]
        pred_val = pre_test_numpy[i, 0]
        error = abs(true_val - pred_val)
        print(f"{i:<8} {true_val:<12.6f} {pred_val:<12.6f} {error:<12.6f}")

# 总结
total_time = (time() - time0) / 60
print("\n" + "=" * 80)
print("评估完成!")
print(f"总耗时: {total_time:.2f} 分钟")
print("=" * 80)

# 保存结果到文件
if args.save_results:
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = f'{results_dir}/vision_mamba_{gsize}mm_d{d_model}_patch{patch_size}_results.txt'
    
    with open(results_file, 'w') as f:
        f.write("Vision Mamba 评估结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"网格尺寸: {gsize}mm\n")
        f.write(f"模型维度: {d_model}\n")
        f.write(f"图像尺寸: {image_size}x{image_size}\n")
        f.write(f"Patch尺寸: {patch_size}x{patch_size}\n")
        f.write(f"参数量: {total_params:,}\n")
        f.write("\n测试集性能:\n")
        f.write(f"  MAE:  {test_mae:.6f}\n")
        f.write(f"  MSE:  {test_mse:.6f}\n")
        f.write(f"  RMSE: {test_rmse:.6f}\n")
        f.write(f"  R²:   {test_r2:.6f}\n")
        f.write(f"\n评估时间: {eval_time:.4f} 秒\n")
        
        if args.eval_train:
            f.write("\n训练集性能:\n")
            f.write(f"  MAE:  {train_mae:.6f}\n")
            f.write(f"  MSE:  {train_mse:.6f}\n")
            f.write(f"  RMSE: {train_rmse:.6f}\n")
            f.write(f"  R²:   {train_r2:.6f}\n")
    
    print(f"\n结果已保存到: {results_file}")

