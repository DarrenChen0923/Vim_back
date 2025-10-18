import os
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch.utils.data as Data
from models.vision_mamba_model import VisionMambaModel
from torch.utils.tensorboard import SummaryWriter
from utils.cli import get_parser
from fvcore.nn import FlopCountAnalysis


parser = get_parser()
args = parser.parse_args()

grids = [args.grid]

# 解析命令行参数
parser = get_parser()
args = parser.parse_args()

# 参数配置
gsize = args.grid
d_model = args.d_model
image_size = args.image_size
patch_size = args.patch_size
data_mode = args.data_mode

# 获取图像数据
print(f"load data mode: {data_mode}")
print(f"grid size: {gsize}mm, image size: {image_size}x{image_size}, patch size: {patch_size}x{patch_size}")

# create data loader [du]

def read_data(train_or_test):
        # set Image file and label file path
    image_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/{train_or_test}_dataset/{grids[0]}mm/images'
    label_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/{train_or_test}_dataset/{grids[0]}mm/labels'
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

    # Create empty list to store iamges and lables
    X = []  # store images
    y = []  # store lables

    # Iterate Images
    for image_path in image_files:

        # Get images file name
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Build path for label file
        label_path = os.path.join(label_folder, f'{image_filename}.txt')

        # Read label file
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip() 

        # Image preprocessing
        with Image.open(image_path) as img:
            img = np.array(img) 

        img = img.transpose((2, 0, 1))

        # put image and label into list
        X.append(img)
        y.append(label)

    return X,y

for fum in fums:
    for grid in grids:
        for degree in degrees:
            X_train,y_train = read_data("train")
            X_test,y_test = read_data("test")
           


# transfer X and y into numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)



# Normalise
X_train = X_train / 255.0
X_test = X_test/255.0

def normalize_mean_std(image):
    mean = np.mean(image)
    stddev = np.std(image)
    normalized_image = (image - mean) / stddev
    return normalized_image

X_train = normalize_mean_std(X_train)
X_test = normalize_mean_std(X_test)
batch = 64



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# #tensor to numpy
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

#  #gpu environment: transfer into cuda
if torch.cuda.is_available():
    X_train_tensor = X_train_tensor.cuda()
    X_test_tensor = X_test_tensor.cuda()
    y_train_tensor = y_train_tensor.cuda()
    y_test_tensor = y_test_tensor.cuda()

# #Cominbe dataset
train_dataset = Data.TensorDataset(X_train_tensor,y_train_tensor)
val_dataset = Data.TensorDataset(X_test_tensor,y_test_tensor)

# #Create dataset loader

train_data_loader = Data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_data_loader = Data.DataLoader(val_dataset, batch_size=batch, shuffle=False)





# 训练参数
batch = 64
lr = args.learning_rate
epo = args.epochs

# TensorBoard
writer = SummaryWriter('runs/vision_mamba_experiment')

# 创建数据加载器
loader = Data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"use device: {device}")

# 创建Vision Mamba模型
model = VisionMambaModel(
    img_size=image_size,
    patch_size=patch_size,
    in_channels=1,  # heatmap single channel
    d_model=d_model,
    d_state=16,
    d_conv=4,
    expand=2,
    num_mamba_layers=args.num_mamba_layers,
    num_classes=1,  # regression task
    use_attention=args.use_attention,
    num_attention_heads=4,
    num_attention_layers=6,
    use_bidirectional_mamba=True,
    dropout=0.1,
    pos_encoding_type='sinusoidal'
).to(device=device)

# 打印模型信息
total_params = model.get_num_params()
print(f"Total parameters: {total_params}")
print(f"Model architecture: Vision Mamba")
print(f"  - Image size: {image_size}x{image_size}")
print(f"  - Patch size: {patch_size}x{patch_size}")
print(f"  - Num patches: {(image_size // patch_size) ** 2}")
print(f"  - Embed dim: {d_model}")
print(f"  - Mamba layers: {args.num_mamba_layers}")
print(f"  - Use attention: {args.use_attention}")
print(f"  - Bidirectional Mamba: True")

# 损失函数和优化器
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 学习率调度器（可选）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epo, eta_min=lr/10)

# 记录开始时间
time0 = time()

# 归一化测试集
y_test_numpy = y_test_tensor.cpu().detach().numpy()

# 训练循环
update_step = 0
best_test_loss = float('inf')

print("\n开始训练...")
print("=" * 80)

with tqdm(range(epo), unit="epoch") as tepochs:
    for e in tepochs:
        model.train()
        total_loss = 0
        
        with tqdm(loader, unit="batch", leave=False) as tepoch:
            tepoch.set_description(f"Epoch {e+1}/{epo}")
            
            for step, (batch_x, batch_y) in enumerate(tepoch):
                # batch_x shape: (B, 1, H, W) - 已经是图像格式
                
    
                
                # 前向传播
                optimizer.zero_grad()
                output = model(batch_x)
                
                # 计算损失
                loss = criterion(output, batch_y)
                total_loss += loss.item()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 记录
                writer.add_scalar('loss/train', loss.item(), update_step)
                update_step += 1
                
                tepoch.set_postfix(loss=loss.item())
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('learning_rate', current_lr, e)
        
        # 每个epoch后评估
        model.eval()
        with torch.no_grad():
            pre = model(x_test_normalized)
            pre_numpy = pre.cpu().detach().numpy()
            mse = mean_squared_error(y_test_numpy, pre_numpy)
            mae = mean_absolute_error(y_test_numpy, pre_numpy)
            
            writer.add_scalar('loss/eval_mse', mse, update_step)
            writer.add_scalar('loss/eval_mae', mae, update_step)
            
            # 保存最佳模型
            if mse < best_test_loss:
                best_test_loss = mse
                best_model_path = f'trained_models/vision_mamba/best_model_{gsize}mm_d{d_model}_patch{patch_size}.pth'
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
        
        # 更新进度条
        avg_train_loss = total_loss / len(loader)
        tepochs.set_postfix(
            train_loss=avg_train_loss, 
            test_mse=mse,
            test_mae=mae,
            lr=current_lr
        )

print("\n" + "=" * 80)
print("训练完成!")

# 保存最终模型
final_model_dir = f'trained_models/vision_mamba'
os.makedirs(final_model_dir, exist_ok=True)
final_model_path = f'{final_model_dir}/mamba_vision_{gsize}mm_d{d_model}_patch{patch_size}_final.pth'
torch.save(model.state_dict(), final_model_path)
print(f"最终模型已保存: {final_model_path}")

# 计算FLOPs
model.eval()
dummy_input = torch.randn(1, 1, image_size, image_size).to(device)
flops = FlopCountAnalysis(model, dummy_input)
print(f"FLOPs: {flops.total():,}")

# 在训练集上评估
print("\n在训练集上评估...")
model.eval()
with torch.no_grad():
    pre_train = model(x_train_normalized)
    pre_train_numpy = pre_train.cpu().detach().numpy()
    y_train_numpy = y_train_tensor.cpu().detach().numpy()

    train_mae = mean_absolute_error(y_train_numpy, pre_train_numpy)
    train_mse = mean_squared_error(y_train_numpy, pre_train_numpy)
    train_rmse = mean_squared_error(y_train_numpy, pre_train_numpy, squared=False)
    train_r2 = r2_score(y_train_numpy, pre_train_numpy)

print("\n" + "=" * 80)
print("训练集性能:")
print(f"  MAE:  {train_mae:.6f}")
print(f"  MSE:  {train_mse:.6f}")
print(f"  RMSE: {train_rmse:.6f}")
print(f"  R²:   {train_r2:.6f}")

# 在测试集上评估
print("\n在测试集上评估...")
with torch.no_grad():
    pre_test = model(x_test_normalized)
    pre_test_numpy = pre_test.cpu().detach().numpy()
    
    test_mae = mean_absolute_error(y_test_numpy, pre_test_numpy)
    test_mse = mean_squared_error(y_test_numpy, pre_test_numpy)
    test_rmse = mean_squared_error(y_test_numpy, pre_test_numpy, squared=False)
    test_r2 = r2_score(y_test_numpy, pre_test_numpy)

print("\n测试集性能:")
print(f"  MAE:  {test_mae:.6f}")
print(f"  MSE:  {test_mse:.6f}")
print(f"  RMSE: {test_rmse:.6f}")
print(f"  R²:   {test_r2:.6f}")

# 总训练时间
total_time = (time() - time0) / 60
print(f"\n总训练时间: {total_time:.2f} 分钟")

# 关闭TensorBoard
writer.close()

print("\n" + "=" * 80)
print("训练完成! 模型和日志已保存。")
print(f"使用以下命令查看TensorBoard: tensorboard --logdir runs/")
print("=" * 80)

