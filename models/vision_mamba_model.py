import torch
import torch.nn as nn
from mamba_ssm import Mamba
from .patch_embedding import PatchEmbedding, PositionalEncoding2D, LearnablePositionalEncoding
from .layers import FFWRelativeSelfAttentionModule


class VisionMambaBlock(nn.Module):
    """
    Vision Mamba块，实现双向Mamba扫描
    
    Args:
        d_model: 模型维度
        d_state: SSM状态维度
        d_conv: 卷积核大小
        expand: 扩展因子
        dropout: Dropout率
        use_bidirectional: 是否使用双向扫描
    """
    
    def __init__(
        self, 
        d_model, 
        d_state=16, 
        d_conv=4, 
        expand=2,
        dropout=0.1,
        use_bidirectional=True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_bidirectional = use_bidirectional
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        
        # Forward Mamba
        self.mamba_forward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Backward Mamba (如果使用双向)
        if use_bidirectional:
            self.mamba_backward = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            # 融合层：将双向结果合并
            self.merge = nn.Linear(d_model * 2, d_model)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, D) - 输入序列
        Returns:
            output: (B, N, D) - 输出序列
        """
        # 保存残差
        residual = x
        
        # Layer Norm
        x = self.norm1(x)
        
        if self.use_bidirectional:
            # 前向扫描
            x_forward = self.mamba_forward(x)
            
            # 后向扫描（翻转序列）
            x_flipped = torch.flip(x, dims=[1])
            x_backward = self.mamba_backward(x_flipped)
            x_backward = torch.flip(x_backward, dims=[1])
            
            # 合并双向结果
            x = torch.cat([x_forward, x_backward], dim=-1)
            x = self.merge(x)
        else:
            # 仅前向扫描
            x = self.mamba_forward(x)
        
        # 残差连接
        x = residual + self.dropout(x)
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class VisionMambaModel(nn.Module):
    """
    完整的Vision Mamba模型
    
    架构: Patch Embedding → Position Encoding → [Attention] → Mamba Blocks → Head
    
    Args:
        img_size: 输入图像尺寸
        patch_size: Patch大小
        in_channels: 输入通道数
        d_model: 模型维度
        d_state: SSM状态维度
        d_conv: 卷积核大小
        expand: 扩展因子
        num_mamba_layers: Mamba块数量
        num_classes: 输出类别数（回归任务为1）
        use_attention: 是否使用Attention层
        num_attention_heads: Attention头数
        num_attention_layers: Attention层数
        use_bidirectional_mamba: 是否使用双向Mamba
        dropout: Dropout率
        pos_encoding_type: 位置编码类型 ('sinusoidal' 或 'learnable')
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=1,
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        num_mamba_layers=6,
        num_classes=1,
        use_attention=True,
        num_attention_heads=4,
        num_attention_layers=6,
        use_bidirectional_mamba=True,
        dropout=0.1,
        pos_encoding_type='sinusoidal'
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # 计算patch数量
        num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=d_model,
            add_cls_token=True  # 添加分类token
        )
        
        # 位置编码
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoding = PositionalEncoding2D(
                embed_dim=d_model,
                max_h=img_size // patch_size,
                max_w=img_size // patch_size,
                dropout=dropout
            )
        elif pos_encoding_type == 'learnable':
            self.pos_encoding = LearnablePositionalEncoding(
                num_patches=num_patches,
                embed_dim=d_model,
                add_cls_token=True
            )
        else:
            raise ValueError(f"不支持的位置编码类型: {pos_encoding_type}")
        
        # 可选的Attention层（保留原有架构优势）
        if use_attention:
            self.attention_blocks = FFWRelativeSelfAttentionModule(
                embedding_dim=d_model,
                num_attn_heads=num_attention_heads,
                num_layers=num_attention_layers,
                use_adaln=False  # Vision任务通常不需要AdaLN
            )
        
        # Vision Mamba Blocks
        self.mamba_blocks = nn.ModuleList([
            VisionMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                use_bidirectional=use_bidirectional_mamba
            )
            for _ in range(num_mamba_layers)
        ])
        
        # 最终的Layer Norm
        self.norm = nn.LayerNorm(d_model)
        
        # 分类/回归头
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - 输入图像
        Returns:
            output: (B, num_classes) - 预测输出
        """
        # Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches+1, d_model)
        
        # 添加位置编码
        x = self.pos_encoding(x)  # (B, num_patches+1, d_model)
        
        # 可选的Attention层
        if self.use_attention:
            # attention_blocks需要(L, B, D)格式
            x = x.transpose(0, 1)  # (num_patches+1, B, d_model)
            x = self.attention_blocks(
                x, 
                diff_ts=None,
                query_pos=None, 
                context=None, 
                context_pos=None,
                pad_mask=None
            )[-1]  # 取最后一层输出
            x = x.transpose(0, 1)  # (B, num_patches+1, d_model)
        
        # Vision Mamba Blocks
        for block in self.mamba_blocks:
            x = block(x)
        
        # 全局池化：使用cls_token或平均池化
        # 方式1: 使用cls_token（推荐）
        x = x[:, 0]  # (B, d_model)
        
        # 方式2: 平均池化（备选）
        # x = x.mean(dim=1)  # (B, d_model)
        
        # Layer Norm
        x = self.norm(x)
        
        # 分类/回归头
        output = self.head(x)  # (B, num_classes)
        
        return output
    
    def get_num_params(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_maps(self, x):
        """
        获取中间特征图（用于可视化）
        
        Args:
            x: (B, C, H, W) - 输入图像
        Returns:
            features: dict - 各层特征图
        """
        features = {}
        
        # Patch Embedding
        x = self.patch_embed(x)
        features['patch_embed'] = x
        
        # Position Encoding
        x = self.pos_encoding(x)
        features['pos_encoding'] = x
        
        # Attention
        if self.use_attention:
            x = x.transpose(0, 1)
            x = self.attention_blocks(
                x, diff_ts=None, query_pos=None, 
                context=None, context_pos=None, pad_mask=None
            )[-1]
            x = x.transpose(0, 1)
            features['attention'] = x
        
        # Mamba Blocks
        for i, block in enumerate(self.mamba_blocks):
            x = block(x)
            features[f'mamba_block_{i}'] = x
        
        return features

