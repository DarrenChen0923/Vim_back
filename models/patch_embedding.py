import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    split 2D image into patches and embed into high-dimensional space
    
    Args:   
        img_size: input image size (default 224)
        patch_size: Patch size (default 16)
        in_channels: input channels (default 1, grayscale)
        embed_dim: embedding dimension (default 128)
        add_cls_token: whether to add class token (default True)
    """
    
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=1, 
        embed_dim=128,
        add_cls_token=True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.add_cls_token = add_cls_token
        
        # calculate patch number
        self.num_patches = (img_size // patch_size) ** 2
        
        # use convolution to implement patch embedding (more efficient)
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # optional class token
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - input image
        Returns:
            patches: (B, num_patches, embed_dim) or (B, num_patches+1, embed_dim)
        """
        B, C, H, W = x.shape
        
        # verify input size
        assert H == self.img_size and W == self.img_size, \
            f"输入图像尺寸({H}, {W})必须匹配img_size({self.img_size})"
        
        # patch embedding: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        
        # flatten patches: (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # add class token
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
        return x


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal position encoding
    add position information to image patches
    
    Args:
        embed_dim: embedding dimension (must be even)
        max_h: maximum height (patches number)
        max_w: maximum width (patches number)
        dropout: dropout rate (default 0.1)
    """
    
    def __init__(self, embed_dim, max_h=14, max_w=14, dropout=0.1):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        
        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w
        self.dropout = nn.Dropout(p=dropout)
        
        # precompute position encoding
        pe = self._make_position_encoding(max_h, max_w, embed_dim)
        self.register_buffer('pe', pe)
        
    def _make_position_encoding(self, h, w, d_model):
        """
        create 2D sinusoidal position encoding
        
        Args:
            h: height (patches number)
            w: width (patches number)  
            d_model: 嵌入维度
        Returns:
            pe: (1, h*w, d_model) - position encoding
        """
        # use half of the embedding dimension for each dimension
        d_model_half = d_model // 2
        
        # create position grid
        y_pos = torch.arange(h).unsqueeze(1).float()  # (h, 1)
        x_pos = torch.arange(w).unsqueeze(0).float()  # (1, w)
        
        # calculate frequency
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2).float() * 
            (-math.log(10000.0) / d_model_half)
        )
        
        # Y direction encoding
        pe_y = torch.zeros(h, w, d_model_half)
        pe_y[:, :, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, :, 1::2] = torch.cos(y_pos * div_term)
        
        # X direction encoding
        pe_x = torch.zeros(h, w, d_model_half)
        pe_x[:, :, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, :, 1::2] = torch.cos(x_pos * div_term)
        
        # concatenate X and Y encoding
        pe = torch.cat([pe_y, pe_x], dim=-1)  # (h, w, d_model)
        
        # flatten to sequence
        pe = pe.reshape(-1, d_model).unsqueeze(0)  # (1, h*w, d_model)
        
        return pe
        
    def forward(self, x):
        """
        Args:
            x: (B, N, D) - input patches, N may contain cls_token
        Returns:
            x_with_pe: (B, N, D) - patches with position encoding
        """
        B, N, D = x.shape
        
        # check if there is cls_token
        has_cls_token = (N == self.max_h * self.max_w + 1)
        
        if has_cls_token:
            # separate cls_token and patches
            cls_token = x[:, :1, :]
            patches = x[:, 1:, :]
            
            # add position encoding to patches
            patches = patches + self.pe[:, :patches.size(1), :]
            
            # concatenate back cls_token (cls_token does not need position encoding)
            x = torch.cat([cls_token, patches], dim=1)
        else:
            # add position encoding directly
            x = x + self.pe[:, :N, :]
            
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    learnable position encoding
    
    Args:
        num_patches: patch number
        embed_dim: embedding dimension
        add_cls_token: whether to include cls_token
    """
    
    def __init__(self, num_patches, embed_dim, add_cls_token=True):
        super().__init__()
        self.add_cls_token = add_cls_token
        
        # position encoding number (if there is cls_token, then +1)
        num_positions = num_patches + 1 if add_cls_token else num_patches
        
        # learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        
        # initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, D) - input patches
        Returns:
            x_with_pe: (B, N, D) - patches with position encoding
        """
        return x + self.pos_embed

