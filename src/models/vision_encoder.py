import torch
import torch.nn as nn 

class PatchEmbedding(nn.Module):
    """ converts image into patches (patch embeddings)
    
    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, image_size: int = 32, patch_size: int = 4, in_channels: int = 3 , embed_dim: int = 256):
        super().__init__()
        
        assert image_size % patch_size == 0
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = ( image_size // patch_size ) ** 2 
        
        self.proj = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embed_dim,
            kernel_size = patch_size,
            stride = patch_size
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images) ### [ B =  Batch size, C = Channels , H = Height, W = Width ] ---> ## [ B = Batch size, D = output dimensions , H/P =  No. of Vertical Patches, W/P = No. of Horizontal Patches]
        x = x.flatten(2) ###  Flatten's the x from the index 2 --> n =  H/P x N/P  [B, D , N]
        x = x.transpose(1,2) ### transposes the 1 , 2 indexes --> [B, N, D]
            
        return x
        

class TinyViTBlock(nn.Module):
    """
    Standard ViT encoder block:
    
    layerNorm --> Self.Attention --> residual 
    layerNorm --> MLP --> residual

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 4,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1
                 ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        
        self.attn = nn.MultiheadAttention( 
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = embed_dim * mlp_ratio
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        
        attn_output, _ = self.attn(
            query=attn_input, 
            key=attn_input,
            value=attn_input,
            need_weights=False
        )
        
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        
        return x

class TinyViEncoder(nn.Module):
    """ This the Vision Encoder BlocK

    Args:
        nn (_type_): _description_
    """

        
    def __init__(self, 
                image_size: int = 32 , 
                patch_size: int = 4 , 
                in_channels = 3 , 
                embed_dim: int = 256,
                num_layers: int = 6, 
                num_heads: int = 4, 
                mlp_ratio: int = 4, 
                dropout: float = 0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding( image_size=image_size, 
                        patch_size=patch_size,
                        in_channels=in_channels, 
                        embed_dim=embed_dim)
        
        self.num_patches = self.patch_embed.num_patches
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        x = x + self.position_embedding
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        
        return x

if __name__ == "__main__":
    model = TinyViEncoder( 
                        image_size = 32 , 
                        patch_size = 4 , 
                        embed_dim = 512,
                        num_layers = 8, 
                        num_heads = 8, 
                        )
    
    dummy_images = torch.randn(2, 3, 32, 32)
    visual_tokens = model(dummy_images)
    
    print("Input images shape:", dummy_images.shape)
    print("Output Visual Tokens:", visual_tokens.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ",total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Trainable parameters: ",trainable_params)