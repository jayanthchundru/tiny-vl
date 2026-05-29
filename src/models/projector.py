import torch
import torch.nn as nn 


class VisualProjector(nn.Module):
    def __init__(self, vision_dim: int = 512, 
                 language_dim: int = 512, 
                 hidden_dim: int = 2048, 
                 dropout: float=0.1
                 ):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, language_dim)
        )
    
    def forward(self, visual_tokens: int = 512):
        projected_tokens = self.projector(visual_tokens)
        
        return projected_tokens

if __name__ == "__main__":
    projector = VisualProjector(vision_dim=512, language_dim=512, hidden_dim=2048)
    
    dummy_visual_tokens = torch.randn(2, 64, 512)
    projected_tokens = projector(dummy_visual_tokens)
    
    print(projected_tokens.shape)