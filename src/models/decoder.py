import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 512 , num_heads: int = 8, dropout: float = 0.1): 
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, x: torch.Tensor ) -> torch.Tensor:
        B, T, D = x.shape 
        
        qkv = self.qkv_proj(x)
        
        q, k , v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = q @ k.transpose(-2, -1)
        
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        
        mask = torch.triu( torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        attn_probs = self.dropout(attn_probs)
        
        out = attn_probs @ v
        out = out.transpose(1, 2)
        
        out = out.contiguous().view(B, T, D)
        
        out = self.out_proj(out)
        
        return out

class DecoderMLP(nn.Module):
    def __init__(self, embed_dim: int=512, mlp_ratio: int = 4 , dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = mlp_ratio * embed_dim
        
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        return self.mlp(x)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = DecoderMLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        
        x = x + self.mlp(self.norm2(x))
        
        return x
    

class TinyDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        visual_dim: int = 512,
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8, 
        mlp_ratio: int = 4, 
        max_seq_len: int = 256,
        dropout: float = 0.1,
        ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )
        
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim, 
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self._init_weights()
        
    def forward(self, text_token_ids: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        text_embeddings = self.token_embedding(text_token_ids)
        
        x = torch.cat([visual_tokens, text_embeddings], dim=1 )
        seq_len = x.size(1)
        x = x + self.position_embedding[:, :seq_len, :]
        
        for block in self.blocks: 
            x = block(x)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    

if __name__ == "__main__":
    vocab_size = 151936
    
    decoder = TinyDecoder(
        vocab_size=vocab_size,
        embed_dim=512,
        num_layers=8,
        num_heads=8,
        max_seq_len=256,
    )
    
    dummy_text_ids = torch.randint(0, vocab_size, (2, 20))
    dummy_visual_tokens = torch.randn(2, 64, 512)
    
    logits = decoder(
        text_token_ids=dummy_text_ids,
        visual_tokens=dummy_visual_tokens,
    )
    
    print("Text token ids:", dummy_text_ids.shape)
    print("Visual tokens:", dummy_visual_tokens.shape)
    print("Logits:", logits.shape)

    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")