import torch 
import torch.nn as nn 

from vision_encoder import TinyViEncoder
from projector import VisualProjector
from decoder import TinyDecoder

### We are using GPT-2 Tokenizer for easier training 
from transformers import AutoTokenizer

class TinyQwenVL(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_size: int = 32,
        patch_size: int = 4,
        vision_dim: int = 512,
        language_dim: int = 512,
        vision_layers: int = 8, 
        decoder_layers: int = 8, 
        num_heads: int = 8,
        max_seq_len: int = 256, 
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vision_encoder = TinyViEncoder(
    image_size=image_size,
    patch_size=patch_size,
    embed_dim=vision_dim,
    num_layers=vision_layers,
    num_heads=num_heads,
    dropout=dropout,
)
        self.projector = VisualProjector(
    vision_dim=vision_dim,
    language_dim=language_dim,
    hidden_dim=4 * language_dim,
    dropout=dropout,
)
        self.decoder = TinyDecoder(
    vocab_size=vocab_size,
    embed_dim=language_dim,
    num_layers=decoder_layers,
    num_heads=num_heads,
    max_seq_len=max_seq_len,
    dropout=dropout,
)
    
    def forward(
        self, 
        images: torch.Tensor,
        text_token_ids: torch.Tensor, 
    ) -> torch.Tensor:
        visual_tokens = self.vision_encoder(images)
        visual_tokens = self.projector(visual_tokens)
        
        logits = self.decoder(
            text_token_ids = text_token_ids,
            visual_tokens = visual_tokens,
        )
        
        return logits

if __name__ == "__main__":
    ### We are using GPT-2 Tokenizer for easier training 
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    model = TinyQwenVL(
        vocab_size=vocab_size,
        image_size=224,
        patch_size=16,
        vision_dim=512,
        language_dim=512,
        vision_layers=8,
        decoder_layers=8,
        num_heads=8,
        max_seq_len=512,
    )

    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_text_ids = torch.randint(0, vocab_size, (2, 20))

    logits = model(
        images=dummy_images,
        text_token_ids=dummy_text_ids,
    )

    print("Images:", dummy_images.shape)
    print("Text token ids:", dummy_text_ids.shape)
    print("Logits:", logits.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")