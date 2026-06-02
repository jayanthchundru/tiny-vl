from typing import List, Dict 

import torch 
from transformers import AutoTokenizer

class VQACollator:
    def __init__(self, tokenizer_name: str = "gpt2", max_text_len: int = 256):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        self.max_text_len = max_text_len
        
    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        images = torch.stack([sample['image']for sample in samples], dim=0)
        
        texts = [
            sample["answer"] 
            for sample in samples
        ]
        
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation = True, 
            max_length = self.max_text_len,
            return_tensors = "pt"
        )
        
        input_ids =  encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "texts": texts,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from multimodal_datasets import CIFAR100VQADataset

    DATA_ROOT = "/data/jiang/chundrja/datasets/tiny-qwen-vl"

    dataset = CIFAR100VQADataset(
        root=DATA_ROOT,
        train=True,
        image_size=224,
        download=False,
    )

    collator = VQACollator(
        tokenizer_name="gpt2",
        max_text_len=256,
    )

    loader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=True,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    print("Images:", batch["images"].shape)
    print("Input IDs:", batch["input_ids"].shape)
    print("Attention mask:", batch["attention_mask"].shape)
    print("Labels:", batch["labels"].shape)
    print("Texts:", batch["texts"])