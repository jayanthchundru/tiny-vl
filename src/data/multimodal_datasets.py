import random
from typing import Dict

from torch.utils.data import Dataset
from torchvision import datasets, transforms
DATA_ROOT = "/data/jiang/chundrja/datasets/tiny-qwen-vl"

GENERAL_PROMPTS = [
    "What is this image?",
    "What can you see in this image?",
    "Identify the main object in the image.",
    "Describe this image briefly.",
    "What is shown in the picture?",
    "Name the main thing in this image.",
    "Classify this image.",
]

FOOD_PROMPTS = [
    "What food is shown in this image?",
    "Identify the dish in the picture.",
    "What meal or dish is visible?",
    "Name the food item shown.",
    "Describe the food in this image.",
]

PET_PROMPTS = [
    "What pet breed is shown?",
    "Identify the animal in this image.",
    "What kind of pet is this?",
    "Name the breed in the picture.",
    "Describe the pet shown in this image.",
]

GENERAL_ANSWER_TEMPLATES = [
    "This is a {label}.",
    "The image shows a {label}.",
    "This picture contains a {label}.",
    "It looks like a {label}.",
    "The main object is a {label}.",
    "This appears to be a {label}.",
]

FOOD_ANSWER_TEMPLATES = [
    "This is {label}.",
    "The dish shown is {label}.",
    "This food appears to be {label}.",
    "The image shows {label}.",
]

PET_ANSWER_TEMPLATES = [
    "This is a {label}.",
    "The pet shown is a {label}.",
    "This animal appears to be a {label}.",
    "The image shows a {label}.",
]


def build_image_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


def clean_label(label: str) -> str:
    return label.replace("_", " ").lower()


class CIFAR100VQADataset(Dataset):
    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        image_size: int = 224,
        download: bool = True,
    ):
        self.dataset = datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=build_image_transform(image_size),
        )
        self.class_names = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        image, label = self.dataset[idx]

        class_name = clean_label(self.class_names[label])
        prompt = random.choice(GENERAL_PROMPTS)
        answer = random.choice(GENERAL_ANSWER_TEMPLATES).format(label=class_name)

        return {
            "image": image,
            "prompt": prompt,
            "answer": answer,
            "label": class_name,
            "source": "cifar100",
        }


class Food101VQADataset(Dataset):
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        image_size: int = 224,
        download: bool = True,
    ):
        self.dataset = datasets.Food101(
            root=root,
            split=split,
            download=download,
            transform=build_image_transform(image_size),
        )
        self.class_names = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        image, label = self.dataset[idx]

        class_name = clean_label(self.class_names[label])
        prompt = random.choice(GENERAL_PROMPTS + FOOD_PROMPTS)
        answer = random.choice(FOOD_ANSWER_TEMPLATES).format(label=class_name)

        return {
            "image": image,
            "prompt": prompt,
            "answer": answer,
            "label": class_name,
            "source": "food101",
        }


class OxfordPetsVQADataset(Dataset):
    def __init__(
        self,
        root: str = "data",
        split: str = "trainval",
        image_size: int = 224,
        download: bool = True,
    ):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="category",
            download=download,
            transform=build_image_transform(image_size),
        )
        self.class_names = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        image, label = self.dataset[idx]

        class_name = clean_label(self.class_names[label])
        prompt = random.choice(GENERAL_PROMPTS + PET_PROMPTS)
        answer = random.choice(PET_ANSWER_TEMPLATES).format(label=class_name)

        return {
            "image": image,
            "prompt": prompt,
            "answer": answer,
            "label": class_name,
            "source": "oxford_pets",
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    datasets_to_test = [
        CIFAR100VQADataset(
            root=DATA_ROOT,
            train=True,
            image_size=224,
            download=True,
        ),
        Food101VQADataset(
            root=DATA_ROOT,
            split="train",
            image_size=224,
            download=True,
        ),
        OxfordPetsVQADataset(
            root=DATA_ROOT,
            split="trainval",
            image_size=224,
            download=True,
        ),
    ]

    for dataset in datasets_to_test:
        sample = dataset[500]
        print("Dataset size:", len(dataset))
        print("=" * 80)
        print("Image shape:", sample["image"].shape)
        print("Prompt:", sample["prompt"])
        print("Answer:", sample["answer"])
        print("Label:", sample["label"])
        print("Source:", sample["source"])

        image = sample["image"] * 0.5 + 0.5
        image = image.permute(1, 2, 0)

        output_path = f"sample_{sample['source']}.png"

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"{sample['source']}: {sample['label']}")
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close()

        print(f"Saved image to {output_path}")