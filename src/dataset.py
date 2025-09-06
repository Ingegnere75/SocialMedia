import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import emoji as emoji_lib

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def default_image_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def count_emojis(text: str) -> int:
    return sum(1 for ch in text if ch in emoji_lib.EMOJI_DATA)

def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    ups = sum(1 for c in letters if c.isupper())
    return ups / len(letters)

class SocialPostDataset(Dataset):
    # CSV richiesto: image_path, text, label[, ...feature numeriche opzionali...]
    # Se non fornisci colonne numeriche, vengono calcolate da text:
    #   - text_len, emoji_count, upper_ratio
    def __init__(self, csv_path, image_root=None, tokenizer=None, numeric_cols=None, img_size=224):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        if 'text' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError("CSV must contain at least ['text','label'] columns.")
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.numeric_cols = numeric_cols or []
        self.img_tf = default_image_transform(img_size)

        # image present?
        self.has_image = 'image_path' in self.df.columns and self.df['image_path'].notna().any()

        # numeric derived if not provided
        if not self.numeric_cols:
            self.df['text_len'] = self.df['text'].astype(str).apply(lambda s: len(s))
            self.df['emoji_count'] = self.df['text'].astype(str).apply(count_emojis)
            self.df['upper_ratio'] = self.df['text'].astype(str).apply(uppercase_ratio)
            self.numeric_cols = ['text_len', 'emoji_count', 'upper_ratio']

        # label encoding if needed
        if self.df['label'].dtype == object:
            classes = sorted(self.df['label'].unique())
            self.class_to_idx = {c:i for i,c in enumerate(classes)}
            self.idx_to_class = {i:c for c,i in self.class_to_idx.items()}
            self.df['label'] = self.df['label'].map(self.class_to_idx)
        else:
            self.class_to_idx = None
            self.idx_to_class = None

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        if (not self.has_image) or pd.isna(rel_path):
            return torch.zeros(3, 224, 224)  # dummy
        p = os.path.join(self.image_root, rel_path) if self.image_root else rel_path
        with Image.open(p).convert("RGB") as img:
            return self.img_tf(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        image = self._load_image(row['image_path']) if self.has_image else torch.zeros(3,224,224)
        y = int(row['label'])
        num_feats = torch.tensor([float(row[c]) for c in self.numeric_cols], dtype=torch.float32)

        sample = {
            "text": text,
            "image": image,
            "numeric": num_feats,
            "label": y,
        }
        return sample

def collate_fn(batch, tokenizer, max_length=128):
    texts = [b["text"] for b in batch]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    images = torch.stack([b["image"] for b in batch])
    nums = torch.stack([b["numeric"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return enc, images, nums, labels
