import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def get_train_image_transform():
    return transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def get_val_image_transform():
    return transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

class ImageOnlyDataset(Dataset):
    def __init__(self, df, image_cols, y_col, root_dir, train=True):
        self.df = df.reset_index(drop=True)
        self.image_cols = image_cols
        self.y_col = y_col
        self.root_dir = root_dir
        self.train = train
        self.transform = get_train_image_transform() if train else get_val_image_transform()

    def _load_image(self, path):
        if not isinstance(path, str) or not os.path.exists(os.path.join(self.root_dir, path)):
            #return torch.zeros(3, 224, 224)
            return torch.zeros(3, 448, 448)
        
        img = Image.open(os.path.join(self.root_dir, path)).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs = []
        valid = 0
        for col in self.image_cols:
            img = self._load_image(row[col])
            if torch.count_nonzero(img) > 0:
                valid += 1
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        label = int(row[self.y_col])
        return {"images": imgs, "image_valid_num": torch.tensor(valid), "label": label}
