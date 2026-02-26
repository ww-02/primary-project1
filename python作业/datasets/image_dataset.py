import os
from PIL import Image
from .base_dataset import BaseDataset

class ImageClsDataset(BaseDataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__(root, split, transform)
        self.data_path = os.path.join(root, split)
        self.classes = sorted(os.listdir(self.data_path))
        self.class2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.img_paths, self.labels = self._load_data()

    def _load_data(self):
        img_paths, labels = [], []
        for cls in self.classes:
            cls_dir = os.path.join(self.data_path, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith((".jpg", ".png", ".jpeg")):
                    img_paths.append(os.path.join(cls_dir, img_name))
                    labels.append(self.class2idx[cls])
        return img_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label