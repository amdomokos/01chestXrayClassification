import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms

from .splits import CLASSES, DATA_DIR, MANIFEST

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ManifestDataset(Dataset):
    """Reads one split ('train' | 'val' | 'test') from data/splits.csv."""

    def __init__(self, split, transform=None, manifest=MANIFEST):
        df = pd.read_csv(manifest)
        df = df[df["split"] == split].reset_index(drop=True)
        self.paths = [DATA_DIR / p for p in df["path"]]
        self.targets = df["label"].tolist()
        self.classes = CLASSES
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[i]


def make_datasets():
    return (ManifestDataset("train", train_transforms),
            ManifestDataset("val", test_transforms),
            ManifestDataset("test", test_transforms))


def make_loaders(batch_size=32, num_workers=4, pin_memory=False):
    train_ds, val_ds, test_ds = make_datasets()

    counts = torch.bincount(torch.tensor(train_ds.targets)).float()
    class_weights = 1.0 / counts
    sample_weights = class_weights[torch.tensor(train_ds.targets)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    kw = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return (DataLoader(train_ds, batch_size, sampler=sampler, **kw),
            DataLoader(val_ds, batch_size, shuffle=False, **kw),
            DataLoader(test_ds, batch_size, shuffle=False, **kw),
            (train_ds, val_ds, test_ds))
