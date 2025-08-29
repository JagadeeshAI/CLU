import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms


class SingleObjectDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_range=None, data_percentage=1.0):
        """
        Args:
            root_dir (str): Path to dataset root (e.g., coco_by_class/train2 or val).
            transform (callable, optional): Transformations to apply.
            class_range (tuple): Range of classes to include (start, end) - inclusive.
            data_percentage (float): Percentage of data to use (0.0 to 1.0).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Collect all class folders
        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        # Apply class range filtering
        if class_range is not None:
            start_class, end_class = class_range
            class_names = class_names[start_class:end_class + 1]

        for idx, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            self.class_to_idx[class_name] = idx
            for img_path in glob.glob(os.path.join(class_path, "*.*")):
                if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((img_path, idx))

        # Apply percentage filtering
        if data_percentage < 1.0:
            num_samples = int(len(self.samples) * data_percentage)
            self.samples = random.sample(self.samples, num_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        coco_bbox = [0, 0, w, h]   # [x_min, y_min, width, height]
        area = float(w * h)

        target = {
            "image_id": idx,  
            "annotations": [
                {
                    "bbox": coco_bbox,
                    "category_id": label,
                    "area": area,
                    "iscrowd": 0
                }
            ]
        }

        if self.transform:
            img = self.transform(img)

        return img, target



def get_dynamic_loader(
    root_dir,
    class_range=(0, 99),
    mode="train",
    batch_size=32,
    image_size=224,
    num_workers=16,
    data_percentage=1.0,
    pin_memory=False,
):
    """
    Create a DataLoader for SingleObjectDataset with range and percentage filtering.
    """

    if not os.path.exists(root_dir):
        raise ValueError(f"Data directory does not exist: {root_dir}")

    # Define transforms
    if mode == "train":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    # Create dataset
    dataset = SingleObjectDataset(
        root_dir=root_dir,
        transform=transform,
        class_range=class_range,
        data_percentage=data_percentage
    )

    # Wrap DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=(mode == "train"),
        collate_fn=lambda x: tuple(zip(*x))  # keeps detection format
    )

    return loader


if __name__ == "__main__":
    train_dir = "/media/jag/volD2/coco/coco_by_class/train2"
    val_dir = "/media/jag/volD2/coco/coco_by_class/val"

    # Example: only classes 0–9, using 50% of data
    train_loader = get_dynamic_loader(
        root_dir=train_dir,
        class_range=(0, 9),
        mode="train",
        batch_size=4,
        data_percentage=0.5
    )

    for imgs, targets in train_loader:
        print("Batch images:", len(imgs))
        print("Target example:", targets[0])
        break
