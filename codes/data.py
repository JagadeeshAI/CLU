# import os
# import random
# import torch
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets, transforms
# from PIL import Image
# import numpy as np
# from config import Config

# # Hardcoded paths for the datasets
# # Update these paths to match your system
# ROOT_CIFAR_TRAIN = "/media/jag/volD2/cifer100/cifer/train"
# ROOT_CIFAR_VAL = "/media/jag/volD2/cifer100/cifer/val"
# ROOT_FACE_TRAIN = "/media/jag/volD2/faces_webface_112x112_sub100_train_test/train"
# ROOT_FACE_VAL = "/media/jag/volD2/faces_webface_112x112_sub100_train_test/test"

# # --- Common Helper Classes from the Original Files ---

# class SafeColorJitter(transforms.ColorJitter):
#     """Safe ColorJitter that handles potential issues with small images"""
#     def __init__(self, brightness=0, contrast=0, saturation=0):
#         super().__init__(
#             brightness=brightness, contrast=contrast, saturation=saturation, hue=0
#         )
#     def __call__(self, img):
#         try:
#             return super().__call__(img)
#         except:
#             return img

# class FaceDataset(datasets.ImageFolder):
#     """
#     Custom dataset class for face recognition
#     Extends ImageFolder to handle face-specific preprocessing
#     """
#     def __init__(self, root, transform=None, target_transform=None):
#         super().__init__(root, transform, target_transform)

#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if sample.mode != "RGB":
#             sample = sample.convert("RGB")
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return sample, target

# # --- Unified get_dynamic_loader function ---

# def get_dynamic_loader(
#     class_range=(0, 99),
#     mode="train",
#     batch_size=32,
#     image_size=224,
#     num_workers=0,
#     data_percentage=1.0,
#     pin_memory=False,
# ):
    
#     """
#     Create a data loader for either face recognition or classification datasets.

#     The dataset loaded is determined by the `Config.TaskName` variable.

#     Args:
#         class_range (tuple): Range of classes to include (start, end) - inclusive.
#         mode (str): "train" or "test"/"val".
#         batch_size (int): Batch size for training.
#         image_size (int): Target image size.
#         num_workers (int): Number of worker processes.
#         data_percentage (float): Percentage of data to use (0.0 to 1.0).
#         pin_memory (bool): Whether to use pin memory.

#     Returns:
#         DataLoader: PyTorch DataLoader object.
#     """
#     task_name = Config.TaskName.lower()
    
#     # 1. Select the correct dataset based on the task name
#     if "face" in task_name:
#         data_dir = ROOT_FACE_TRAIN if mode == "train" else ROOT_FACE_VAL
#         dataset_class = FaceDataset
#         # print(f"Loading Face Recognition data from {data_dir}")
#     elif "classif" in task_name:
#         data_dir = ROOT_CIFAR_TRAIN if mode == "train" else ROOT_CIFAR_VAL
#         dataset_class = datasets.ImageFolder
#         # print(f"Loading Classification (CIFAR-100) data from {data_dir}")
#     else:
#         raise ValueError(f"Unknown task name: {task_name}")

#     if not os.path.exists(data_dir):
#         raise ValueError(f"Data directory does not exist: {data_dir}")

#     # 2. Define transforms based on the task and mode
#     if mode == "train":
#         if "face" in task_name:
#             transform = transforms.Compose([
#                 transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#                 transforms.RandomRotation(degrees=5),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#             ])
#         else: # Classification task
#             transform = transforms.Compose([
#                 transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
#                 transforms.RandomHorizontalFlip(),
#                 SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#             ])
#     else: # Test/Validation mode
#         transform = transforms.Compose([
#             transforms.Resize(int(image_size * 1.14)),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ])

#     # 3. Create the dataset instance
#     dataset = dataset_class(root=data_dir, transform=transform)

#     # 4. Filter classes based on range
#     start_class, end_class = class_range
#     allowed_classes = list(range(start_class, end_class + 1))
    
#     indices = [
#         i for i, (_, label) in enumerate(dataset.samples) if label in allowed_classes
#     ]

#     if len(indices) == 0:
#         raise ValueError(f"No samples found for classes {class_range} in {data_dir}")

#     # 5. Apply data percentage sampling
#     if data_percentage < 1.0:
#         num_samples = int(len(indices) * data_percentage)
#         indices = random.sample(indices, min(num_samples, len(indices)))

#     # 6. Create the subset and DataLoader
#     subset_dataset = Subset(dataset, indices)
    
#     loader = DataLoader(
#         subset_dataset,
#         batch_size=batch_size,
#         shuffle=(mode == "train"),
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         persistent_workers=(num_workers > 0),
#         drop_last=(mode == "train"),
#     )

#     return loader

# # --- Example Usage ---

# if __name__ == "__main__":
#     print("=== Testing the Unified Data Loader ===")
    
#     # Example 1: Load Face Recognition data
#     Config.TaskName = "Face_recognition"
#     print(f"\nTask set to: {Config.TaskName}")
#     try:
#         train_loader_face = get_dynamic_loader(
#             class_range=(0, 9), mode="train", batch_size=16, data_percentage=0.1
#         )
#         val_loader_face = get_dynamic_loader(
#             class_range=(0, 9), mode="test", batch_size=16, data_percentage=0.1
#         )
#         train_images, train_labels = next(iter(train_loader_face))
#         print(f"✓ Face Train loader created with {len(train_loader_face)} batches.")
#         print(f"  Sample batch shape: {train_images.shape}, Labels shape: {train_labels.shape}")
#     except Exception as e:
#         print(f"✗ Error loading Face data: {e}")

#     # Example 2: Load Classification data (CIFAR-100)
#     Config.TaskName = "classification"
#     print(f"\nTask set to: {Config.TaskName}")
#     try:
#         train_loader_cifar = get_dynamic_loader(
#             class_range=(0, 19), mode="train", batch_size=16, data_percentage=0.1
#         )
#         val_loader_cifar = get_dynamic_loader(
#             class_range=(0, 19), mode="val", batch_size=16, data_percentage=0.1
#         )
#         train_images, train_labels = next(iter(train_loader_cifar))
#         print(f"✓ CIFAR-100 Train loader created with {len(train_loader_cifar)} batches.")
#         print(f"  Sample batch shape: {train_images.shape}, Labels shape: {train_labels.shape}")
#     except Exception as e:
#         print(f"✗ Error loading CIFAR-100 data: {e}")



import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from config import Config

# Hardcoded paths for the datasets
ROOT_CIFAR_TRAIN = "/media/jag/volD2/cifer100/cifer/train"
ROOT_CIFAR_VAL   = "/media/jag/volD2/cifer100/cifer/val"
ROOT_FACE_TRAIN  = "/media/jag/volD2/faces_webface_112x112_sub100_train_test/train"
ROOT_FACE_VAL    = "/media/jag/volD2/faces_webface_112x112_sub100_train_test/test"

# ---------- NEW: Numeric-sorted ImageFolder ----------
class NumericImageFolder(datasets.ImageFolder):
    """
    ImageFolder that maps class folder names numerically:
      '0','1','2',...'10',...'99' -> labels 0..99
    Falls back to alphabetical if non-numeric dirs are present.
    """
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        try:
            # sort by integer value of folder names
            classes = sorted(classes, key=lambda x: int(x))
        except ValueError:
            # fallback: default alphabetical
            classes = sorted(classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# ---------- Common Helper Classes ----------
class SafeColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=0)
    def __call__(self, img):
        try:
            return super().__call__(img)
        except:
            return img

class FaceDataset(NumericImageFolder):  # <-- inherit numeric-sorted folder
    """
    Custom dataset class for face recognition with numeric class mapping.
    """
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample.mode != "RGB":
            sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

# ---------- Unified get_dynamic_loader ----------
def get_dynamic_loader(
    class_range=(0, 99),
    mode="train",
    batch_size=32,
    image_size=224,
    num_workers=0,
    data_percentage=1.0,
    pin_memory=False,
):
    """
    Create a data loader for face recognition or classification.
    Uses numeric folder sorting so class_range=(a,b) means true IDs a..b.
    """
    task_name = Config.TaskName.lower()

    # 1) Choose dataset + root
    if "face" in task_name:
        data_dir = ROOT_FACE_TRAIN if mode == "train" else ROOT_FACE_VAL
        dataset_class = FaceDataset                     # numeric-sorted
    elif "classif" in task_name:
        data_dir = ROOT_CIFAR_TRAIN if mode == "train" else ROOT_CIFAR_VAL
        dataset_class = NumericImageFolder              # numeric-sorted
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # 2) Transforms
    if mode == "train":
        if "face" in task_name:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:  # classification
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    else:  # test/val
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    # 3) Dataset (numeric class_to_idx)
    dataset = dataset_class(root=data_dir, transform=transform)

    # 4) Filter by true numeric class IDs
    start_class, end_class = class_range
    allowed = set(range(start_class, end_class + 1))

    # sanity check: ensure mapping matches numeric ids for numeric folders
    # (this will be true for faces/cifar with folders 0..99)
    # Uncomment if you want to verify once:
    # for k in [0, 1, 9, 10, 11, 49, 50, 59, 99]:
    #     if str(k) in dataset.class_to_idx:
    #         assert dataset.class_to_idx[str(k)] == k, f"Mapping mismatch for class {k}"

    indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl in allowed]
    if len(indices) == 0:
        raise ValueError(f"No samples found for classes {class_range} in {data_dir}")

    # 5) Optionally subsample
    if data_percentage < 1.0:
        num = int(len(indices) * data_percentage)
        indices = random.sample(indices, min(num, len(indices)))

    # 6) DataLoader
    subset_dataset = Subset(dataset, indices)
    loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=(mode == "train"),
    )
    return loader

# --- Example self-test (optional) ---
if __name__ == "__main__":
    print("=== Testing the Unified Data Loader ===")
    # Face
    Config.TaskName = "Face_recognition"
    tr = get_dynamic_loader(class_range=(50,59), mode="train", batch_size=16, data_percentage=0.05)
    te = get_dynamic_loader(class_range=(50,59), mode="test",  batch_size=16, data_percentage=0.05)
    print("Face train batches:", len(tr), "val batches:", len(te))
    # Classification
    Config.TaskName = "classification"
    trc = get_dynamic_loader(class_range=(10,49), mode="train", batch_size=16, data_percentage=0.05)
    tec = get_dynamic_loader(class_range=(10,49), mode="val",   batch_size=16, data_percentage=0.05)
    print("Cls train batches:", len(trc), "val batches:", len(tec))
