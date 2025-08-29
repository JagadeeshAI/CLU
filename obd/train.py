import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import ViTModel

from data import get_dynamic_loader  # <-- your dataset loader


# ------------------------------
# Lightweight ViT Object Detector
# ------------------------------
class ViTObjectDetector(nn.Module):
    """
    Vision Transformer based Object Detector
    Uses ViT-tiny (deit-tiny-patch16-224) as backbone
    """
    def __init__(self, num_classes=100, max_objects=1, hidden_dim=192):
        super(ViTObjectDetector, self).__init__()
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.hidden_dim = hidden_dim

        # Pretrained DeiT-tiny backbone
        self.backbone = ViTModel.from_pretrained(
            "facebook/deit-tiny-patch16-224",
            output_hidden_states=True,
            output_attentions=False,
            use_safetensors=True
        )


        # Heads
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_objects * (num_classes + 1))  # +1 for background
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_objects * 4)  # [x1, y1, x2, y2]
        )
        self.obj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_objects)
        )

    def forward(self, images):
        outputs = self.backbone(pixel_values=images)
        cls_features = outputs.last_hidden_state[:, 0]  # [CLS] token
        B = images.shape[0]

        class_logits = self.class_head(cls_features).view(B, self.max_objects, self.num_classes + 1)
        box_coords = torch.sigmoid(self.box_head(cls_features).view(B, self.max_objects, 4))
        objectness = self.obj_head(cls_features).view(B, self.max_objects)

        return {
            "class_logits": class_logits,
            "box_coords": box_coords,
            "objectness": objectness
        }


# ------------------------------
# Training / Validation
# ------------------------------
def save_checkpoint(model, optimizer, epoch, best_val_loss, class_range, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_range{class_range[0]}-{class_range[1]}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "class_range": class_range
    }, ckpt_path)
    return ckpt_path


def compute_loss(outputs, targets, device):
    """
    Simple detection loss:
    - BCEWithLogits for class logits
    - SmoothL1 for bounding boxes
    - BCEWithLogits for objectness
    """
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_box = nn.SmoothL1Loss()
    criterion_obj = nn.BCEWithLogitsLoss()

    loss_cls = 0.0
    loss_box = 0.0
    loss_obj = 0.0

    for i in range(len(targets)):
        label = targets[i]["annotations"][0]["category_id"]
        bbox = torch.tensor(targets[i]["annotations"][0]["bbox"], dtype=torch.float32, device=device)

        # Class targets
        cls_target = torch.zeros(outputs["class_logits"].shape[-1], device=device)
        cls_target[label] = 1.0
        cls_pred = outputs["class_logits"][i, 0]

        # Box targets (normalize to [0,1])
        img_w = targets[i]["annotations"][0]["bbox"][2]
        img_h = targets[i]["annotations"][0]["bbox"][3]
        box_target = torch.tensor([0, 0, 1.0, 1.0], dtype=torch.float32, device=device)
        box_pred = outputs["box_coords"][i, 0]

        # Objectness target
        obj_target = torch.tensor(1.0, device=device)
        obj_pred = outputs["objectness"][i, 0]

        # Losses
        loss_cls += criterion_cls(cls_pred, cls_target)
        loss_box += criterion_box(box_pred, box_target)
        loss_obj += criterion_obj(obj_pred, obj_target)

    total_loss = loss_cls + loss_box + loss_obj
    return total_loss


def train(
    train_dir,
    val_dir,
    class_range,
    num_epochs=5,
    batch_size=8,
    lr=1e-4,
    image_size=224,
    data_percentage=1.0,
    device="cuda",
    save_dir="/home/jag/codes/CLU/obd/checkpoint"
):
    os.makedirs(save_dir, exist_ok=True)

    # Model
    num_classes = class_range[1] - class_range[0] + 1
    model = ViTObjectDetector(num_classes=num_classes, max_objects=1).to(device)

    # Data loaders
    train_loader = get_dynamic_loader(
        root_dir=train_dir,
        class_range=class_range,
        mode="train",
        batch_size=batch_size,
        image_size=image_size,
        data_percentage=data_percentage,
    )
    val_loader = get_dynamic_loader(
        root_dir=val_dir,
        class_range=class_range,
        mode="val",
        batch_size=batch_size,
        image_size=image_size,
        data_percentage=data_percentage,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # -------- Train --------
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Range {class_range}] Epoch {epoch}/{num_epochs} [Train]", leave=False)

        for images, targets in pbar:
            images = torch.stack([img for img in images]).to(device)  # already tensors
            loss = compute_loss(model(images), targets, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)

        # -------- Val --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"[Range {class_range}] Epoch {epoch}/{num_epochs} [Val]", leave=False)
            for images, targets in pbar_val:
                images = torch.stack([img for img in images]).to(device)
                loss = compute_loss(model(images), targets, device)
                val_loss += loss.item()
                pbar_val.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        print(f"[Range {class_range}] Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # -------- Save Best --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = save_checkpoint(model, optimizer, epoch, best_val_loss, class_range, save_dir)
            print(f"✅ New best val loss: {avg_val_loss:.4f}, saved checkpoint: {ckpt_path}")
        else:
            print(f"❌ No improvement, best val loss still {best_val_loss:.4f}")


if __name__ == "__main__":
    train_dir = "/media/jag/volD2/coco/coco_by_class/train2"
    val_dir = "/media/jag/volD2/coco/coco_by_class/val"

    class_ranges = [
        (0, 49),
        (10, 59),
        (20, 69),
        (30, 79),
        (40, 89),
        (50, 99),
    ]

    for cr in class_ranges:
        print(f"\n===== Training for class range {cr} =====\n")
        train(
            train_dir=train_dir,
            val_dir=val_dir,
            class_range=cr,
            num_epochs=10,
            batch_size=16,
            lr=1e-4,
            image_size=224,
            data_percentage=1.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_dir="/home/jag/codes/CLU/obd/checkpoint"
        )
