import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from config import Config
from codes.utils import get_model, load_model_weights
from codes.data import get_dynamic_loader

# ----------------- Training & Evaluation Functions -----------------

def train_one_epoch(model, dataloader, optimizer, criterion, device, task_type="classification"):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        if task_type == "classification":
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
        else:  # face recognition
            loss_output, embeddings = model(images, labels)
            loss = criterion(loss_output, labels)
            preds = loss_output.argmax(dim=1).detach().cpu().numpy()
            
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device, task_type="classification"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if task_type == "classification":
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1).cpu().numpy()
            else:  # face recognition
                loss_output, embeddings = model(images, labels)
                loss = criterion(loss_output, labels)
                preds = loss_output.argmax(dim=1).cpu().numpy()

            total_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ----------------- Oracle Training Loop -----------------

def train_oracle_model(class_start, class_end):
    class_range = (class_start, class_end)
    
    # Get task type from config
    task_name = getattr(Config, 'taskName', getattr(Config, 'TaskName', 'classification')).lower()
    task_type = "classification" if task_name == "classification" else "face"
    
    print(f"\n🚀 Training Oracle Model ({task_type.upper()}) on Classes: {class_start}–{class_end}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100 

    # Create model based on config
    model = get_model(
        config=Config,
        num_classes=num_classes, 
        pretrained=True,  
        device=device
    )

    # Data
    train_loader = get_dynamic_loader(class_range=class_range, mode="train", batch_size=64)
    val_loader = get_dynamic_loader(class_range=class_range, mode="val", batch_size=32)

    # Training config
    num_epochs = 30
    lr = 3e-4
    weight_decay = 0.1
    label_smoothing = 0.1

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0

    # Create save directory based on task type
    save_dir = f"checkpoints/{task_type}/oracle"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{num_epochs} — {task_type.upper()} — Class Range: {class_start}-{class_end}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, task_type)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, task_type)

        print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"📊 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"{save_dir}/{class_start}_{class_end}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best Val Acc: {val_acc * 100:.2f}% — Model saved to {save_path}")
        else:
            print(f"No improvement. Best so far: {best_val_acc * 100:.2f}%")

        scheduler.step()

    print("🏁 Training finished.")


# ----------------- Main -----------------

def main():
    # Get task type from config for directory creation
    task_name = getattr(Config, 'taskName', getattr(Config, 'TaskName', 'classification')).lower()
    task_type = "classification" if task_name == "classification" else "face"
    
    print(f"🔥 Starting Oracle Training for {task_type.upper()} task")
    print(f"📁 Models will be saved to: checkpoints/{task_type}/oracle/")
    
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs(f"./checkpoints/{task_type}", exist_ok=True)
    os.makedirs(f"./checkpoints/{task_type}/oracle", exist_ok=True)

    class_ranges = [(0, 49), (10, 59), (20, 69), (30, 79), (40, 89), (50, 99)]

    for start, end in class_ranges:
        train_oracle_model(start, end)

    print(f"\n🎉 All Oracle models trained and saved in checkpoints/{task_type}/oracle/")


if __name__ == "__main__":
    main()