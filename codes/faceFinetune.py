import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np

# Project imports
from config import Config
from codes.utils import get_model, load_model_weights, print_parameter_stats
from codes.data import get_dynamic_loader


def verify_data_paths():
    """Verify that data paths exist"""
    # This function should be implemented based on your data structure
    pass


@torch.no_grad()
def _margin_free_logits_from_emb(model, emb):
    """
    Compute logits WITHOUT ArcFace margin for metrics (face recognition only):
      logits_eval = <normalize(emb)> · <normalize(W)>^T
    """
    emb_n = F.normalize(emb, dim=1)
    W = model.loss.weight  # [C, D]
    W_n = F.normalize(W, dim=1)
    logits_eval = F.linear(emb_n, W_n) * 64.0
    return logits_eval


def train_one_epoch(model, dataloader, optimizer, criterion, device, task_type="classification", class_offset=0):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        
        # Adjust labels based on class offset
        labels = (labels - class_offset).to(device, non_blocking=True).long()

        optimizer.zero_grad()
        
        if task_type == "classification":
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            
        else:  # face recognition
            logits_train, embeddings = model(images, labels)
            loss = criterion(logits_train, labels)
            
            # Use margin-free logits for predictions
            with torch.no_grad():
                logits_eval = _margin_free_logits_from_emb(model, embeddings)
                preds = logits_eval.argmax(dim=1).cpu().numpy()
            
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        current_acc = (torch.tensor(preds) == labels.cpu()).float().mean().item() * 100
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    return avg_loss, acc, precision, recall, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, task_type="classification", class_offset=0):
    """Validation loop"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = (labels - class_offset).to(device, non_blocking=True).long()

        if task_type == "classification":
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
        else:  # face recognition
            logits_train, embeddings = model(images, labels)
            loss = criterion(logits_train, labels)
            
            # Use margin-free logits for predictions
            logits_eval = _margin_free_logits_from_emb(model, embeddings)
            preds = logits_eval.argmax(dim=1).cpu().numpy()

        total_loss += loss.item()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    return avg_loss, acc, precision, recall, f1


def train_model_for_class_range(class_start, class_end, task_type, pretrained_path=None):
    """Train model for a specific class range"""
    
    class_range = (class_start, class_end)
    num_classes = class_end - class_start + 1
    class_offset = class_start
    
    print(f"\n🚀 Training {task_type.upper()} Model on Classes: {class_start}–{class_end}")
    print(f"📊 Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model using your utils function
    model = get_model(
        config=Config,
        num_classes=100, 
        pretrained=False,
        device=device
    )
    pretrained_path = "checkpoints/face/oracle/0_49.pth"
    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        load_model_weights(model, pretrained_path, strict=False)
    
    # Print parameter statistics
    print_parameter_stats(model)

    # Data loaders using your data function
    train_loader = get_dynamic_loader(
        class_range=class_range, 
        mode="train", 
        batch_size=64 if task_type == "classification" else 32,
        image_size=224,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )
    
    val_loader = get_dynamic_loader(
        class_range=class_range, 
        mode="val" if task_type == "classification" else "test", 
        batch_size=32 if task_type == "classification" else 64,
        image_size=224,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    # Training configuration
    if task_type == "classification":
        num_epochs = 30
        lr = 3e-4
        weight_decay = 0.1
        label_smoothing = 0.1
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:  # face recognition
        num_epochs = 150  
        lr = 1e-4
        weight_decay = 1e-4
        criterion = nn.CrossEntropyLoss()

    # Optimizer with different learning rates for backbone vs head
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'loss' in name or 'head' in name or 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    if len(head_params) > 0 and len(backbone_params) > 0:
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': lr}  # Higher LR for head
        ], weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # Training tracking
    best_val_acc = 0.0
    patience = 1000
    patience_counter = 0

    # Create save directory
    save_dir = f"checkpoints/{task_type}/oracle"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📁 Models will be saved to: {save_dir}")
    print(f"🎯 Target epochs: {num_epochs}, Learning rate: {lr}")

    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{num_epochs} — {task_type.upper()} — Classes: {class_start}-{class_end}")

        # Training
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, task_type, class_offset
        )
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device, task_type, class_offset
        )

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics
        print(f"📊 Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
              f"Prec: {train_prec:.2f}%, Rec: {train_rec:.2f}%, F1: {train_f1:.2f}%")
        print(f"📊 Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
              f"Prec: {val_prec:.2f}%, Rec: {val_rec:.2f}%, F1: {val_f1:.2f}%")
        print(f"📈 Learning Rate: {current_lr:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_path = f"{save_dir}/{class_start}_{class_end}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ New Best Val Acc: {val_acc:.2f}% — Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"📉 No improvement. Best so far: {best_val_acc:.2f}% "
                  f"(patience: {patience_counter}/{patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

    training_time = (time.time() - start_time) / 60.0
    print(f"🏁 Training finished in {training_time:.1f} minutes.")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    return best_val_acc


def main():
    """Main function to train all models"""
    
    # Get task type from config
    task_name = getattr(Config, 'taskName', getattr(Config, 'TaskName', 'classification')).lower()
    task_type = "classification" if task_name == "classification" else "face"
    
    print(f"🔥 Starting Training for {task_type.upper()} task")
    print(f"📁 Models will be saved to: checkpoints/{task_type}/oracle/")
    
    # Verify data paths
    try:
        verify_data_paths()
    except:
        print("⚠️ Data path verification function not available, continuing...")
    
    # Create directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs(f"./checkpoints/{task_type}", exist_ok=True)
    os.makedirs(f"./checkpoints/{task_type}/oracle", exist_ok=True)

    # Define class ranges for training
    class_ranges = [(0, 49)]
    # class_ranges = [(0, 49), (10, 59), (20, 69), (30, 79), (40, 89), (50, 99)]
    
    results = {}
    total_start_time = time.time()
    
    # Optional: Load a base pretrained model for fine-tuning
    # Uncomment and adjust path if you have a pretrained model to start from
    # pretrained_base_path = f"./pretrained_models/{task_type}/base_model.pth"
    pretrained_base_path = None

    print(f"\n🎯 Training plan:")
    for i, (start, end) in enumerate(class_ranges, 1):
        print(f"   {i}. Classes {start:2d}-{end:2d} ({end-start+1:2d} classes)")
    print()

    # Train models for each class range
    for i, (start, end) in enumerate(class_ranges, 1):
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING MODEL {i}/{len(class_ranges)}")
        print(f"{'='*80}")
        
        try:
            best_acc = train_model_for_class_range(
                class_start=start,
                class_end=end,
                task_type=task_type,
                pretrained_path=pretrained_base_path
            )
            
            results[(start, end)] = best_acc
            print(f"\n✅ Completed training for classes {start}-{end}: {best_acc:.2f}%")
            
        except Exception as e:
            print(f"\n❌ Error training classes {start}-{end}: {str(e)}")
            results[(start, end)] = 0.0
            continue

    # Print final summary
    total_time = (time.time() - total_start_time) / 60.0
    
    print(f"\n{'='*80}")
    print(f"🎉 ALL TRAINING COMPLETED! ({total_time:.1f} minutes total)")
    print(f"{'='*80}")
    
    print(f"\n📊 Final Results Summary ({task_type.upper()}):")
    print(f"{'='*50}")
    
    valid_results = [(k, v) for k, v in results.items() if v > 0]
    
    for (start, end), acc in results.items():
        status = "✅" if acc > 0 else "❌"
        print(f"   {status} Classes {start:2d}-{end:2d}: {acc:6.2f}%")
    
    if valid_results:
        avg_acc = sum(acc for _, acc in valid_results) / len(valid_results)
        print(f"\n🎯 Average Accuracy: {avg_acc:.2f}% ({len(valid_results)}/{len(class_ranges)} successful)")
        
        best_range, best_acc = max(valid_results, key=lambda x: x[1])
        print(f"🏆 Best Performance: Classes {best_range[0]}-{best_range[1]} with {best_acc:.2f}%")
    
    print(f"\n📁 All models saved in: checkpoints/{task_type}/oracle/")
    print(f"🔍 Model files: {start}_{end}.pth for each class range")


if __name__ == "__main__":
    main()