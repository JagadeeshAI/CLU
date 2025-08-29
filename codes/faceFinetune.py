import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
from backbone.vitface import ViTs_face
# Project imports
from config import Config
from codes.utils import get_model, load_model_weights, print_parameter_stats
from codes.data import get_dynamic_loader


@torch.no_grad()
def _margin_free_logits_from_emb(model, emb):
    """
    Compute logits WITHOUT ArcFace margin for metrics (face recognition only):
      logits_eval = <normalize(emb)> · <normalize(W)>^T
    """
    emb_n = F.normalize(emb, dim=1)
    
    # Check if model has ArcFace loss layer
    if hasattr(model, 'loss') and hasattr(model.loss, 'weight'):
        W = model.loss.weight  # [C, D]
        W_n = F.normalize(W, dim=1)
        logits_eval = F.linear(emb_n, W_n) * 64.0
    else:
        # Fallback: use final layer weights
        if hasattr(model, 'fc'):
            W = model.fc.weight
        elif hasattr(model, 'classifier'):
            W = model.classifier.weight
        else:
            # Create dummy weights if none found
            W = torch.randn(emb.size(1), emb.size(1), device=emb.device)
        
        W_n = F.normalize(W, dim=1)
        logits_eval = F.linear(emb_n, W_n) * 64.0
    
    return logits_eval


def train_one_epoch(model, dataloader, optimizer, criterion, device, class_offset=0):
    """Training loop for one epoch - Face Recognition only"""
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
        
        # Face recognition training
        try:
            logits_train, embeddings = model(images, labels)
            loss = criterion(logits_train, labels)
            
            # Use margin-free logits for predictions
            with torch.no_grad():
                logits_eval = _margin_free_logits_from_emb(model, embeddings)
                preds = logits_eval.argmax(dim=1).cpu().numpy()
        except (TypeError, ValueError):
            # Fallback for models that don't support face recognition mode
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            
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
def evaluate(model, dataloader, criterion, device, class_offset=0):
    """Validation loop - Face Recognition only"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = (labels - class_offset).to(device, non_blocking=True).long()

        # Face recognition evaluation
        try:
            logits_train, embeddings = model(images, labels)
            loss = criterion(logits_train, labels)
            
            # Use margin-free logits for predictions
            logits_eval = _margin_free_logits_from_emb(model, embeddings)
            preds = logits_eval.argmax(dim=1).cpu().numpy()
        except (TypeError, ValueError):
            # Fallback for models that don't support face recognition mode
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1).cpu().numpy()

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


def train_model_for_class_range(class_start, class_end, pretrained_path=None):
    """Train face recognition model for a specific class range"""
    
    class_range = (class_start, class_end)
    num_classes = class_end - class_start + 1
    class_offset = class_start
    
    print(f"\n🚀 Training FACE RECOGNITION Model on Classes: {class_start}–{class_end}")
    print(f"📊 Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Ensure Config is set to face_recognition
    Config.TaskName = "face_recognition"
    if hasattr(Config, 'taskName'):
        Config.taskName = "face_recognition"
    
    print(f"🔧 Config.TaskName set to: '{Config.TaskName}'")
    
    # Create face recognition model
    model = ViTs_face(
        loss_type="ArcFace",
        GPU_ID=[0], 
        num_class=num_classes,
        image_size=224, 
        patch_size=16, 
        ac_patch_size=8, 
        pad=0,
        dim=192, 
        depth=12, 
        heads=3, 
        mlp_dim=768, 
        dim_head=64,
        lora_rank=0
    )
    
    # CRITICAL FIX: Move model to device BEFORE loading weights
    model = model.to(device)
    print(f"✅ Model moved to device: {device}")
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        # Load weights to the same device
        checkpoint = torch.load(pretrained_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights with proper device mapping
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Pretrained weights loaded and mapped to {device}")
    
    print_parameter_stats(model)

    train_loader = get_dynamic_loader(
        class_range=class_range, 
        mode="train", 
        batch_size=64,
        image_size=224,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )
    
    val_loader = get_dynamic_loader(
        class_range=class_range, 
        mode="test", 
        batch_size=32,
        image_size=224,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    num_epochs = 150  
    lr = 1e-4
    weight_decay = 1e-4
    criterion = nn.CrossEntropyLoss()

    # Separate backbone and head parameters
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in ['loss', 'head', 'classifier', 'fc']):
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    # Create optimizer with different learning rates
    if len(head_params) > 0 and len(backbone_params) > 0:
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  
            {'params': head_params, 'lr': lr}  
        ], weight_decay=weight_decay)
        print(f"📈 Using differential learning rates: backbone={lr*0.1:.2e}, head={lr:.2e}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"📈 Using uniform learning rate: {lr:.2e}")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # Training tracking
    best_val_acc = 0.0
    patience = 1000
    patience_counter = 0

    # Create save directory
    save_dir = "checkpoints_face/face_recognition/oracle"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📁 Models will be saved to: {save_dir}")
    print(f"🎯 Target epochs: {num_epochs}, Learning rate: {lr}")

    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{num_epochs} — FACE RECOGNITION — Classes: {class_start}-{class_end}")

        # Training
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, class_offset
        )
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device, class_offset
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
    """Main function to train all face recognition models"""
    
    print(f"🔥 Starting Training for FACE RECOGNITION task")
    print(f"📁 Models will be saved to: checkpoints_face/face_recognition/oracle/")
    
    # Create directories
    os.makedirs("./checkpoints_face", exist_ok=True)
    os.makedirs("./checkpoints_face/face_recognition", exist_ok=True)
    os.makedirs("./checkpoints_face/face_recognition/oracle", exist_ok=True)

    class_ranges = [(10, 59), (20, 69), (30, 79), (40, 89), (50, 99)]
    
    results = {}
    total_start_time = time.time()
   
    pretrained_base_path = None

    print(f"\n🎯 Training plan:")
    for i, (start, end) in enumerate(class_ranges, 1):
        print(f"   {i}. Classes {start:2d}-{end:2d} ({end-start+1:2d} classes)")
    print()

    # Train models for each class range
    for i, (start, end) in enumerate(class_ranges, 1):
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING FACE RECOGNITION MODEL {i}/{len(class_ranges)}")
        print(f"{'='*80}")
        
        try:
            best_acc = train_model_for_class_range(
                class_start=start,
                class_end=end,
                pretrained_path=pretrained_base_path
            )
            
            results[(start, end)] = best_acc
            print(f"\n✅ Completed training for classes {start}-{end}: {best_acc:.2f}%")
            
        except Exception as e:
            print(f"\n❌ Error training classes {start}-{end}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[(start, end)] = 0.0
            continue

    # Print final summary
    total_time = (time.time() - total_start_time) / 60.0
    
    print(f"\n{'='*80}")
    print(f"🎉 ALL FACE RECOGNITION TRAINING COMPLETED! ({total_time:.1f} minutes total)")
    print(f"{'='*80}")
    
    print(f"\n📊 Final Results Summary (FACE RECOGNITION):")
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
    
    print(f"\n📁 All models saved in: checkpoints_face/face_recognition/oracle/")
    print(f"🔍 Model files: [start]_[end].pth for each class range")


if __name__ == "__main__":
    main()