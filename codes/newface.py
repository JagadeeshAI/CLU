import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict

# Import your modules
from data import get_dynamic_loader
from utils import get_model, print_parameter_stats, load_model_weights
from config import Config

class BIDLoRAFaceTrainer:
    """
    Bi-Directional LoRA (BID-LoRA) Trainer for Face Recognition CLU
    Implements the methodology from the paper with sliding window evaluation
    """
    
    def __init__(self, num_classes=100, lora_rank=8, device='cuda'):
        self.device = device
        self.num_classes = num_classes
        self.lora_rank = lora_rank
        
        # Updated loss weights for face recognition
        self.alpha = 0.15   # erasure loss weight
        self.gamma = 0.3    # acquisition loss weight
        self.erasure_limit = 100  # BND parameter for bounded forgetting
        print("Model submodules:", dict(self.model.named_modules()).keys())
        # Initialize model
        self.model = get_model(
            num_classes=num_classes, 
            lora_rank=lora_rank, 
            pretrained=True,
            device=device
        )
        print_parameter_stats(self.model)
        
        # Checkpoints directory
        self.checkpoint_dir = "/home/jag/codes/CLU/checkpoints/face/bid_lora"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ===============================
    # 🔑 FIX: LABEL OFFSET
    # ===============================
    def _offset_labels(self, targets, class_range):
        """Shift labels back to global IDs instead of 0..N remap from ImageFolder."""
        start_class, _ = class_range
        return targets + start_class
    
    def _freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() or 'loss' in name.lower() or 'head' in name.lower() or 'classifier' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("🔒 Frozen backbone parameters, keeping LoRA adapters and loss head trainable")
    
    def erasure_loss(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='mean')
        bounded_loss = F.relu(-ce_loss + self.erasure_limit)
        return bounded_loss
    
    def retention_loss(self, logits, targets):
        return F.cross_entropy(logits, targets, reduction='mean')
    
    def acquisition_loss(self, logits, targets):
        return F.cross_entropy(logits, targets, reduction='mean')
    
    def compute_total_loss(self, retain_logits, retain_targets, 
                          forget_logits, forget_targets,
                          new_logits, new_targets):
        l_retain = self.retention_loss(retain_logits, retain_targets)
        l_erasure = self.erasure_loss(forget_logits, forget_targets)
        l_acquisition = self.acquisition_loss(new_logits, new_targets)
        total_loss = l_retain + self.alpha * l_erasure + self.gamma * l_acquisition
        return total_loss, {
            'retention': l_retain.item(),
            'erasure': l_erasure.item(), 
            'acquisition': l_acquisition.item(),
            'total': total_loss.item()
        }
    
    def _margin_free_logits_from_emb(self, embeddings):
        """
        Compute margin-free logits from embeddings by applying the classifier head directly.
        This avoids ArcFace margin modifications during evaluation.
        """
        # Assuming model has classifier head in self.model.head or self.model.classifier
        if hasattr(self.model, 'head'):
            weight = self.model.head.weight
            scale = getattr(self.model.head, 's', 1.0)
        elif hasattr(self.model, 'classifier'):
            weight = self.model.classifier.weight
            scale = getattr(self.model.classifier, 's', 1.0)
        else:
            raise AttributeError("Model does not have a recognizable classifier head for margin-free logits.")
        # Normalize embeddings and weights (cosine similarity)
        emb_norm = F.normalize(embeddings)
        weight_norm = F.normalize(weight)
        logits = F.linear(emb_norm, weight_norm) * scale
        return logits

    # ===============================
    # VALIDATION (with label offset)
    # ===============================
    def validate_ranges(self, retain_classes, forget_classes, new_classes, verbose=True):
        self.model.eval()
        if verbose:
            print(f"\n📊 VALIDATION RESULTS:")
            print(f"=" * 60)
        metrics = {}
        with torch.no_grad():
            ranges = [
                ('Forget', forget_classes), 
                ('Retain', retain_classes), 
                ('New', new_classes)
            ]
            for name, class_range in ranges:
                if class_range[0] > class_range[1]:
                    metrics[name.lower()] = 0.0
                    continue
                loader = get_dynamic_loader(
                    class_range=class_range, mode="test", batch_size=64,
                    data_percentage=1.0
                )
                correct, total = 0, 0
                for data, targets in tqdm(loader, desc=f"Validating {name}", leave=False, disable=not verbose):
                    data, targets = data.to(self.device), targets.to(self.device)
                    # forward pass with embeddings
                    logits_train, embeddings = self.model(data, targets)
                    logits_eval = self._margin_free_logits_from_emb(embeddings)
                    _, predicted = torch.max(logits_eval.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                metrics[name.lower()] = 100 * correct / total if total > 0 else 0
                if verbose:
                    arrow = "↓" if name == "Forget" else "↑"
                    print(f"  {name} {arrow} ({class_range[0]}-{class_range[1]}): {metrics[name.lower()]:.2f}% ({correct}/{total})")
        # Overall validation
        overall_start = min(retain_classes[0], new_classes[0])
        overall_end = max(retain_classes[1], new_classes[1])
        if overall_start <= overall_end:
            loader = get_dynamic_loader(
                class_range=(overall_start, overall_end), mode="test", 
                batch_size=64, data_percentage=1.0
            )
            correct, total = 0, 0
            for data, targets in tqdm(loader, desc="Validating Overall", leave=False, disable=not verbose):
                data, targets = data.to(self.device), targets.to(self.device)
                logits_train, embeddings = self.model(data, targets)
                logits_eval = self._margin_free_logits_from_emb(embeddings)
                _, predicted = torch.max(logits_eval.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            metrics['overall'] = 100 * correct / total if total > 0 else 0
        else:
            metrics['overall'] = 0.0
        return metrics
    
    # ===============================
    # TRAINING (with validation metrics print)
    # ===============================
    def train_step(self, retain_classes, forget_classes, new_classes, num_epochs=50, step_idx=1):
        print(f"\n🔄 Training Step:")
        print(f"  📚 Retain classes: {retain_classes[0]}-{retain_classes[1]}")
        print(f"  🗑️ Forget classes: {forget_classes[0]}-{forget_classes[1]}")
        print(f"  ✨ New classes: {new_classes[0]}-{new_classes[1]}")
        
        retain_loader = get_dynamic_loader(retain_classes, mode="train", batch_size=32, data_percentage=0.15)
        forget_loader = get_dynamic_loader(forget_classes, mode="train", batch_size=32, data_percentage=1.0)
        new_loader = get_dynamic_loader(new_classes, mode="train", batch_size=32, data_percentage=1.0)
        
        backbone_params, lora_params, head_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora' in name.lower():
                lora_params.append(param)
            elif 'loss' in name.lower() or 'head' in name.lower() or 'classifier' in name.lower():
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        param_groups = []
        if backbone_params: param_groups.append({'params': backbone_params, 'lr': 1e-5})
        if lora_params: param_groups.append({'params': lora_params, 'lr': 3e-4})
        if head_params: param_groups.append({'params': head_params, 'lr': 5e-4})
        
        optimizer = optim.AdamW(param_groups if param_groups else self.model.parameters(), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        self.model.train()
        best_overall_acc = 0.0
        epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs", position=0)
        
        for epoch in epoch_pbar:
            epoch_losses = defaultdict(float)
            num_batches = 0
            retain_iter, forget_iter, new_iter = iter(retain_loader), iter(forget_loader), iter(new_loader)
            max_batches = max(len(retain_loader), len(forget_loader), len(new_loader))
            batch_pbar = tqdm(range(max_batches), desc=f"Epoch {epoch+1}", leave=False, position=1)
            
            for _ in batch_pbar:
                optimizer.zero_grad()
                try: retain_data, retain_targets = next(retain_iter)
                except StopIteration: retain_iter = iter(retain_loader); retain_data, retain_targets = next(retain_iter)
                try: forget_data, forget_targets = next(forget_iter)
                except StopIteration: forget_iter = iter(forget_loader); forget_data, forget_targets = next(forget_iter)
                try: new_data, new_targets = next(new_iter)
                except StopIteration: new_iter = iter(new_loader); new_data, new_targets = next(new_iter)
                
                retain_data, retain_targets = retain_data.to(self.device), retain_targets.to(self.device)
                forget_data, forget_targets = forget_data.to(self.device), forget_targets.to(self.device)
                new_data, new_targets = new_data.to(self.device), new_targets.to(self.device)
                
                retain_logits, _ = self.model(retain_data, retain_targets)
                forget_logits, _ = self.model(forget_data, forget_targets)
                new_logits, _ = self.model(new_data, new_targets)
                
                step_loss, loss_info = self.compute_total_loss(
                    retain_logits, retain_targets,
                    forget_logits, forget_targets, 
                    new_logits, new_targets
                )
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                for k, v in loss_info.items():
                    epoch_losses[k] += v
                num_batches += 1
            
            scheduler.step()
            avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
            epoch_metrics = self.validate_ranges(retain_classes, forget_classes, new_classes, verbose=False)
            current_overall_acc = epoch_metrics['overall']
            if current_overall_acc > best_overall_acc:
                best_overall_acc = current_overall_acc
                checkpoint_path = os.path.join(self.checkpoint_dir, f"step{step_idx}_best.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
            
            # Print metrics after each epoch
            print(f"Epoch {epoch+1}/{num_epochs} Metrics -> Forget: {epoch_metrics['forget']:.2f}% | Retain: {epoch_metrics['retain']:.2f}% | New: {epoch_metrics['new']:.2f}% | Overall: {epoch_metrics['overall']:.2f}%")
            
            epoch_pbar.set_postfix({
                'Loss': f"{avg_losses['total']:.3f}",
                'Overall': f"{current_overall_acc:.2f}%",
                'Best': f"{best_overall_acc:.2f}%"
            })
        
        final_metrics = self.validate_ranges(retain_classes, forget_classes, new_classes, verbose=True)
        return final_metrics

    # ===============================
    # SLIDING WINDOW EXPERIMENT
    # ===============================
    def run_sliding_window_experiment(self):
        print("🎯 Starting BID-LoRA Face Recognition Sliding Window Experiment")
        print("=" * 60)
        print(f"🔧 Configuration:")
        print(f"   Alpha (retention weight): {self.alpha}")
        print(f"   Gamma (acquisition weight): {self.gamma}")
        print(f"   BND (erasure limit): {self.erasure_limit}")
        print("=" * 60)
        
        pretrained_path = "/home/jag/codes/CLU/checkpoints/face/oracle/0_49.pth"
        if os.path.exists(pretrained_path):
            load_model_weights(self.model, pretrained_path)
            print(f"✅ Loaded pretrained checkpoint: {pretrained_path}")
        else:
            print(f"⚠️ Pretrained checkpoint not found: {pretrained_path}")
        
        steps = [
            {'retain': (10, 49), 'forget': (0, 9), 'new': (50, 59), 'checkpoint_to_load': pretrained_path},
            {'retain': (20, 59), 'forget': (0, 19), 'new': (60, 69), 'checkpoint_to_load': os.path.join(self.checkpoint_dir, "step1_best.pth")},
            {'retain': (30, 69), 'forget': (0, 29), 'new': (70, 79), 'checkpoint_to_load': os.path.join(self.checkpoint_dir, "step2_best.pth")},
            {'retain': (40, 79), 'forget': (0, 39), 'new': (80, 89), 'checkpoint_to_load': os.path.join(self.checkpoint_dir, "step3_best.pth")},
            {'retain': (50, 89), 'forget': (0, 49), 'new': (90, 99), 'checkpoint_to_load': os.path.join(self.checkpoint_dir, "step4_best.pth")}
        ]
        
        results = []
        for step_idx, step_config in enumerate(steps, 1):
            print(f"\n🔥 === STEP {step_idx}/5 ===")
            if step_idx > 1:
                checkpoint_path = step_config['checkpoint_to_load']
                if os.path.exists(checkpoint_path):
                    load_model_weights(self.model, checkpoint_path)
                    print(f"✅ Loaded checkpoint: {checkpoint_path}")
                else:
                    print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            metrics = self.train_step(
                retain_classes=step_config['retain'],
                forget_classes=step_config['forget'], 
                new_classes=step_config['new'],
                num_epochs=50,
                step_idx=step_idx
            )
            print(f"\n📋 STEP {step_idx} FINAL SUMMARY:")
            print(f"  🗑️ Forget ↓: {metrics['forget']:.2f}% (Classes {step_config['forget'][0]}-{step_config['forget'][1]})")
            print(f"  📚 Retain ↑: {metrics['retain']:.2f}% (Classes {step_config['retain'][0]}-{step_config['retain'][1]})") 
            print(f"  ✨ New ↑: {metrics['new']:.2f}% (Classes {step_config['new'][0]}-{step_config['new'][1]})")
            print(f"  🎯 Overall ↑: {metrics['overall']:.2f}%")
            results.append({'step': step_idx, 'config': step_config, 'metrics': metrics})
            checkpoint_path = os.path.join(self.checkpoint_dir, f"step{step_idx}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
        self._print_results_table(results)
        return results
    
    def _print_results_table(self, results):
        print("\n" + "="*80)
        print("📋 FINAL RESULTS SUMMARY - FACE RECOGNITION")
        print("="*80)
        print(f"{'Step':<6}{'Forget ↓':<10}{'Retain ↑':<10}{'New ↑':<10}{'Overall ↑':<12}{'Classes'}")
        print("-"*80)
        for result in results:
            step = result['step']
            metrics = result['metrics']
            config = result['config']
            forget_range = f"({config['forget'][0]}-{config['forget'][1]})"
            retain_range = f"({config['retain'][0]}-{config['retain'][1]})"  
            new_range = f"({config['new'][0]}-{config['new'][1]})"
            print(f"{step:<6}{metrics['forget']:<10.2f}{metrics['retain']:<10.2f}"
                  f"{metrics['new']:<10.2f}{metrics['overall']:<12.2f}"
                  f"F{forget_range} R{retain_range} N{new_range}")
        print("="*80)
        print("✅ BID-LoRA Face Recognition Sliding Window Experiment Complete!")


def main():
    Config.TaskName = "face"
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    trainer = BIDLoRAFaceTrainer(
        num_classes=100,
        lora_rank=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    results = trainer.run_sliding_window_experiment()
   
if __name__ == "__main__":
    results = main()
