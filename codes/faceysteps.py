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
import config as Config

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
        self.gamma = 1    # acquisition loss weight
        self.erasure_limit = 100  # BND parameter for bounded forgetting
        
        # Initialize model - get_model with lora_rank already handles freezing
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
        
    def _freeze_backbone(self):
        """Freeze backbone parameters, keep only LoRA adapters and classifier trainable"""
        for name, param in self.model.named_parameters():
            # Keep LoRA parameters and loss/classifier head trainable
            if 'lora' in name.lower() or 'loss' in name.lower() or 'head' in name.lower() or 'classifier' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        print("🔒 Frozen backbone parameters, keeping LoRA adapters and loss head trainable")
    
    def erasure_loss(self, logits, targets):
        """
        Erasure Loss from Eq. (10) in paper
        Uses ReLU bounded negative loss to forget unwanted knowledge
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='mean')
        bounded_loss = F.relu(-ce_loss + self.erasure_limit)
        return bounded_loss
    
    def retention_loss(self, logits, targets):
        """
        Knowledge Retention Loss from Eq. (11)
        Standard cross-entropy to maintain performance on retained classes
        """
        return F.cross_entropy(logits, targets, reduction='mean')
    
    def acquisition_loss(self, logits, targets):
        """
        Acquisition Loss from Eq. (12)  
        Standard cross-entropy to learn new classes
        """
        return F.cross_entropy(logits, targets, reduction='mean')
    
    def compute_total_loss(self, retain_logits, retain_targets, 
                          forget_logits, forget_targets,
                          new_logits, new_targets):
        """
        Total Loss from Eq. (13): L_total = L_retain + α*L_Erasure + β*L_new
        """
        # Retention loss
        l_retain = self.retention_loss(retain_logits, retain_targets)
        
        # Erasure loss (for forgetting)
        l_erasure = self.erasure_loss(forget_logits, forget_targets)
        
        # Acquisition loss (for new learning)
        l_acquisition = self.acquisition_loss(new_logits, new_targets)
        
        # Combined loss
        total_loss = l_retain + self.alpha * l_erasure + self.gamma * l_acquisition
        
        return total_loss, {
            'retention': l_retain.item(),
            'erasure': l_erasure.item(), 
            'acquisition': l_acquisition.item(),
            'total': total_loss.item()
        }
    
    @torch.no_grad()
    def _margin_free_logits_from_emb(self, embeddings):
        """
        Compute logits WITHOUT ArcFace margin for metrics (face recognition only):
          logits_eval = <normalize(emb)> · <normalize(W)>^T
        """
        emb_n = F.normalize(embeddings, dim=1)
        W = self.model.loss.weight  # [C, D]
        W_n = F.normalize(W, dim=1)
        logits_eval = F.linear(emb_n, W_n) * 64.0
        return logits_eval
    
    def validate_ranges(self, retain_classes, forget_classes, new_classes, verbose=True):
        """
        Validate model on all ranges and print detailed logs
        """
        self.model.eval()
        
        if verbose:
            print(f"\n📊 VALIDATION RESULTS:")
            print(f"=" * 60)
        
        metrics = {}
        
        with torch.no_grad():
            # Evaluate on each class range
            ranges = [
                ('Forget', forget_classes), 
                ('Retain', retain_classes), 
                ('New', new_classes)
            ]
            
            for name, class_range in ranges:
                if class_range[0] > class_range[1]:  # Skip invalid ranges
                    metrics[name.lower()] = 0.0
                    if verbose:
                        print(f"  {name} ({class_range[0]}-{class_range[1]}): INVALID RANGE")
                    continue
                    
                loader = get_dynamic_loader(
                    class_range=class_range, mode="test", batch_size=64,
                    data_percentage=1.0
                )
                
                correct = 0
                total = 0
                
                pbar_desc = f"Validating {name}" if verbose else None
                show_pbar = verbose
                
                for data, targets in tqdm(loader, desc=pbar_desc, leave=False, disable=not show_pbar):
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    # Face recognition forward pass
                    logits_train, embeddings = self.model(data, targets)
                    
                    # 🔑 Use margin-free logits for predictions (for fair evaluation)
                    logits_eval = self._margin_free_logits_from_emb(embeddings)
                    _, predicted = torch.max(logits_eval.data, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
                accuracy = 100 * correct / total if total > 0 else 0
                metrics[name.lower()] = accuracy
                
                if verbose:
                    # Print with appropriate arrow
                    arrow = "↓" if name == "Forget" else "↑"
                    print(f"  {name} {arrow} ({class_range[0]}-{class_range[1]}): {accuracy:.2f}% ({correct}/{total})")
        
        # Overall accuracy (retain + new classes)
        overall_start = min(retain_classes[0], new_classes[0])
        overall_end = max(retain_classes[1], new_classes[1])
        
        if overall_start <= overall_end:
            overall_loader = get_dynamic_loader(
                class_range=(overall_start, overall_end), mode="test", 
                batch_size=64, data_percentage=1.0
            )
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                pbar_desc = "Validating Overall" if verbose else None
                show_pbar = verbose
                
                for data, targets in tqdm(overall_loader, desc=pbar_desc, leave=False, disable=not show_pbar):
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    # Face recognition forward pass
                    logits_train, embeddings = self.model(data, targets)
                    
                    # 🔑 Use margin-free logits for predictions
                    logits_eval = self._margin_free_logits_from_emb(embeddings)
                    _, predicted = torch.max(logits_eval.data, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            overall_acc = 100 * correct / total if total > 0 else 0
            metrics['overall'] = overall_acc
            if verbose:
                print(f"  Overall ↑ ({overall_start}-{overall_end}): {overall_acc:.2f}% ({correct}/{total})")
        else:
            metrics['overall'] = 0.0
            if verbose:
                print(f"  Overall: INVALID RANGE")
        
        if verbose:
            print(f"=" * 60)
        
        return metrics

    def _imprint_new_class_weights(self, new_classes):
        """Initialize ArcFace weights for NEW classes using class-centroid (weight imprinting).
        This gives the classifier a strong starting point and typically boosts
        NEW accuracy from ~0–30% to much higher in the first few epochs.
        """
        self.model.eval()
        loader = get_dynamic_loader(class_range=new_classes, mode="train", batch_size=64, data_percentage=1.0)
        # collect normalized embeddings per class
        from collections import defaultdict
        sums = defaultdict(lambda: torch.zeros(self.model.loss.weight.shape[1], device=self.device))
        counts = defaultdict(int)
        with torch.no_grad():
            for data, targets in tqdm(loader, desc="Imprinting (new)", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                # forward to get embeddings; labels are required by ArcFace forward
                _, emb = self.model(data, targets)
                emb = F.normalize(emb, dim=1)
                for e, t in zip(emb, targets):
                    sums[int(t.item())] += e
                    counts[int(t.item())] += 1
        # write normalized centroids into the corresponding rows of ArcFace weight
        with torch.no_grad():
            for cls, cnt in counts.items():
                if cnt > 0:
                    w = F.normalize(sums[cls] / cnt, dim=0)
                    self.model.loss.weight[cls].copy_(w)
        self.model.train()
        print(f"✅ Imprinted weights for classes {new_classes[0]}-{new_classes[1]} (initialized from centroids)")

    def train_step(self, retain_classes, forget_classes, new_classes, num_epochs=50, step_idx=1):
        """Single training step for CLU adaptation with:
        - NEW-class head warmup (margin-based) to quickly learn W[new]
        - Higher LR for classifier head vs LoRA/backbone
        - Stronger acquisition weight (gamma)
        - ✅ NEW: weight imprinting for NEW classes before training
        """
        print(f"🔄 Training Step:")
        print(f"  📚 Retain classes: {retain_classes[0]}-{retain_classes[1]}")
        print(f"  🗑️ Forget classes: {forget_classes[0]}-{forget_classes[1]}")
        print(f"  ✨ New classes: {new_classes[0]}-{new_classes[1]}")

        # -------- Loaders --------
        retain_loader = get_dynamic_loader(class_range=retain_classes, mode="train", batch_size=32, data_percentage=0.1)
        forget_loader = get_dynamic_loader(class_range=forget_classes, mode="train", batch_size=32, data_percentage=0.5)
        new_loader    = get_dynamic_loader(class_range=new_classes,    mode="train", batch_size=32, data_percentage=1.0)

        # -------- Optimizer: higher LR for head (ArcFace) --------
        head_params = []
        other_params = []
        for n,p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if ('loss' in n.lower()) or ('head' in n.lower()) or ('classifier' in n.lower()):
                head_params.append(p)
            else:
                other_params.append(p)
        base_lr = 1e-4
        head_lr = 5e-4  # 5× LR for classifier head
        optimizer = optim.AdamW([
            { 'params': other_params, 'lr': base_lr, 'weight_decay': 1e-4 },
            { 'params': head_params,  'lr': head_lr,  'weight_decay': 1e-4 },
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # -------- Loss weights: strengthen acquisition, soften erase --------
        self.alpha = 0.05   # erasure weight (small)
        self.gamma = 2.0    # acquisition weight (stronger)

        # ✅ Imprint new-class weights before any training
        self._imprint_new_class_weights(new_classes)

        self.model.train()
        best_overall_acc = 0.0

        # -------- Warmup: train ONLY on NEW classes for a few epochs --------
        warmup_epochs = 5
        print(f"⚡ Head warmup on NEW classes for {warmup_epochs} epochs (margin ON, head_lr={head_lr})")
        for we in range(warmup_epochs):
            epoch_loss = 0.0
            for new_data, new_targets in tqdm(new_loader, desc=f"Warmup {we+1}/{warmup_epochs}", leave=False):
                new_data, new_targets = new_data.to(self.device), new_targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                new_logits_train, _ = self.model(new_data, new_targets)   # margin-based
                l_acq = F.cross_entropy(new_logits_train, new_targets)
                l_acq.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += l_acq.item()
            print(f"    Warmup {we+1}: acquisition loss {epoch_loss/max(1,len(new_loader)):.4f}")

        # -------- Main training --------
        from collections import defaultdict
        epoch_range = range(num_epochs)
        from tqdm import tqdm as _tqdm
        epoch_pbar = _tqdm(epoch_range, desc="Training Epochs", position=0)

        for epoch in epoch_pbar:
            epoch_losses = defaultdict(float)
            num_batches = 0

            retain_iter, forget_iter, new_iter = iter(retain_loader), iter(forget_loader), iter(new_loader)
            max_batches = max(len(retain_loader), len(forget_loader), len(new_loader))
            batch_pbar = _tqdm(range(max_batches), desc=f"Epoch {epoch+1}", leave=False, position=1)

            for _ in batch_pbar:
                optimizer.zero_grad(set_to_none=True)

                # fetch batches (cycle)
                try: retain_data, retain_targets = next(retain_iter)
                except StopIteration: retain_iter = iter(retain_loader); retain_data, retain_targets = next(retain_iter)
                try: forget_data, forget_targets = next(forget_iter)
                except StopIteration: forget_iter = iter(forget_loader); forget_data, forget_targets = next(forget_iter)
                try: new_data, new_targets = next(new_iter)
                except StopIteration: new_iter = iter(new_loader); new_data, new_targets = next(new_iter)

                retain_data, retain_targets = retain_data.to(self.device), retain_targets.to(self.device)
                forget_data, forget_targets = forget_data.to(self.device), forget_targets.to(self.device)
                new_data, new_targets       = new_data.to(self.device), new_targets.to(self.device)

                # forward (retain/forget with margin; new with margin)
                retain_logits, _ = self.model(retain_data, retain_targets)
                forget_logits, _ = self.model(forget_data, forget_targets)
                new_logits_train, _ = self.model(new_data, new_targets)

                # combine losses (retain + alpha*erase + gamma*acq)
                step_loss, loss_info = self.compute_total_loss(
                    retain_logits, retain_targets,
                    forget_logits, forget_targets,
                    new_logits_train, new_targets
                )

                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                for k, v in loss_info.items():
                    epoch_losses[k] += v
                num_batches += 1

            scheduler.step()
            avg_losses = {k: v/max(1,num_batches) for k, v in epoch_losses.items()}

            # validate (margin-free eval)
            epoch_metrics = self.validate_ranges(retain_classes, forget_classes, new_classes, verbose=False)
            current_overall_acc = epoch_metrics['overall']

            if current_overall_acc > best_overall_acc:
                best_overall_acc = current_overall_acc
                checkpoint_path = os.path.join(self.checkpoint_dir, f"step{step_idx}_best.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                improvement_str = f" ⭐ BEST! Saved: {checkpoint_path}"
            else:
                improvement_str = ""

            epoch_pbar.set_postfix({
                'Loss': f"{avg_losses.get('total',0.0):.3f}",
                'Overall': f"{current_overall_acc:.2f}%",
                'Best': f"{best_overall_acc:.2f}%",
                'Forget': f"{epoch_metrics['forget']:.1f}%",
                'Retain': f"{epoch_metrics['retain']:.1f}%",
                'New': f"{epoch_metrics['new']:.1f}%"
            })

            print(f" 📈 Epoch {epoch+1}/{num_epochs}:")
            print(f"      Loss: {avg_losses.get('total',0.0):.4f} | LR_head: {scheduler.get_last_lr()[0]:.6f}")
            print(f"      Forget ↓: {epoch_metrics['forget']:.2f}% | Retain ↑: {epoch_metrics['retain']:.2f}%")
            print(f"      New ↑: {epoch_metrics['new']:.2f}% | Overall ↑: {current_overall_acc:.2f}%{improvement_str}")

        print(f" ✅ Training completed! Best Overall: {best_overall_acc:.2f}%")
        final_metrics = self.validate_ranges(retain_classes, forget_classes, new_classes, verbose=True)
        return final_metrics

    def train_step(self, retain_classes, forget_classes, new_classes, num_epochs=50, step_idx=1):
        """Single training step for CLU adaptation with:
        - NEW-class head warmup (margin-based) to quickly learn W[new]
        - Higher LR for classifier head vs LoRA/backbone
        - Stronger acquisition weight (gamma)
        - ✅ NEW: weight imprinting for NEW classes before training
        """
        print(f"\n🔄 Training Step:")
        print(f"  📚 Retain classes: {retain_classes[0]}-{retain_classes[1]}")
        print(f"  🗑️ Forget classes: {forget_classes[0]}-{forget_classes[1]}")
        print(f"  ✨ New classes: {new_classes[0]}-{new_classes[1]}")

        # -------- Loaders --------
        retain_loader = get_dynamic_loader(class_range=retain_classes, mode="train", batch_size=32, data_percentage=0.1)
        forget_loader = get_dynamic_loader(class_range=forget_classes, mode="train", batch_size=32, data_percentage=0.5)
        new_loader    = get_dynamic_loader(class_range=new_classes,    mode="train", batch_size=32, data_percentage=1.0)

        # -------- Optimizer: higher LR for head (ArcFace) --------
        head_params = []
        other_params = []
        for n,p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if ('loss' in n.lower()) or ('head' in n.lower()) or ('classifier' in n.lower()):
                head_params.append(p)
            else:
                other_params.append(p)
        base_lr = 1e-4
        head_lr = 5e-4  # 5× LR for classifier head
        optimizer = optim.AdamW([
            { 'params': other_params, 'lr': base_lr, 'weight_decay': 1e-4 },
            { 'params': head_params,  'lr': head_lr,  'weight_decay': 1e-4 },
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # -------- Loss weights: strengthen acquisition, soften erase --------
        self.alpha = 0.05   # erasure weight (small)
        self.gamma = 2.0    # acquisition weight (stronger)

        # ✅ Imprint new-class weights before any training
        self._imprint_new_class_weights(new_classes)

        self.model.train()
        best_overall_acc = 0.0

        # -------- Warmup: train ONLY on NEW classes for a few epochs --------
        warmup_epochs = 5
        print(f"⚡ Head warmup on NEW classes for {warmup_epochs} epochs (margin ON, head_lr={head_lr})")
        for we in range(warmup_epochs):
            epoch_loss = 0.0
            for new_data, new_targets in tqdm(new_loader, desc=f"Warmup {we+1}/{warmup_epochs}", leave=False):
                new_data, new_targets = new_data.to(self.device), new_targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                new_logits_train, _ = self.model(new_data, new_targets)   # margin-based
                l_acq = F.cross_entropy(new_logits_train, new_targets)
                l_acq.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += l_acq.item()
            print(f"    Warmup {we+1}: acquisition loss {epoch_loss/max(1,len(new_loader)):.4f}")

        # -------- Main training --------
        from collections import defaultdict
        epoch_range = range(num_epochs)
        from tqdm import tqdm as _tqdm
        epoch_pbar = _tqdm(epoch_range, desc="Training Epochs", position=0)

        for epoch in epoch_pbar:
            epoch_losses = defaultdict(float)
            num_batches = 0

            retain_iter, forget_iter, new_iter = iter(retain_loader), iter(forget_loader), iter(new_loader)
            max_batches = max(len(retain_loader), len(forget_loader), len(new_loader))
            batch_pbar = _tqdm(range(max_batches), desc=f"Epoch {epoch+1}", leave=False, position=1)

            for _ in batch_pbar:
                optimizer.zero_grad(set_to_none=True)

                # fetch batches (cycle)
                try: retain_data, retain_targets = next(retain_iter)
                except StopIteration: retain_iter = iter(retain_loader); retain_data, retain_targets = next(retain_iter)
                try: forget_data, forget_targets = next(forget_iter)
                except StopIteration: forget_iter = iter(forget_loader); forget_data, forget_targets = next(forget_iter)
                try: new_data, new_targets = next(new_iter)
                except StopIteration: new_iter = iter(new_loader); new_data, new_targets = next(new_iter)

                retain_data, retain_targets = retain_data.to(self.device), retain_targets.to(self.device)
                forget_data, forget_targets = forget_data.to(self.device), forget_targets.to(self.device)
                new_data, new_targets       = new_data.to(self.device), new_targets.to(self.device)

                # forward (retain/forget with margin; new with margin)
                retain_logits, _ = self.model(retain_data, retain_targets)
                forget_logits, _ = self.model(forget_data, forget_targets)
                new_logits_train, _ = self.model(new_data, new_targets)

                # combine losses (retain + alpha*erase + gamma*acq)
                step_loss, loss_info = self.compute_total_loss(
                    retain_logits, retain_targets,
                    forget_logits, forget_targets,
                    new_logits_train, new_targets
                )

                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                for k, v in loss_info.items():
                    epoch_losses[k] += v
                num_batches += 1

            scheduler.step()
            avg_losses = {k: v/max(1,num_batches) for k, v in epoch_losses.items()}

            # validate (margin-free eval)
            epoch_metrics = self.validate_ranges(retain_classes, forget_classes, new_classes, verbose=False)
            current_overall_acc = epoch_metrics['overall']

            if current_overall_acc > best_overall_acc:
                best_overall_acc = current_overall_acc
                checkpoint_path = os.path.join(self.checkpoint_dir, f"step{step_idx}_best.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                improvement_str = f" ⭐ BEST! Saved: {checkpoint_path}"
            else:
                improvement_str = ""

            epoch_pbar.set_postfix({
                'Loss': f"{avg_losses.get('total',0.0):.3f}",
                'Overall': f"{current_overall_acc:.2f}%",
                'Best': f"{best_overall_acc:.2f}%",
                'Forget': f"{epoch_metrics['forget']:.1f}%",
                'Retain': f"{epoch_metrics['retain']:.1f}%",
                'New': f"{epoch_metrics['new']:.1f}%"
            })

            print(f"\n    📈 Epoch {epoch+1}/{num_epochs}:")
            print(f"      Loss: {avg_losses.get('total',0.0):.4f} | LR_head: {scheduler.get_last_lr()[0]:.6f}")
            print(f"      Forget ↓: {epoch_metrics['forget']:.2f}% | Retain ↑: {epoch_metrics['retain']:.2f}%")
            print(f"      New ↑: {epoch_metrics['new']:.2f}% | Overall ↑: {current_overall_acc:.2f}%{improvement_str}")

        print(f"\n✅ Training completed! Best Overall: {best_overall_acc:.2f}%")
        final_metrics = self.validate_ranges(retain_classes, forget_classes, new_classes, verbose=True)
        return final_metrics

    
    def run_sliding_window_experiment(self):
        """
        Run the 5-step sliding window experiment as described in the paper
        Progressively transitions from classes 0-49 to 50-99
        """
        print("🎯 Starting BID-LoRA Face Recognition Sliding Window Experiment")
        print("=" * 60)
        print(f"🔧 Configuration:")
        print(f"   Alpha (retention weight): {self.alpha}")
        print(f"   Gamma (acquisition weight): {self.gamma}")
        print(f"   BND (erasure limit): {self.erasure_limit}")
        print("=" * 60)
        
        # Load pretrained checkpoint
        pretrained_path = "/home/jag/codes/CLU/checkpoints/face/oracle/0_49.pth"
        if os.path.exists(pretrained_path):
            load_model_weights(self.model, pretrained_path)
            print(f"✅ Loaded pretrained checkpoint: {pretrained_path}")
        else:
            print(f"⚠️ Pretrained checkpoint not found: {pretrained_path}")
            print("    Continuing with timm pretrained weights...")
        
        # Define the 5-step sliding window protocol from paper
        steps = [
            # Step 1: (0-9 forget, 10-49 retain, 50-59 new)
            {
                'retain': (10, 49),
                'forget': (0, 9), 
                'new': (50, 59),
                'checkpoint_to_load': pretrained_path
            },
            # Step 2: (0-19 forget, 20-59 retain, 60-69 new)
            {
                'retain': (20, 59),
                'forget': (0, 19),
                'new': (60, 69),
                'checkpoint_to_load': os.path.join(self.checkpoint_dir, "step1.pth")
            },
            # Step 3: (0-29 forget, 30-69 retain, 70-79 new)
            {
                'retain': (30, 69),
                'forget': (0, 29),
                'new': (70, 79),
                'checkpoint_to_load': os.path.join(self.checkpoint_dir, "step2.pth")
            },
            # Step 4: (0-39 forget, 40-79 retain, 80-89 new)  
            {
                'retain': (40, 79),
                'forget': (0, 39),
                'new': (80, 89),
                'checkpoint_to_load': os.path.join(self.checkpoint_dir, "step3.pth")
            },
            # Step 5: (0-49 forget, 50-89 retain, 90-99 new)
            {
                'retain': (50, 89), 
                'forget': (0, 49),
                'new': (90, 99),
                'checkpoint_to_load': os.path.join(self.checkpoint_dir, "step4.pth")
            }
        ]
        
        results = []
        
        for step_idx, step_config in enumerate(steps, 1):
            print(f"\n🔥 === STEP {step_idx}/5 ===")
            
            # Load checkpoint from previous step (except step 1 which uses pretrained)
            if step_idx > 1:
                checkpoint_path = step_config['checkpoint_to_load']
                if os.path.exists(checkpoint_path):
                    load_model_weights(self.model, checkpoint_path)
                    print(f"✅ Loaded checkpoint: {checkpoint_path}")
                else:
                    print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            
            # Training phase with validation
            metrics = self.train_step(
                retain_classes=step_config['retain'],
                forget_classes=step_config['forget'], 
                new_classes=step_config['new'],
                num_epochs=50,
                step_idx=step_idx
            )
            
            # Print step summary
            print(f"\n📋 STEP {step_idx} FINAL SUMMARY:")
            print(f"  🗑️ Forget ↓: {metrics['forget']:.2f}% "
                  f"(Classes {step_config['forget'][0]}-{step_config['forget'][1]})")
            print(f"  📚 Retain ↑: {metrics['retain']:.2f}% "
                  f"(Classes {step_config['retain'][0]}-{step_config['retain'][1]})") 
            print(f"  ✨ New ↑: {metrics['new']:.2f}% "
                  f"(Classes {step_config['new'][0]}-{step_config['new'][1]})")
            print(f"  🎯 Overall ↑: {metrics['overall']:.2f}%")
            
            results.append({
                'step': step_idx,
                'config': step_config,
                'metrics': metrics
            })
            
            # Save checkpoint for next step
            checkpoint_path = os.path.join(self.checkpoint_dir, f"step{step_idx}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Print final summary table
        self._print_results_table(results)
        
        return results
    
    def _print_results_table(self, results):
        """Print results in paper table format"""
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
    """Main training function"""
    # Set task for face recognition
    Config.TaskName = "face"
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize trainer
    trainer = BIDLoRAFaceTrainer(
        num_classes=100,
        lora_rank=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run the sliding window experiment
    results = trainer.run_sliding_window_experiment()
    
    return results

if __name__ == "__main__":
    results = main()