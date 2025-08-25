import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import logging
import argparse
from collections import defaultdict

# Project imports
from data import get_dynamic_loader
from utils import get_model, print_parameter_stats
from config import Config

# ---------------- Enhanced Settings ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_IDS = [0] if DEVICE.type == "cuda" else None

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 200
BASE_LR = 1e-3
WEIGHT_DECAY = 5e-5

FORGET_EPOCHS = 120  # Increased for better convergence
CHECKPOINTS_STEPS_DIR = "checkpoints/steps"
CHECKPOINTS_ORACLE_DIR = "checkpoints/oracle"

# Enhanced GS-LoRA parameters
ALPHA_RETAIN = 2.0    # Increased importance
ALPHA_FORGET = 0.15   # Reduced interference
ALPHA_NEW = 2.0       # Increased importance
BND = 150.0
BETA = 0.08           # Reduced forget interference
ALPHA_K = 0.003       # Reduced sparsity pressure
K_WARMUP = 10         # Increased warmup
LORA_RANK = 16        # Doubled capacity

# Learning rate schedule parameters
RETAIN_LR_MULT = 0.5   # Conservative for retention
NEW_LR_MULT = 1.5      # Aggressive for new learning
LORA_LR_MULT = 2.0     # Very aggressive for LoRA

# Training phases
PHASE1_EPOCHS = 30     # Retention focus
PHASE2_EPOCHS = 60     # Balanced
PHASE3_EPOCHS = 30     # Fine-tuning

# ---------------- Logging ----------------
def setup_logging():
    """Setup logging for the enhanced CLU training process"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/clu_training"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"enhanced_clu_training_{timestamp}.log")

    logger = logging.getLogger("Enhanced_CLU_Training")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler for important messages
    ch = logging.StreamHandler()
    # ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(ch)
    
    return logger

# ---------------- Enhanced Early Stopping ----------------
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.5, restore_best_weights=True, metric='combined'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.metric = metric
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.history = []
        
    def __call__(self, metrics_dict, model):
        if self.metric == 'combined':
            score = metrics_dict['combined_acc']
        elif self.metric == 'balanced':
            # Balanced score considering both retain and new
            score = (metrics_dict['retain_acc'] + metrics_dict['new_acc']) / 2.0
        else:
            score = metrics_dict.get(self.metric, 0)
            
        self.history.append(score)
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}

# ---------------- Enhanced LoRA Group Management ----------------
def get_enhanced_lora_groups(model):
    """
    Enhanced LoRA parameter grouping with layer-wise organization
    """
    groups = []
    layer_groups = defaultdict(list)
    
    # Group by layer type and index
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            layer_info = name.split('.')
            layer_key = '.'.join(layer_info[:-1]) if len(layer_info) > 1 else name
            
            group_params = []
            if hasattr(module, 'lora_A') and module.lora_A is not None:
                group_params.append(module.lora_A)
            if hasattr(module, 'lora_B') and module.lora_B is not None:
                group_params.append(module.lora_B)
            
            if group_params:
                layer_groups[layer_key].extend(group_params)
    
    # Convert to list of groups
    for layer_key, params in layer_groups.items():
        if params:
            groups.append(params)
    
    return groups

def compute_enhanced_group_sparse_loss(lora_groups):
    """
    Enhanced group sparse loss with layer-wise normalization
    """
    if not lora_groups:
        return torch.tensor(0.0, device=DEVICE)
    
    group_losses = []
    for group in lora_groups:
        group_loss = 0.0
        param_count = 0
        for param in group:
            if param is not None:
                # Frobenius norm with size normalization
                norm = torch.norm(param, p='fro')
                # Normalize by parameter size for fair comparison
                norm = norm / (param.numel() ** 0.5)
                group_loss += norm
                param_count += 1
        
        if param_count > 0:
            group_loss = group_loss / param_count  # Average over parameters in group
            group_losses.append(group_loss)
    
    return sum(group_losses) / len(group_losses) if group_losses else torch.tensor(0.0, device=DEVICE)

def compute_zero_group_ratio(lora_groups, threshold=1e-5):
    """
    Compute ratio of effectively zero groups with adaptive threshold
    """
    if not lora_groups:
        return 0.0
    
    zero_groups = 0
    total_groups = len(lora_groups)
    
    for group in lora_groups:
        group_norm = 0.0
        param_count = 0
        for param in group:
            if param is not None:
                norm = torch.norm(param, p='fro').item()
                norm = norm / (param.numel() ** 0.5)  # Normalized
                group_norm += norm
                param_count += 1
        
        if param_count > 0:
            avg_norm = group_norm / param_count
            if avg_norm < threshold:
                zero_groups += 1
    
    return zero_groups / max(total_groups, 1)

# ---------------- Enhanced Model Builder ----------------
def build_enhanced_model(num_classes=100, lora_rank=16):
    """Build model with enhanced LoRA configuration"""
    Config.TaskName = "face_recognition"
    
    model = get_model(
        config=Config,
        num_classes=num_classes,
        lora_rank=lora_rank,
        pretrained=True,
        drop_rate=0.0,
        device=DEVICE
    )
    
    print_parameter_stats(model)
    
    # Initialize LoRA parameters with better initialization
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and module.lora_A is not None:
            # Xavier uniform for LoRA_A (Parameter objects don't have .weight)
            with torch.no_grad():
                nn.init.xavier_uniform_(module.lora_A)
        if hasattr(module, 'lora_B') and module.lora_B is not None:
            # Zero initialization for LoRA_B (standard practice)
            with torch.no_grad():
                nn.init.zeros_(module.lora_B)
    
    return model

# ---------------- Enhanced Parameter Groups ----------------
def get_enhanced_parameter_groups(model):
    """
    Create parameter groups with different learning rates for different components
    """
    retain_params = []
    new_params = []
    lora_params = []
    backbone_params = []
    
    # Identify parameter types
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        elif 'loss.weight' in name:  # Classification head
            # Check if this corresponds to retain or new classes
            # For now, treat as shared
            backbone_params.append(param)
        else:
            backbone_params.append(param)
    
    param_groups = [
        {
            'params': lora_params,
            'lr': BASE_LR * LORA_LR_MULT,
            'weight_decay': WEIGHT_DECAY * 0.1,  # Reduced weight decay for LoRA
            'name': 'lora'
        },
        {
            'params': backbone_params,
            'lr': BASE_LR,
            'weight_decay': WEIGHT_DECAY,
            'name': 'backbone'
        }
    ]
    
    return param_groups

# ---------------- Enhanced Optimizer and Scheduler ----------------
def get_enhanced_optimizer_and_scheduler(model, total_epochs):
    """Create enhanced optimizer with parameter-specific learning rates"""
    
    param_groups = get_enhanced_parameter_groups(model)
    
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Enhanced scheduler with different phases
    def lambda_fn(epoch):
        if epoch < PHASE1_EPOCHS:
            # Phase 1: Retention focus - conservative learning
            return 1.0
        elif epoch < PHASE1_EPOCHS + PHASE2_EPOCHS:
            # Phase 2: Balanced - normal learning
            progress = (epoch - PHASE1_EPOCHS) / PHASE2_EPOCHS
            return 1.0 - 0.3 * progress  # Gradual decay
        else:
            # Phase 3: Fine-tuning - reduced learning
            return 0.3
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
    
    return optimizer, scheduler

# ---------------- Enhanced Margin-free Logits ----------------
@torch.no_grad()
def _enhanced_margin_free_logits_from_emb(model, emb, temperature=1.0):
    """Compute logits WITHOUT ArcFace margin with temperature scaling"""
    emb_n = F.normalize(emb, dim=1)
    W = model.loss.weight
    W_n = F.normalize(W, dim=1)
    logits_eval = F.linear(emb_n, W_n) * 64.0 / temperature
    return logits_eval

# ---------------- Enhanced Evaluation ----------------
@torch.no_grad()
def enhanced_evaluate(model, loader, temperature=1.0):
    """Enhanced evaluation with calibrated predictions"""
    model.eval()
    total_loss, all_preds, all_labels, all_confidences = 0.0, [], [], []
    all_logits, all_embeddings = [], []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).long()

        # Get logits and embeddings
        logits_train, emb = model(images, labels)
        loss = F.cross_entropy(logits_train, labels)
        total_loss += loss.item()

        # Temperature-scaled logits for better calibration
        logits_eval = _enhanced_margin_free_logits_from_emb(model, emb, temperature)
        probs = F.softmax(logits_eval, dim=1)
        preds = torch.argmax(logits_eval, dim=1)
        confidences = torch.max(probs, dim=1)[0]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())
        all_logits.append(logits_eval.cpu())
        all_embeddings.append(emb.cpu())

    # Calculate comprehensive metrics
    all_preds_t = torch.from_numpy(np.array(all_preds))
    all_labels_t = torch.from_numpy(np.array(all_labels))

    avg_loss = total_loss / max(1, len(loader))
    acc = (all_preds_t == all_labels_t).float().mean().item() * 100.0
    avg_confidence = np.mean(all_confidences) * 100.0
    
    # Additional metrics
    precision = precision_score(all_labels_t, all_preds_t, average="macro", zero_division=0) * 100.0
    recall = recall_score(all_labels_t, all_preds_t, average="macro", zero_division=0) * 100.0
    f1 = f1_score(all_labels_t, all_preds_t, average="macro", zero_division=0) * 100.0
    
    # Embedding quality metrics
    all_embeddings = torch.cat(all_embeddings, dim=0)
    emb_mean_norm = torch.norm(all_embeddings, dim=1).mean().item()
    emb_std_norm = torch.norm(all_embeddings, dim=1).std().item()

    return {
        'loss': avg_loss,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confidence': avg_confidence,
        'emb_mean_norm': emb_mean_norm,
        'emb_std_norm': emb_std_norm
    }

# ---------------- Enhanced Training Phase Management ----------------
def get_phase_weights(epoch):
    """Get training phase weights for different objectives"""
    if epoch <= PHASE1_EPOCHS:
        # Phase 1: Retention focus
        return {
            'retain': 3.0,
            'new': 0.5,
            'forget': 0.8
        }
    elif epoch <= PHASE1_EPOCHS + PHASE2_EPOCHS:
        # Phase 2: Balanced
        progress = (epoch - PHASE1_EPOCHS) / PHASE2_EPOCHS
        retain_weight = 3.0 - 1.0 * progress  # 3.0 -> 2.0
        new_weight = 0.5 + 1.5 * progress     # 0.5 -> 2.0
        return {
            'retain': retain_weight,
            'new': new_weight,
            'forget': 0.8
        }
    else:
        # Phase 3: Fine-tuning
        return {
            'retain': 1.8,
            'new': 1.8,
            'forget': 0.6
        }

# ---------------- Enhanced Loss Computation ----------------
def compute_enhanced_balanced_loss(model, rx, ry, fx, fy, nx, ny, lora_groups, epoch, alpha_current):
    """Enhanced loss computation with phase-aware balancing"""
    
    phase_weights = get_phase_weights(epoch)
    
    # Enhanced retention loss with consistency regularization
    logits_retain, emb_retain = model(rx, ry)
    loss_retain = F.cross_entropy(logits_retain, ry)
    
    # Confidence regularization for retained classes
    retain_probs = F.softmax(logits_retain, dim=1)
    retain_confidence = torch.max(retain_probs, dim=1)[0]
    confidence_reg = -torch.log(retain_confidence + 1e-8).mean()
    
    # Embedding norm regularization for stability
    emb_norm_reg = torch.norm(emb_retain, dim=1).var()
    
    loss_retain = loss_retain + 0.1 * confidence_reg + 0.01 * emb_norm_reg
    
    # Enhanced new class learning with progressive difficulty
    logits_new, emb_new = model(nx, ny)
    loss_new = F.cross_entropy(logits_new, ny, reduction='none')
    
    # Progressive curriculum learning
    new_probs = F.softmax(logits_new, dim=1)
    new_entropy = -torch.sum(new_probs * torch.log(new_probs + 1e-8), dim=1)
    
    # Adaptive curriculum based on epoch
    curriculum_factor = min(1.0, epoch / (PHASE1_EPOCHS + PHASE2_EPOCHS))
    curriculum_weight = torch.sigmoid(curriculum_factor * (3.0 - new_entropy))
    loss_new = (loss_new * curriculum_weight).mean()
    
    # Enhanced embedding consistency for new classes
    new_emb_norm_reg = torch.norm(emb_new, dim=1).var()
    loss_new = loss_new + 0.01 * new_emb_norm_reg
    
    # Adaptive forgetting loss with phase awareness
    logits_forget, emb_forget = model(fx, fy)
    ce_forget = F.cross_entropy(logits_forget, fy, reduction='none')
    
    # Phase-aware adaptive bound
    phase_factor = 1.0
    if epoch <= PHASE1_EPOCHS:
        phase_factor = 0.8  # More aggressive forgetting in retention phase
    elif epoch <= PHASE1_EPOCHS + PHASE2_EPOCHS:
        phase_factor = 1.0  # Normal forgetting in balanced phase
    else:
        phase_factor = 1.2  # Gentler forgetting in fine-tuning phase
    
    adaptive_bnd = BND * phase_factor
    loss_forget = F.relu(adaptive_bnd - ce_forget).mean()
    
    # Enhanced group sparse loss with phase awareness
    loss_structure = compute_enhanced_group_sparse_loss(lora_groups)
    
    # Phase-aware loss balancing
    loss_data = (
        ALPHA_RETAIN * phase_weights['retain'] * loss_retain +
        BETA * phase_weights['forget'] * loss_forget +
        ALPHA_NEW * phase_weights['new'] * loss_new
    )
    
    # Adaptive sparsity loss
    sparsity_weight = alpha_current
    if epoch > PHASE1_EPOCHS + PHASE2_EPOCHS:
        sparsity_weight *= 1.5  # Increase sparsity in fine-tuning
    
    total_loss = loss_data + sparsity_weight * loss_structure
    
    return {
        'total': total_loss,
        'retain': loss_retain,
        'forget': loss_forget,
        'new': loss_new,
        'structure': loss_structure,
        'data': loss_data
    }

# ---------------- Enhanced Training Step ----------------
def enhanced_training_step(model, optimizer, scheduler, retain_loader, forget_loader, new_loader, lora_groups, epoch):
    """Enhanced training step with improved balancing and optimization"""
    model.train()
    
    # Progressive alpha scheduling with enhanced warm-up
    if epoch < K_WARMUP:
        alpha_current = 0.0
    else:
        # Smooth progressive sparsity
        progress = (epoch - K_WARMUP) / (FORGET_EPOCHS - K_WARMUP)
        alpha_current = ALPHA_K * (1.0 - np.exp(-3.0 * progress))  # Exponential growth
    
    # Enhanced data loading with smart batching
    retain_iter = iter(retain_loader)
    forget_iter = iter(forget_loader)
    new_iter = iter(new_loader)
    
    max_batches = max(len(retain_loader), len(forget_loader), len(new_loader))
    
    epoch_metrics = {
        'total_loss': 0.0,
        'retain_loss': 0.0,
        'forget_loss': 0.0,
        'new_loss': 0.0,
        'sparse_loss': 0.0,
        'data_loss': 0.0
    }
    
    pbar = tqdm(range(max_batches), desc=f"Training Epoch {epoch}", leave=False)
    
    for batch_idx in pbar:
        # Cycle through loaders with error handling
        try:
            rx, ry = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            rx, ry = next(retain_iter)
        
        try:
            fx, fy = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            fx, fy = next(forget_iter)
        
        try:
            nx, ny = next(new_iter)
        except StopIteration:
            new_iter = iter(new_loader)
            nx, ny = next(new_iter)
        
        # Move to device
        rx, ry = rx.to(DEVICE), ry.to(DEVICE).long()
        fx, fy = fx.to(DEVICE), fy.to(DEVICE).long()
        nx, ny = nx.to(DEVICE), ny.to(DEVICE).long()
        
        # Forward pass with enhanced loss
        losses = compute_enhanced_balanced_loss(
            model, rx, ry, fx, fy, nx, ny, lora_groups, epoch, alpha_current
        )
        
        total_loss = losses['total']
        
        # Backward pass with enhanced gradient handling
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping with adaptive norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        # Check for gradient explosion
        # if grad_norm > 10.0:
            # print(f"Warning: Large gradient norm {grad_norm:.2f} at epoch {epoch}, batch {batch_idx}")
        
        optimizer.step()
        
        # Update metrics
        for key, value in losses.items():
            if key in epoch_metrics:
                if key == 'total':
                    epoch_metrics['total_loss'] += value.item()
                elif key == 'retain':
                    epoch_metrics['retain_loss'] += value.item()
                elif key == 'forget':
                    epoch_metrics['forget_loss'] += value.item()
                elif key == 'new':
                    epoch_metrics['new_loss'] += value.item()
                elif key == 'structure':
                    epoch_metrics['sparse_loss'] += value.item()
                elif key == 'data':
                    epoch_metrics['data_loss'] += value.item()
        
        # Update progress bar with phase info
        phase = "Retention" if epoch <= PHASE1_EPOCHS else ("Balanced" if epoch <= PHASE1_EPOCHS + PHASE2_EPOCHS else "Fine-tuning")
        phase_weights = get_phase_weights(epoch)
        
        pbar.set_postfix({
            'Phase': phase,
            'Total': f'{total_loss.item():.3f}',
            'R': f'{losses["retain"].item():.2f}',
            'N': f'{losses["new"].item():.2f}',
            'F': f'{losses["forget"].item():.2f}',
            'S': f'{losses["structure"].item():.3f}',
            'α': f'{alpha_current:.4f}',
            'RW': f'{phase_weights["retain"]:.1f}',
            'NW': f'{phase_weights["new"]:.1f}'
        })
    
    # Step the scheduler
    scheduler.step()
    
    # Average metrics
    for key in epoch_metrics:
        epoch_metrics[key] /= max_batches
    
    return epoch_metrics, alpha_current

# ---------------- Enhanced GS-LoRA Training Step ----------------
def train_single_step_enhanced_gs_lora(
    step_num,
    input_checkpoint,
    retain_classes,
    forget_classes,
    new_classes,
    output_checkpoint,
    logger=None,
):
    """
    Enhanced GS-LoRA training step with improved balance and optimization
    """
    print("\n" + "=" * 80)
    print(f"ENHANCED GS-LoRA STEP {step_num}")
    print(f"Retain Classes: {retain_classes[0]}-{retain_classes[-1]} ({len(retain_classes)} classes)")
    print(f"Forget Classes: {forget_classes[0]}-{forget_classes[-1]} ({len(forget_classes)} classes)")
    print(f"New Classes:    {new_classes[0]}-{new_classes[-1]} ({len(new_classes)} classes)")
    print(f"LoRA Rank: {LORA_RANK}, Epochs: {FORGET_EPOCHS}")
    print("=" * 80)

    # Build enhanced model
    model = build_enhanced_model(num_classes=100, lora_rank=LORA_RANK)

    # Load checkpoint with better error handling
    if not os.path.exists(input_checkpoint):
        print(f"❌ Input checkpoint not found: {input_checkpoint}")
        return False

    try:
        state_dict = torch.load(input_checkpoint, map_location=DEVICE)
        
        # Smart state dict loading
        model_state = model.state_dict()
        compatible_state = {}
        
        for k, v in state_dict.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    compatible_state[k] = v
                else:
                    print(f"Shape mismatch for {k}: model {model_state[k].shape} vs checkpoint {v.shape}")
            else:
                print(f"Key {k} not found in model")
        
        missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} (this is normal for LoRA parameters)")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
        print(f"✓ Loaded model from: {input_checkpoint}")
        print(f"  Compatible parameters: {len(compatible_state)}/{len(state_dict)}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

    # Get enhanced LoRA groups
    lora_groups = get_enhanced_lora_groups(model)
    print(f"✓ Found {len(lora_groups)} LoRA groups for enhanced sparse regularization")

    # Enhanced dataloaders with better configuration
    pin_mem = (DEVICE.type == "cuda")
    loader_kwargs = {
        'batch_size': BATCH_SIZE,
        'image_size': IMAGE_SIZE,
        'num_workers': 4,
        'pin_memory': pin_mem
    }
    
    retain_loader = get_dynamic_loader(
        class_range=(retain_classes[0], retain_classes[-1]),
        mode="train", data_percentage=0.15, **loader_kwargs  # Increased data for retention
    )
    forget_loader = get_dynamic_loader(
        class_range=(forget_classes[0], forget_classes[-1]),
        mode="train", **loader_kwargs
    )
    new_loader = get_dynamic_loader(
        class_range=(new_classes[0], new_classes[-1]),
        mode="train", data_percentage=1.0, **loader_kwargs  # Full data for new classes
    )

    print(f"✓ Data loaders created:")
    print(f"  Retain: {len(retain_loader)} batches")
    print(f"  Forget: {len(forget_loader)} batches")
    print(f"  New:    {len(new_loader)} batches")

    # Enhanced optimizer and scheduler
    optimizer, scheduler = get_enhanced_optimizer_and_scheduler(model, FORGET_EPOCHS)
    print(f"✓ Enhanced optimizer created with {len(optimizer.param_groups)} parameter groups")

    # Enhanced early stopping
    early_stopping = EarlyStopping(patience=25, min_delta=0.3, metric='balanced')

    # Training tracking
    best_metrics = {
        'combined_acc': 0.0,
        'balanced_acc': 0.0,
        'zero_ratio': 0.0,
        'epoch': 0
    }
    
    training_history = []

    print(f"\n🚀 Starting Enhanced Training for {FORGET_EPOCHS} epochs")
    print(f"Phase Schedule: Retention({PHASE1_EPOCHS}) -> Balanced({PHASE2_EPOCHS}) -> Fine-tuning({PHASE3_EPOCHS})")

    for epoch in range(1, FORGET_EPOCHS + 1):
        # Enhanced training step
        epoch_metrics, alpha_current = enhanced_training_step(
            model, optimizer, scheduler, retain_loader, forget_loader, new_loader, lora_groups, epoch
        )

        # Compute sparsity metrics
        zero_ratio = compute_zero_group_ratio(lora_groups)

        # Enhanced validation
        val_loader_kwargs = dict(loader_kwargs)
        val_loader_kwargs['batch_size'] = 16  # Smaller batch for validation
        
        retain_val_loader = get_dynamic_loader(
            class_range=(retain_classes[0], retain_classes[-1]),
            mode="val", **val_loader_kwargs
        )
        forget_val_loader = get_dynamic_loader(
            class_range=(forget_classes[0], forget_classes[-1]),
            mode="val", **val_loader_kwargs
        )
        new_val_loader = get_dynamic_loader(
            class_range=(new_classes[0], new_classes[-1]),
            mode="val", **val_loader_kwargs
        )
        combined_val_loader = get_dynamic_loader(
            class_range=(retain_classes[0], new_classes[-1]),
            mode="val", **val_loader_kwargs
        )

        # Enhanced evaluation with temperature scaling
        retain_metrics = enhanced_evaluate(model, retain_val_loader, temperature=1.1)
        forget_metrics = enhanced_evaluate(model, forget_val_loader, temperature=1.1)
        new_metrics = enhanced_evaluate(model, new_val_loader, temperature=1.1)
        combined_metrics = enhanced_evaluate(model, combined_val_loader, temperature=1.1)

        # Calculate enhanced metrics
        retain_acc = retain_metrics['acc']
        forget_acc = forget_metrics['acc']
        new_acc = new_metrics['acc']
        combined_acc = combined_metrics['acc']
        
        # Enhanced H-Mean calculation with better formulation
        if step_num == 1:
            # For first step, measure forgetting effectiveness
            forget_drop = max(0.0, 100.0 - forget_acc)
        else:
            forget_drop = max(0.0, 100.0 - forget_acc)
        
        # Balanced accuracy between retention and new learning
        balanced_acc = (retain_acc + new_acc) / 2.0
        
        # Harmonic mean for retention-forgetting trade-off
        if retain_acc + forget_drop > 0:
            h_mean = (2 * retain_acc * forget_drop) / (retain_acc + forget_drop)
        else:
            h_mean = 0.0
        
        # Enhanced efficiency metric (combines accuracy and sparsity)
        efficiency = balanced_acc * (1.0 + zero_ratio)

        # Phase identification
        if epoch <= PHASE1_EPOCHS:
            phase = "Retention"
        elif epoch <= PHASE1_EPOCHS + PHASE2_EPOCHS:
            phase = "Balanced"
        else:
            phase = "Fine-tuning"
        
        # Get current learning rates
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        avg_lr = np.mean(current_lrs)

        # Enhanced logging with all metrics
        print(
            f"Epoch {epoch:03d}/{FORGET_EPOCHS:03d} [{phase:10s}] | "
            f"Retain {retain_acc:.2f}%({retain_metrics['confidence']:.1f}) | "
            f"Forget {forget_acc:.2f}%({forget_metrics['confidence']:.1f}) | "
            f"New {new_acc:.2f}%({new_metrics['confidence']:.1f}) | "
            f"Combined {combined_acc:.2f}% | "
            f"Balanced {balanced_acc:.2f}% | "
            f"H-Mean {h_mean:.2f} | "
            f"Efficiency {efficiency:.2f} | "
            f"Zero {zero_ratio:.3f} | "
            f"LR {avg_lr:.2e} | "
            f"α {alpha_current:.4f}"
        )

        # Store training history
        epoch_record = {
            'epoch': epoch,
            'phase': phase,
            'retain_acc': retain_acc,
            'forget_acc': forget_acc,
            'new_acc': new_acc,
            'combined_acc': combined_acc,
            'balanced_acc': balanced_acc,
            'h_mean': h_mean,
            'efficiency': efficiency,
            'zero_ratio': zero_ratio,
            'alpha_current': alpha_current,
            'avg_lr': avg_lr,
            'losses': epoch_metrics
        }
        training_history.append(epoch_record)

        # Enhanced model saving with multiple criteria
        save_model = False
        save_reason = ""
        
        current_metrics = {
            'retain_acc': retain_acc,
            'new_acc': new_acc,
            'combined_acc': combined_acc,
            'balanced_acc': balanced_acc,
            'zero_ratio': zero_ratio
        }
        
        # Save if combined accuracy improves significantly
        if combined_acc > best_metrics['combined_acc'] + 0.5:
            best_metrics.update(current_metrics)
            best_metrics['epoch'] = epoch
            save_model = True
            save_reason = f"Combined Acc {combined_acc:.2f}%"
            
        # Save if balanced accuracy improves with reasonable sparsity
        elif (balanced_acc > best_metrics['balanced_acc'] + 0.3 and zero_ratio >= 0.1):
            best_metrics.update(current_metrics)
            best_metrics['epoch'] = epoch
            save_model = True
            save_reason = f"Balanced Acc {balanced_acc:.2f}% + Sparsity {zero_ratio:.3f}"
            
        # Save if significant sparsity improvement with stable accuracy
        elif (zero_ratio > best_metrics['zero_ratio'] + 0.05 and 
              abs(balanced_acc - best_metrics['balanced_acc']) < 2.0):
            best_metrics.update(current_metrics)
            best_metrics['epoch'] = epoch
            save_model = True
            save_reason = f"Sparsity {zero_ratio:.3f} + Stable Acc {balanced_acc:.2f}%"

        if save_model:
            os.makedirs(CHECKPOINTS_STEPS_DIR, exist_ok=True)
            ckpt_path = os.path.join(CHECKPOINTS_STEPS_DIR, f"best_step{step_num}_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved new best model: {save_reason}")
            print(f"    Path: {ckpt_path}")

        # Log to file if logger exists
        if logger:
            logger.info(
                f"Step {step_num}, Epoch {epoch}: "
                f"Retain={retain_acc:.2f}, New={new_acc:.2f}, "
                f"Forget={forget_acc:.2f}, Combined={combined_acc:.2f}, "
                f"Balanced={balanced_acc:.2f}, Zero_Ratio={zero_ratio:.3f}, "
                f"Phase={phase}"
            )

        # Enhanced early stopping check
        if early_stopping(current_metrics, model):
            print(f"\n⏰ Early stopping triggered at epoch {epoch}")
            print(f"   Best balanced accuracy: {early_stopping.best_score:.2f}%")
            break

        # Phase transition messages
        if epoch == PHASE1_EPOCHS:
            print(f"\n🔄 Transitioning to Balanced Phase (emphasis on both retain and new)")
        elif epoch == PHASE1_EPOCHS + PHASE2_EPOCHS:
            print(f"\n🔧 Transitioning to Fine-tuning Phase (optimizing efficiency)")

    # Save final checkpoint
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint)
    
    # Final comprehensive evaluation
    print(f"\n" + "=" * 80)
    print(f"ENHANCED GS-LoRA STEP {step_num} COMPLETE")
    print(f"=" * 80)
    
    final_zero_ratio = compute_zero_group_ratio(lora_groups)
    
    # Final validation on all datasets
    print(f"📊 Final Comprehensive Evaluation:")
    
    final_retain_metrics = enhanced_evaluate(model, retain_val_loader, temperature=1.0)
    final_forget_metrics = enhanced_evaluate(model, forget_val_loader, temperature=1.0)
    final_new_metrics = enhanced_evaluate(model, new_val_loader, temperature=1.0)
    final_combined_metrics = enhanced_evaluate(model, combined_val_loader, temperature=1.0)
    
    print(f"   Retain Classes ({retain_classes[0]}-{retain_classes[-1]}):")
    print(f"     Accuracy: {final_retain_metrics['acc']:.2f}%")
    print(f"     F1-Score: {final_retain_metrics['f1']:.2f}%")
    print(f"     Confidence: {final_retain_metrics['confidence']:.1f}%")
    
    print(f"   Forget Classes ({forget_classes[0]}-{forget_classes[-1]}):")
    print(f"     Accuracy: {final_forget_metrics['acc']:.2f}% (lower is better)")
    print(f"     F1-Score: {final_forget_metrics['f1']:.2f}%")
    print(f"     Confidence: {final_forget_metrics['confidence']:.1f}%")
    
    print(f"   New Classes ({new_classes[0]}-{new_classes[-1]}):")
    print(f"     Accuracy: {final_new_metrics['acc']:.2f}%")
    print(f"     F1-Score: {final_new_metrics['f1']:.2f}%")
    print(f"     Confidence: {final_new_metrics['confidence']:.1f}%")
    
    print(f"   Combined Performance:")
    print(f"     Overall Accuracy: {final_combined_metrics['acc']:.2f}%")
    print(f"     Balanced Accuracy: {(final_retain_metrics['acc'] + final_new_metrics['acc'])/2:.2f}%")
    
    final_balanced = (final_retain_metrics['acc'] + final_new_metrics['acc']) / 2.0
    final_efficiency = final_balanced * (1.0 + final_zero_ratio)
    
    print(f"   Model Efficiency:")
    print(f"     Zero Group Ratio: {final_zero_ratio:.3f} ({final_zero_ratio*100:.1f}% sparse)")
    print(f"     Efficiency Score: {final_efficiency:.2f}")
    
    # Save training history
    history_path = os.path.join(CHECKPOINTS_STEPS_DIR, f"training_history_step{step_num}.json")
    import json
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"   Training History: {history_path}")
    
    print(f"   Best Performance Achieved:")
    print(f"     Epoch: {best_metrics['epoch']}")
    print(f"     Combined Accuracy: {best_metrics['combined_acc']:.2f}%")
    print(f"     Balanced Accuracy: {best_metrics['balanced_acc']:.2f}%")
    print(f"     Best Zero Ratio: {best_metrics['zero_ratio']:.3f}")
    
    print(f"   Final Model: {output_checkpoint}")
    print(f"=" * 80)
    
    # Success criteria check
    success_criteria = {
        'min_retain_acc': 60.0,
        'min_new_acc': 60.0,
        'max_forget_acc': 80.0,
        'min_combined_acc': 60.0,
        'min_sparsity': 0.1
    }
    
    success = True
    if final_retain_metrics['acc'] < success_criteria['min_retain_acc']:
        print(f"⚠️  Retain accuracy {final_retain_metrics['acc']:.2f}% below threshold {success_criteria['min_retain_acc']:.2f}%")
        success = False
    if final_new_metrics['acc'] < success_criteria['min_new_acc']:
        print(f"⚠️  New accuracy {final_new_metrics['acc']:.2f}% below threshold {success_criteria['min_new_acc']:.2f}%")
        success = False
    if final_forget_metrics['acc'] > success_criteria['max_forget_acc']:
        print(f"⚠️  Forget accuracy {final_forget_metrics['acc']:.2f}% above threshold {success_criteria['max_forget_acc']:.2f}%")
        success = False
    if final_combined_metrics['acc'] < success_criteria['min_combined_acc']:
        print(f"⚠️  Combined accuracy {final_combined_metrics['acc']:.2f}% below threshold {success_criteria['min_combined_acc']:.2f}%")
        success = False
    if final_zero_ratio < success_criteria['min_sparsity']:
        print(f"⚠️  Sparsity {final_zero_ratio:.3f} below threshold {success_criteria['min_sparsity']:.3f}")
        success = False
    
    if success:
        print(f"✅ All success criteria met!")
    else:
        print(f"❌ Some success criteria not met, but training completed")
    
    return success

# ---------------- Step Configuration ----------------
def get_step_config(step_num):
    """Get configuration for a specific step with enhanced settings"""
    steps_config = {
        1: {
            "input_ckpt": "checkpoints/face/oracle/0_49.pth",
            "output_ckpt": os.path.join(CHECKPOINTS_STEPS_DIR, "10_59.pth"),
            "retain_classes": list(range(10, 50)),
            "forget_classes": list(range(0, 10)),
            "new_classes": list(range(50, 60)),
            "description": "First step: Retain 10-49, Forget 0-9, Learn 50-59"
        },
        2: {
            "input_ckpt": os.path.join(CHECKPOINTS_STEPS_DIR, "10_59.pth"),
            "output_ckpt": os.path.join(CHECKPOINTS_STEPS_DIR, "20_69.pth"),
            "retain_classes": list(range(20, 60)),
            "forget_classes": list(range(10, 20)),
            "new_classes": list(range(60, 70)),
            "description": "Second step: Retain 20-59, Forget 10-19, Learn 60-69"
        },
        3: {
            "input_ckpt": os.path.join(CHECKPOINTS_STEPS_DIR, "20_69.pth"),
            "output_ckpt": os.path.join(CHECKPOINTS_STEPS_DIR, "30_79.pth"),
            "retain_classes": list(range(30, 70)),
            "forget_classes": list(range(20, 30)),
            "new_classes": list(range(70, 80)),
            "description": "Third step: Retain 30-69, Forget 20-29, Learn 70-79"
        },
        4: {
            "input_ckpt": os.path.join(CHECKPOINTS_STEPS_DIR, "30_79.pth"),
            "output_ckpt": os.path.join(CHECKPOINTS_STEPS_DIR, "40_89.pth"),
            "retain_classes": list(range(40, 80)),
            "forget_classes": list(range(30, 40)),
            "new_classes": list(range(80, 90)),
            "description": "Fourth step: Retain 40-79, Forget 30-39, Learn 80-89"
        },
        5: {
            "input_ckpt": os.path.join(CHECKPOINTS_STEPS_DIR, "40_89.pth"),
            "output_ckpt": os.path.join(CHECKPOINTS_STEPS_DIR, "50_99.pth"),
            "retain_classes": list(range(50, 90)),
            "forget_classes": list(range(40, 50)),
            "new_classes": list(range(90, 100)),
            "description": "Final step: Retain 50-89, Forget 40-49, Learn 90-99"
        }
    }
    return steps_config.get(step_num)

# ---------------- Enhanced Main Function ----------------
def main():
    parser = argparse.ArgumentParser(description="Enhanced GS-LoRA CLU Training with Improved Balance")
    parser.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Step number to run (1-5)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a specific checkpoint")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with additional logging")
    args = parser.parse_args()

    # Setup enhanced logging
    logger = setup_logging()
    
    print(f"🚀 Enhanced GS-LoRA CLU Training - Step {args.step}")
    print(f"=" * 60)
    print(f"🔧 Configuration:")
    print(f"   Device: {DEVICE}")
    print(f"   LoRA Rank: {LORA_RANK} (Enhanced)")
    print(f"   Base Learning Rate: {BASE_LR}")
    print(f"   Training Phases: {PHASE1_EPOCHS} + {PHASE2_EPOCHS} + {PHASE3_EPOCHS} = {FORGET_EPOCHS} epochs")
    print(f"   Loss Weights: Retain={ALPHA_RETAIN}, New={ALPHA_NEW}, Forget={ALPHA_FORGET}")
    print(f"   Sparsity Settings: Warmup={K_WARMUP}, Alpha_K={ALPHA_K}")
    print(f"   Learning Rate Multipliers: LoRA={LORA_LR_MULT}x, Retain={RETAIN_LR_MULT}x, New={NEW_LR_MULT}x")
    
    # Get step configuration
    step_config = get_step_config(args.step)
    if not step_config:
        print(f"❌ Invalid step number: {args.step}")
        return
    
    print(f"   Step Description: {step_config['description']}")
    print(f"=" * 60)
    
    # Override input checkpoint if resuming
    if args.resume:
        if os.path.exists(args.resume):
            step_config["input_ckpt"] = args.resume
            print(f"🔄 Resuming from: {args.resume}")
        else:
            print(f"❌ Resume checkpoint not found: {args.resume}")
            return
    
    # Create directories
    os.makedirs(CHECKPOINTS_STEPS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_ORACLE_DIR, exist_ok=True)
    
    # Log start time
    start_time = time.time()
    
    # Run enhanced GS-LoRA step
    success = train_single_step_enhanced_gs_lora(
        step_num=args.step,
        input_checkpoint=step_config["input_ckpt"],
        retain_classes=step_config["retain_classes"],
        forget_classes=step_config["forget_classes"],
        new_classes=step_config["new_classes"],
        output_checkpoint=step_config["output_ckpt"],
        logger=logger
    )
    
    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"\n⏱️  Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    if success:
        print(f"\n🎉 Enhanced GS-LoRA Step {args.step} Training Complete!")
        print(f"📁 Output saved to: {step_config['output_ckpt']}")
        if args.step < 5:
            print(f"➡️  Next command: python enhanced_gs_lora_clu_train.py --step {args.step + 1}")
        else:
            print(f"🏁 All steps complete! Final model ready for evaluation.")
        
        # Log success
        logger.info(f"Enhanced GS-LoRA Step {args.step} completed successfully in {training_time:.0f}s")
    else:
        print(f"\n❌ Enhanced GS-LoRA Step {args.step} completed with issues!")
        print(f"📁 Model still saved to: {step_config['output_ckpt']}")
        print(f"💡 Consider adjusting hyperparameters or training for more epochs")
        
        # Log failure
        logger.warning(f"Enhanced GS-LoRA Step {args.step} completed with issues in {training_time:.0f}s")

if __name__ == "__main__":
    main()