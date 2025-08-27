# codes/face_steps.py
import os
import copy
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from config import Config
from codes.utils import get_model, load_model_weights, print_parameter_stats
from codes.data import get_dynamic_loader

from torch.utils.data import DataLoader, Subset

# -----------------------------
# Step-1 class splits (sliding window)
# -----------------------------
FORGET_RANGE = (0, 9)
RETAIN_RANGE = (10, 49)
NEW_RANGE    = (50, 59)

# Expected unique class counts for sanity checks
EXPECTED_FORGET = 10
EXPECTED_RETAIN = 40
EXPECTED_NEW    = 10

# -----------------------------
# Hyperparameters (gentle/stable)
# -----------------------------
EPOCHS       = 50
BATCH_SIZE   = 32
WEIGHT_DECAY = 1e-4
HEAD_LR      = 1e-4    # classifier rows learn faster
LORA_LR      = 3e-5    # LoRA learns slower (stability)
EL           = 10.0    # erasure bound (lower -> stronger erasure)
SCALE_LOSS   = 10.0    # normalize CE magnitudes due to ArcFace scaling
KD_T         = 2.0     # distillation temperature
KD_LAMBDA    = 0.7     # retained KD strength
MAX_NORM     = 1.0     # grad clipping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Helpers: data inspection / proxy-val
# -----------------------------
def _peek_label_stats(loader, max_batches=3):
    labels = []
    taken = 0
    for _, y in loader:
        labels.extend(y.tolist())
        taken += 1
        if taken >= max_batches:
            break
    if not labels:
        return 0, None, None
    import numpy as np
    u = np.unique(labels)
    return len(u), min(labels), max(labels)

def _debug_loader(name, loader):
    try:
        n_batches = len(loader)
    except TypeError:
        n_batches = -1
    n_samples = len(loader.dataset) if hasattr(loader, "dataset") else -1
    ucnt, lmin, lmax = _peek_label_stats(loader, max_batches=3)
    print(f"[Data] {name}: batches={n_batches}, samples={n_samples}, "
          f"labels[min,max]={lmin},{lmax} unique={ucnt}")

def _build_proxy_val_from_train(train_loader, take_ratio=0.25, batch_size=32, num_workers=4, pin_memory=False):
    """
    Use a small, held-out slice of the *train* subset as a validation proxy,
    without touching data.py.
    """
    assert hasattr(train_loader, "dataset"), "train_loader must have .dataset (Subset)."
    train_subset = train_loader.dataset  # Subset over ImageFolder
    full_indices = train_subset.indices if hasattr(train_subset, "indices") else list(range(len(train_subset)))
    if len(full_indices) == 0:
        raise RuntimeError("Proxy-val: train subset has zero indices.")

    # take every k-th sample to avoid heavy overlap
    step = max(int(1.0 / max(take_ratio, 1e-3)), 5)  # e.g., 0.25 -> step≈4, clamp min 5
    proxy_indices = full_indices[::step]
    if len(proxy_indices) < batch_size:
        proxy_indices = full_indices[: min(len(full_indices), max(batch_size, len(full_indices)//5 + 1))]

    # IMPORTANT: reindex original dataset (not the Subset)
    proxy_subset = Subset(train_subset.dataset, proxy_indices)
    proxy_loader = DataLoader(proxy_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return proxy_loader

# -----------------------------
# Helpers: losses / eval
# -----------------------------
@torch.no_grad()
def _margin_free_logits_from_emb(model, emb):
    """
    Margin-free logits for evaluation (ArcFace/CosFace heads):
      logits = <normalize(emb)> · <normalize(W)>^T * s
    """
    emb_n = F.normalize(emb, dim=1)
    W = model.loss.weight
    W_n = F.normalize(W, dim=1)
    return F.linear(emb_n, W_n) * 64.0

def erasure_loss(logits_train, labels):
    """Bounded erasure loss: ReLU(EL - CE). Minimizing this encourages CE↑ until EL."""
    ce = F.cross_entropy(logits_train, labels, reduction="mean")
    return F.relu(EL - ce)

def retention_ce(logits_train, labels):
    return F.cross_entropy(logits_train, labels, reduction="mean")

def acquisition_ce(logits_train, labels):
    return F.cross_entropy(logits_train, labels, reduction="mean")

def kd_loss(student_logits, teacher_logits, T=2.0):
    """KL(student || teacher) with temperature (batchmean)."""
    log_p = F.log_softmax(student_logits / T, dim=1)
    q     = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)

@torch.no_grad()
def evaluate(model, retain_loader, forget_loader, new_loader, epoch):
    model.eval()

    def eval_split(loader, tag):
        preds, labels = [], []
        pbar = tqdm(loader, desc=f"🔵 Val Epoch {epoch} [{tag}]", leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits_train, emb = model(x, y)
            logits_eval = _margin_free_logits_from_emb(model, emb)
            pred = logits_eval.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
        if not labels:
            return 0.0
        return accuracy_score(labels, preds) * 100

    acc_forget = eval_split(forget_loader, "forget")
    acc_retain = eval_split(retain_loader, "retain")
    acc_new    = eval_split(new_loader,    "new")

    # Overall over retain+new (active target set)
    all_preds, all_labels = [], []
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset([retain_loader.dataset, new_loader.dataset])
    overall_loader = DataLoader(combined, batch_size=BATCH_SIZE)
    pbar = tqdm(overall_loader, desc=f"🔵 Val Epoch {epoch} [overall]", leave=False)
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits_train, emb = model(x, y)
        logits_eval = _margin_free_logits_from_emb(model, emb)
        pred = logits_eval.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc_overall = accuracy_score(all_labels, all_preds) * 100 if all_labels else 0.0

    return acc_forget, acc_retain, acc_new, acc_overall

# -----------------------------
# Optimizer + grad masking
# -----------------------------
def make_optimizer(model):
    """Separate LR for LoRA vs ArcFace head."""
    lora_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        elif name.startswith("loss."):
            head_params.append(p)

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": LORA_LR})
    if head_params:
        param_groups.append({"params": head_params, "lr": HEAD_LR})

    if not param_groups:  # Fallback
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": HEAD_LR}]
    return optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

def mask_classifier_grads(model):
    """
    Gradient mask on ArcFace head rows:
      - freeze forgotten rows (0-9)
      - down-weight retained rows (10-49)
      - full for new rows (50-59)
      - small for others (60-99) to keep steady
    """
    W = model.loss.weight
    if W.grad is None:
        return
    g = W.grad
    with torch.no_grad():
        dev = g.device
        forget_ids  = torch.arange(FORGET_RANGE[0], FORGET_RANGE[1] + 1, device=dev)
        retain_ids  = torch.arange(RETAIN_RANGE[0], RETAIN_RANGE[1] + 1, device=dev)
        new_ids     = torch.arange(NEW_RANGE[0],    NEW_RANGE[1] + 1,    device=dev)
        other_ids   = torch.arange(60, 100, device=dev)  # untouched classes in this step

        g[forget_ids] *= 0.0
        g[retain_ids] *= 0.25
        g[new_ids]    *= 1.0
        g[other_ids]  *= 0.1

# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, teacher, retain_loader, forget_loader, new_loader, optimizer, epoch):
    model.train()
    # Curriculum on weights per epoch
    warm = min(epoch / 10.0, 1.0)
    cur_ALPHA = 0.5 * (1.0 - 0.5 * warm)      # 0.5 -> ~0.25
    cur_BETA  = 1.0 * (0.75 + 0.25 * warm)    # 0.75 -> 1.0

    num_iters = min(len(retain_loader), len(forget_loader), len(new_loader))
    assert num_iters > 0, "No iterations possible — check your loaders (retain/forget/new)!"
    pbar = tqdm(zip(retain_loader, forget_loader, new_loader),
                total=num_iters, desc=f"🟢 Train Epoch {epoch}", leave=False)

    running = 0.0
    for (rb, fb, nb) in pbar:
        optimizer.zero_grad()

        # --- Retain ---
        xr, yr = rb[0].to(DEVICE), rb[1].to(DEVICE)
        logits_r_train, emb_r = model(xr, yr)
        logits_r_eval = _margin_free_logits_from_emb(model, emb_r)

        with torch.no_grad():
            t_logits_train, t_emb_r = teacher(xr, yr)
            t_logits_r_eval = _margin_free_logits_from_emb(teacher, t_emb_r)

        ce_retain = retention_ce(logits_r_train, yr)
        # CE + KD on margin-free logits for stability of retained set
        loss_retain = 0.5 * ce_retain + 0.5 * KD_LAMBDA * kd_loss(logits_r_eval, t_logits_r_eval, T=KD_T)

        # --- Forget ---
        xf, yf = fb[0].to(DEVICE), fb[1].to(DEVICE)
        logits_f_train, emb_f = model(xf, yf)
        loss_forget = erasure_loss(logits_f_train, yf)

        # --- New ---
        xn, yn = nb[0].to(DEVICE), nb[1].to(DEVICE)
        logits_n_train, emb_n = model(xn, yn)
        loss_new = acquisition_ce(logits_n_train, yn)

        # Normalize loss scales (ArcFace CE tends to be large)
        total = (loss_retain + cur_ALPHA * loss_forget + cur_BETA * loss_new) / SCALE_LOSS
        total.backward()

        # Gradient masking on ArcFace rows
        mask_classifier_grads(model)

        # Clip for safety
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)

        optimizer.step()

        running += total.item()
        pbar.set_postfix(loss=f"{total.item():.4f}")

    return running / num_iters

# -----------------------------
# Main
# -----------------------------
def main():
    # Ensure face task in config if data.py depends on it
    Config.TaskName = "face_recognition"

    # --- Build model + load pretrained 0_49 checkpoint ---
    model = get_model(
        num_classes=100,
        pretrained=False,
        device=DEVICE,
        lora_rank=8  # LoRA enabled
    )
    print("✅ Built face model")

    ckpt = "/home/jag/codes/CLU/checkpoints/face/oracle/0_49.pth"
    if os.path.exists(ckpt):
        print(f"🔄 Loading model weights from {ckpt}...")
        load_model_weights(model, ckpt, strict=False)
    else:
        print(f"⚠️ Pretrained checkpoint not found at: {ckpt}")

    print_parameter_stats(model)

    # Frozen teacher (epoch-0 snapshot)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Data loaders (no label remapping; head has 100 outputs) ---
    pin = (DEVICE.type == "cuda")
    retain_tr = get_dynamic_loader(class_range=RETAIN_RANGE, mode="train", batch_size=BATCH_SIZE,
                                   image_size=224, num_workers=4, pin_memory=pin)
    forget_tr = get_dynamic_loader(class_range=FORGET_RANGE, mode="train", batch_size=BATCH_SIZE,
                                   image_size=224, num_workers=4, pin_memory=pin)
    new_tr    = get_dynamic_loader(class_range=NEW_RANGE,    mode="train", batch_size=BATCH_SIZE,
                                   image_size=224, num_workers=4, pin_memory=pin)

    retain_val = get_dynamic_loader(class_range=RETAIN_RANGE, mode="test", batch_size=BATCH_SIZE,
                                    image_size=224, num_workers=4, pin_memory=pin)
    forget_val = get_dynamic_loader(class_range=FORGET_RANGE, mode="test", batch_size=BATCH_SIZE,
                                    image_size=224, num_workers=4, pin_memory=pin)
    new_val    = get_dynamic_loader(class_range=NEW_RANGE,    mode="test", batch_size=BATCH_SIZE,
                                    image_size=224, num_workers=4, pin_memory=pin)

    # --- Inspect coverage ---
    _debug_loader("retain_tr", retain_tr)
    _debug_loader("forget_tr", forget_tr)
    _debug_loader("new_tr",    new_tr)
    _debug_loader("retain_val", retain_val)
    _debug_loader("forget_val", forget_val)
    _debug_loader("new_val",    new_val)

    # --- If val coverage is too narrow, build proxy-val from train ---
    MIN_FRACTION = 0.5  # accept if we see at least half expected unique classes
    ret_u, _, _ = _peek_label_stats(retain_val)
    new_u, _, _ = _peek_label_stats(new_val)
    for_u, _, _ = _peek_label_stats(forget_val)

    need_proxy_retain = (ret_u < max(1, int(EXPECTED_RETAIN * MIN_FRACTION)))
    need_proxy_new    = (new_u < max(1, int(EXPECTED_NEW    * MIN_FRACTION)))
    need_proxy_forget = (for_u < max(1, int(EXPECTED_FORGET * MIN_FRACTION)))

    if need_proxy_retain:
        print("⚠️ retain_val has too few classes — building proxy-val from retain_tr")
        retain_val = _build_proxy_val_from_train(retain_tr, take_ratio=0.25,
                                                 batch_size=BATCH_SIZE, num_workers=4, pin_memory=pin)
    if need_proxy_new:
        print("⚠️ new_val has too few classes — building proxy-val from new_tr")
        new_val = _build_proxy_val_from_train(new_tr, take_ratio=0.25,
                                              batch_size=BATCH_SIZE, num_workers=4, pin_memory=pin)
    if need_proxy_forget:
        print("⚠️ forget_val has too few classes — building proxy-val from forget_tr")
        forget_val = _build_proxy_val_from_train(forget_tr, take_ratio=0.25,
                                                 batch_size=BATCH_SIZE, num_workers=4, pin_memory=pin)

    # Reprint after fixing
    _debug_loader("retain_val(final)", retain_val)
    _debug_loader("forget_val(final)", forget_val)
    _debug_loader("new_val(final)",    new_val)

    # --- Optimizer with param groups ---
    optimizer = make_optimizer(model)

    # --- Initial evaluation (epoch 0) ---
    f0, r0, n0, o0 = evaluate(model, retain_val, forget_val, new_val, epoch=0)
    print(f"| Forget ↓ {f0:.2f}% | Retain ↑ {r0:.2f}% | New ↑ {n0:.2f}% | Overall ↑ {o0:.2f}%")

    # --- Training ---
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, teacher, retain_tr, forget_tr, new_tr, optimizer, epoch)
        f, r, n, o = evaluate(model, retain_val, forget_val, new_val, epoch)

        print(f"\nEpoch {epoch:03d}/{EPOCHS:03d} | Train Loss {train_loss:.4f} "
              f"| Forget ↓ {f:.2f}% | Retain ↑ {r:.2f}% | New ↑ {n:.2f}% | Overall ↑ {o:.2f}%\n")

    # --- Save Step-1 checkpoint ---
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/step1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Step 1 completed. Saved to {save_path}")

if __name__ == "__main__":
    main()
