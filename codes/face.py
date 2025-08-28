# face_steps_5.py
import os
import csv
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Your repo structure imports
from codes.utils import get_model, load_model_weights, print_parameter_stats
from codes.data import get_dynamic_loader

# ======================
# Hyperparameters (tuned to be gentle/stable)
# ======================
ALPHA = 0.5   # forgetting weight
BETA  = 1.0   # new learning weight
EL    = 10.0  # erasure limit
LR    = 5e-4  # you can lower to 5e-5 if still aggressive
EPOCHS = 50
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Losses
# ======================
def erasure_loss(logits, labels):
    ce_loss = F.cross_entropy(logits, labels, reduction="mean")
    return F.relu(EL - ce_loss)

def retention_loss(logits, labels):
    return F.cross_entropy(logits, labels, reduction="mean")

def acquisition_loss(logits, labels):
    return F.cross_entropy(logits, labels, reduction="mean")

@torch.no_grad()
def _margin_free_logits_from_emb(model, emb):
    emb_n = F.normalize(emb, dim=1)
    W = model.loss.weight
    W_n = F.normalize(W, dim=1)
    return F.linear(emb_n, W_n) * 64.0

# ======================
# Train / Eval
# ======================
def train_one_epoch(model, retain_loader, forget_loader, new_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(
        zip(retain_loader, forget_loader, new_loader),
        total=min(len(retain_loader), len(forget_loader), len(new_loader)),
        desc=f"🟢 Train Epoch {epoch}",
        leave=False,
    )

    for (retain_batch, forget_batch, new_batch) in pbar:
        optimizer.zero_grad()

        # Retain (labels already in 10–49, 20–59, etc. depending on step)
        x_r, y_r = retain_batch[0].to(DEVICE), retain_batch[1].to(DEVICE)
        logits_r, _ = model(x_r, y_r)
        loss_retain = retention_loss(logits_r, y_r)

        # Forget (labels in the to-be-forgotten 10-class slice)
        x_f, y_f = forget_batch[0].to(DEVICE), forget_batch[1].to(DEVICE)
        logits_f, _ = model(x_f, y_f)
        loss_forget = erasure_loss(logits_f, y_f)

        # New (labels in the 10 new classes)
        x_n, y_n = new_batch[0].to(DEVICE), new_batch[1].to(DEVICE)
        logits_n, _ = model(x_n, y_n)
        loss_new = acquisition_loss(logits_n, y_n)

        # Optional mellowing (kept from your final single-step script)
        scale_factor = 10.0
        loss_retain /= scale_factor
        loss_new    /= scale_factor

        loss = loss_retain*2 + ALPHA * loss_forget + BETA * loss_new
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(pbar))

@torch.no_grad()
def evaluate(model, retain_loader, forget_loader, new_loader, epoch):
    model.eval()

    def split_acc(loader, tag):
        all_preds, all_labels = [], []
        pbar = tqdm(loader, desc=f"🔵 Val Epoch {epoch} [{tag}]", leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits_train, emb = model(x, y)
            logits_eval = _margin_free_logits_from_emb(model, emb)
            preds = torch.argmax(logits_eval, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        if not all_labels:
            return 0.0
        return accuracy_score(all_labels, all_preds) * 100.0

    acc_forget = split_acc(forget_loader, "forget")
    acc_retain = split_acc(retain_loader, "retain")
    acc_new    = split_acc(new_loader,    "new")

    # Overall = retain + new
    combined = torch.utils.data.ConcatDataset([retain_loader.dataset, new_loader.dataset])
    all_loader = torch.utils.data.DataLoader(combined, batch_size=BATCH_SIZE)
    all_preds, all_labels = [], []
    pbar = tqdm(all_loader, desc=f"🔵 Val Epoch {epoch} [overall]", leave=False)
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits_train, emb = model(x, y)
        logits_eval = _margin_free_logits_from_emb(model, emb)
        preds = torch.argmax(logits_eval, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc_overall = accuracy_score(all_labels, all_preds) * 100.0

    return acc_forget, acc_retain, acc_new, acc_overall

# ======================
# Step spec utilities
# ======================
def step_specs():
    """
    Sliding window over CIFAR-100 classes:
    Step1: forget 0–9,   retain 10–49, new 50–59
    Step2: forget 10–19, retain 20–59, new 60–69
    Step3: forget 20–29, retain 30–69, new 70–79
    Step4: forget 30–39, retain 40–79, new 80–89
    Step5: forget 40–49, retain 50–89, new 90–99
    """
    return [
        {"forget": (0, 9),   "retain": (10, 49), "new": (50, 59)},
        {"forget": (10, 19), "retain": (20, 59), "new": (60, 69)},
        {"forget": (20, 29), "retain": (30, 69), "new": (70, 79)},
        {"forget": (30, 39), "retain": (40, 79), "new": (80, 89)},
        {"forget": (40, 49), "retain": (50, 89), "new": (90, 99)},
    ]

def ckpt_in_for_step(step_idx):
    """Step indices are 1..5"""
    if step_idx == 1:
        return "/home/jag/codes/CLU/checkpoints/face/oracle/0_49.pth"
    else:
        return f"./checkpoints/step{step_idx-1}.pth"

def ckpt_out_for_step(step_idx):
    return f"./checkpoints/step{step_idx}.pth"

# ======================
# Runner for a single step
# ======================
def run_step(step_idx, spec, log_writer):
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    print(f"\n{'='*88}")
    print(f"🚀 STEP {step_idx} | Forget {spec['forget'][0]}-{spec['forget'][1]}  "
          f"| Retain {spec['retain'][0]}-{spec['retain'][1]}  "
          f"| New {spec['new'][0]}-{spec['new'][1]}")
    print(f"{'='*88}")

    # Model
    model = get_model(num_classes=100, lora_rank=8, pretrained=False, device=DEVICE)
    ckpt_in = ckpt_in_for_step(step_idx)
    if os.path.exists(ckpt_in):
        print(f"📥 Loading checkpoint: {ckpt_in}")
        load_model_weights(model, ckpt_in, strict=False)
    else:
        print(f"⚠️ Warning: input checkpoint not found at {ckpt_in}. Starting without preload.")

    print_parameter_stats(model)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Loaders
    retain_tr = get_dynamic_loader(class_range=spec["retain"], mode="train", batch_size=BATCH_SIZE)
    forget_tr = get_dynamic_loader(class_range=spec["forget"], mode="train", batch_size=BATCH_SIZE)
    new_tr    = get_dynamic_loader(class_range=spec["new"],    mode="train", batch_size=BATCH_SIZE)

    retain_va = get_dynamic_loader(class_range=spec["retain"], mode="test",  batch_size=BATCH_SIZE)
    forget_va = get_dynamic_loader(class_range=spec["forget"], mode="test",  batch_size=BATCH_SIZE)
    new_va    = get_dynamic_loader(class_range=spec["new"],    mode="test",  batch_size=BATCH_SIZE)

    # Initial eval (epoch 0)
    f0, r0, n0, o0 = evaluate(model, retain_va, forget_va, new_va, epoch=0)
    print(f"Epoch 000/{EPOCHS:03d} | Forget ↓ {f0:.2f}% | Retain ↑ {r0:.2f}% | New ↑ {n0:.2f}% | Overall ↑ {o0:.2f}%")
    log_writer.writerow([step_idx, 0, "-", f"{f0:.4f}", f"{r0:.4f}", f"{n0:.4f}", f"{o0:.4f}"])

    # Train epochs
    best_overall = -1.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, retain_tr, forget_tr, new_tr, optimizer, epoch)
        f_acc, r_acc, n_acc, o_acc = evaluate(model, retain_va, forget_va, new_va, epoch)
        print(f"Epoch {epoch:03d}/{EPOCHS:03d} | Train Loss {tr_loss:.4f} "
              f"| Forget ↓ {f_acc:.2f}% | Retain ↑ {r_acc:.2f}% | New ↑ {n_acc:.2f}% | Overall ↑ {o_acc:.2f}%")
        log_writer.writerow([step_idx, epoch, f"{tr_loss:.4f}", f"{f_acc:.4f}", f"{r_acc:.4f}", f"{n_acc:.4f}", f"{o_acc:.4f}"])

        # Save best-per-overall within the step (optional)
        if o_acc > best_overall:
            best_overall = o_acc
            torch.save(model.state_dict(), ckpt_out_for_step(step_idx) + ".best")
            # keep quiet to avoid too many prints

    # Save final step checkpoint
    ckpt_out = ckpt_out_for_step(step_idx)
    torch.save(model.state_dict(), ckpt_out)
    print(f"✅ Saved STEP {step_idx} checkpoint → {ckpt_out} (best overall in-step = {best_overall:.2f}%)")

# ======================
# Main: all 5 steps
# ======================
def main():
    specs = step_specs()
    os.makedirs("./logs", exist_ok=True)
    csv_path = "./logs/face_steps_metrics.csv"
    fresh_file = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if fresh_file:
            writer.writerow(["step", "epoch", "train_loss", "forget_acc", "retain_acc", "new_acc", "overall_acc"])

        for idx, spec in enumerate(specs, start=1):
            run_step(idx, spec, writer)

    print(f"\n📒 Logs appended to: {csv_path}")
    print("🏁 All 5 steps completed.")

if __name__ == "__main__":
    main()
