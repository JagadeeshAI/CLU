import os
import torch
import json
from tqdm import tqdm
from codes.utils import get_model, load_model_weights
from codes.data import get_dynamic_loader
from config import Config
import torch.nn.functional as F

# ✅ Step-wise ranges (consistent keys: "forget", "retain", "new", "overall")
STEP_RANGES = {
    0: {'forget': (0, 9), 'retain': (0, 49)},
    1: {'forget': (0, 9), 'retain': (10, 49), 'new': (50, 59), 'overall': (10, 59)},
    2: {'forget': (0, 19), 'retain': (20, 59), 'new': (60, 69), 'overall': (20, 69)},
    3: {'forget': (0, 29), 'retain': (30, 69), 'new': (70, 79), 'overall': (30, 79)},
    4: {'forget': (0, 39), 'retain': (40, 79), 'new': (80, 89), 'overall': (40, 89)},
    5: {'forget': (0, 49), 'retain': (50, 89), 'new': (90, 99), 'overall': (50, 99)}
}


@torch.no_grad()
def _margin_free_logits_from_emb(model, emb):
    """Compute logits WITHOUT ArcFace margin (for evaluation)."""
    emb_n = F.normalize(emb, dim=1)
    W = model.loss.weight
    W_n = F.normalize(W, dim=1)
    logits_eval = F.linear(emb_n, W_n) * 64.0
    return logits_eval


def evaluate(model, dataloader, device, desc="Evaluating"):
    """Evaluate model on given dataloader, return accuracy (%)"""
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc, leave=False, unit="batch"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass (face recognition model returns logits + emb)
            logits_train, emb = model(images, labels)

            # Use margin-free logits for evaluation
            logits_eval = _margin_free_logits_from_emb(model, emb)
            preds = logits_eval.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_checkpoint(checkpoint_path, step, batch_size=64):
    """Load checkpoint and evaluate on forget/retain/new/overall ranges"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Evaluating {checkpoint_path} on STEP {step} (device={device})")

    # Always build model with 100 classes
    model = get_model(config=Config, num_classes=100, pretrained=False, device=device)
    load_model_weights(model, checkpoint_path, strict=False)
    model.eval()

    results = {}
    for eval_type, class_range in STEP_RANGES[step].items():
        print(f"\n🔍 Evaluating {eval_type} classes: {class_range[0]}–{class_range[1]}")
        loader = get_dynamic_loader(
            class_range=class_range,
            mode="val",
            batch_size=batch_size,
            image_size=224,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )
        acc = evaluate(model, loader, device, desc=f"🔍 {eval_type} {class_range}")
        results[eval_type] = round(acc, 2)
        print(f"  ✅ {eval_type:<8} ({class_range[0]}–{class_range[1]}): {acc:.2f}%")

    return results


if __name__ == "__main__":
    # ✅ Example usage — adjust these
    # checkpoint = "checkpoints/steps/best_step1_epoch47.pth"
    # checkpoint = "checkpoints/steps/best_step2_epoch91.pth"
    # checkpoint = "checkpoints/steps/best_step3_epoch41.pth"
    checkpoint = "checkpoints/steps/best_step5_epoch44.pth"
    step = 5

    results = evaluate_checkpoint(checkpoint, step)
    print("\n📊 Final Results:", json.dumps(results, indent=2))
