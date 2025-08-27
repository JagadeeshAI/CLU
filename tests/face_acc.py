import os
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

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
    W = model.loss.weight  # [C, D]
    W_n = F.normalize(W, dim=1)
    logits_eval = F.linear(emb_n, W_n) * 64.0
    return logits_eval


@torch.no_grad()
def evaluate_model_detailed(model, dataloader, device, class_offset=0, class_names=None):
    """
    Comprehensive evaluation of the face recognition model
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confidences = []
    all_embeddings = []
    
    print("🔍 Evaluating model performance...")
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        original_labels = labels.clone()
        labels = (labels - class_offset).to(device, non_blocking=True).long()
        
        # Get model outputs
        logits_train, embeddings = model(images, labels)
        
        # Use margin-free logits for evaluation
        logits_eval = _margin_free_logits_from_emb(model, embeddings)
        
        # Get predictions and confidences
        probs = F.softmax(logits_eval, dim=1)
        confidences, preds = torch.max(probs, dim=1)
        
        # Store results
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())
        all_embeddings.append(embeddings.cpu().numpy())
        
        # Update progress bar with current batch accuracy
        batch_acc = (preds == labels).float().mean().item() * 100
        pbar.set_postfix(acc=f"{batch_acc:.2f}%")
    
    # Combine all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(
        all_labels, all_preds, all_confidences, class_names
    )
    
    return metrics, all_embeddings, all_labels, all_preds, all_confidences


def calculate_comprehensive_metrics(labels, preds, confidences, class_names=None):
    """Calculate comprehensive evaluation metrics"""
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, preds) * 100
    metrics['precision'] = precision_score(labels, preds, average='macro', zero_division=0) * 100
    metrics['recall'] = recall_score(labels, preds, average='macro', zero_division=0) * 100
    metrics['f1_score'] = f1_score(labels, preds, average='macro', zero_division=0) * 100
    
    # Per-class metrics
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0) * 100
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0) * 100
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0) * 100
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    # Confidence statistics
    metrics['avg_confidence'] = np.mean(confidences) * 100
    metrics['confidence_std'] = np.std(confidences) * 100
    
    # Correct vs incorrect predictions confidence
    correct_mask = np.array(labels) == np.array(preds)
    metrics['correct_confidence'] = np.mean(np.array(confidences)[correct_mask]) * 100 if np.any(correct_mask) else 0
    metrics['incorrect_confidence'] = np.mean(np.array(confidences)[~correct_mask]) * 100 if np.any(~correct_mask) else 0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    
    return metrics


def save_results_to_json(metrics, class_range, checkpoint_path, total_samples, evaluation_time):
    """Save evaluation results to JSON file with overall and per-class accuracy for 100 classes"""
    
    start_class, end_class = class_range
    
    # Create simple results dictionary with only accuracy
    results = {
        "overall_accuracy": round(float(metrics['accuracy']), 4),
        "class_wise_accuracy": {}
    }
    
    # Per-class accuracy for all 100 classes
    for class_id in range(100):
        if class_id >= start_class and class_id <= end_class:
            # This class was evaluated
            local_class_id = class_id - start_class
            if local_class_id < len(metrics['recall_per_class']):
                results["class_wise_accuracy"][str(class_id)] = round(float(metrics['recall_per_class'][local_class_id]), 4)
            else:
                results["class_wise_accuracy"][str(class_id)] = 0.0
        else:
            # This class was not evaluated
            results["class_wise_accuracy"][str(class_id)] = 0.0
    
    # Create results directory
    os.makedirs("./results", exist_ok=True)
    
    # Save JSON file
    json_filename = f"./results/face_recognition_results_{start_class}_{end_class}.json"
    
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {json_filename}")
    
    return json_filename


def print_evaluation_results(metrics, class_range, class_names=None):
    """Print comprehensive evaluation results"""
    
    start_class, end_class = class_range
    num_classes = end_class - start_class + 1
    
    print(f"\n{'='*80}")
    print(f"📊 FACE RECOGNITION MODEL EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"🎯 Model: Classes {start_class}-{end_class} ({num_classes} classes)")
    print(f"📁 Checkpoint: /home/jag/codes/CLU/checkpoints/face/oracle/{start_class}_{end_class}.pth")
    print(f"{'='*80}")
    
    # Overall metrics
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Accuracy:     {metrics['accuracy']:6.2f}%")
    print(f"   Precision:    {metrics['precision']:6.2f}%")
    print(f"   Recall:       {metrics['recall']:6.2f}%")
    print(f"   F1-Score:     {metrics['f1_score']:6.2f}%")
    
    # Confidence analysis
    print(f"\n🔍 CONFIDENCE ANALYSIS:")
    print(f"   Average Confidence:           {metrics['avg_confidence']:6.2f}% (±{metrics['confidence_std']:.2f}%)")
    print(f"   Correct Predictions Conf:     {metrics['correct_confidence']:6.2f}%")
    print(f"   Incorrect Predictions Conf:   {metrics['incorrect_confidence']:6.2f}%")
    print(f"   Confidence Gap:               {metrics['correct_confidence'] - metrics['incorrect_confidence']:+6.2f}%")
    
    # Per-class performance summary
    print(f"\n📈 PER-CLASS PERFORMANCE SUMMARY:")
    print(f"   Best Class F1:     {np.max(metrics['f1_per_class']):6.2f}%")
    print(f"   Worst Class F1:    {np.min(metrics['f1_per_class']):6.2f}%")
    print(f"   F1 Std Dev:        {np.std(metrics['f1_per_class']):6.2f}%")
    
    # Detailed per-class results (show top 10 and bottom 10)
    class_performance = list(zip(
        range(len(metrics['f1_per_class'])),
        metrics['precision_per_class'],
        metrics['recall_per_class'],
        metrics['f1_per_class']
    ))
    
    # Sort by F1 score
    class_performance.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n🏆 TOP 10 PERFORMING CLASSES:")
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print(f"{'-'*45}")
    for i, (class_id, prec, rec, f1) in enumerate(class_performance[:10]):
        actual_class = class_id + start_class
        print(f"{actual_class:<8} {prec:>9.2f}%   {rec:>7.2f}%   {f1:>7.2f}%")
    
    print(f"\n📉 BOTTOM 10 PERFORMING CLASSES:")
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print(f"{'-'*45}")
    for i, (class_id, prec, rec, f1) in enumerate(class_performance[-10:]):
        actual_class = class_id + start_class
        print(f"{actual_class:<8} {prec:>9.2f}%   {rec:>7.2f}%   {f1:>7.2f}%")


def save_confusion_matrix(confusion_matrix, class_range, save_dir="./results"):
    """Save confusion matrix visualization"""
    
    os.makedirs(save_dir, exist_ok=True)
    start_class, end_class = class_range
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=False, cmap='Blues', fmt='d')
    plt.title(f'Confusion Matrix - Face Recognition Model (Classes {start_class}-{end_class})')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    save_path = f"{save_dir}/confusion_matrix_{start_class}_{end_class}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Confusion matrix saved to: {save_path}")


def main():
    """Main evaluation function"""
    
    # Configuration
    checkpoint_path = "/home/jag/codes/CLU/checkpoints/face/oracle/0_49.pth"
    class_range = (0, 49)  # Classes 0-49
    class_start, class_end = class_range
    num_classes = class_end - class_start + 1
    
    print(f"🚀 Face Recognition Model Accuracy Evaluation")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"🎯 Classes: {class_start}-{class_end} ({num_classes} classes)")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Set Config for face recognition task
    Config.TaskName = "Face_recognition"  # Make sure this matches your config
    
    # Create model
    print(f"\n🏗️ Creating model...")
    model = get_model(
        num_classes=100,  # Full dataset classes
        pretrained=False,
        device=device
    )
    
    # Load checkpoint
    print(f"📥 Loading checkpoint...")
    try:
        load_model_weights(model, checkpoint_path, strict=False)
        print(f"✅ Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Print model statistics
    print_parameter_stats(model)
    
    # Create test data loader
    print(f"\n📊 Creating test data loader...")
    try:
        test_loader = get_dynamic_loader(
            class_range=class_range,
            mode="test",  # Use test mode for face recognition
            batch_size=64,
            image_size=224,
            num_workers=4,
            pin_memory=(device.type == "cuda")
        )
        print(f"✅ Test loader created with {len(test_loader)} batches")
        
        # Calculate total samples
        total_samples = len(test_loader.dataset)
        print(f"📈 Total test samples: {total_samples}")
        
    except Exception as e:
        print(f"❌ Error creating data loader: {e}")
        return
    
    # Run evaluation
    print(f"\n🔍 Starting evaluation...")
    start_time = time.time()
    
    try:
        metrics, embeddings, labels, preds, confidences = evaluate_model_detailed(
            model=model,
            dataloader=test_loader,
            device=device,
            class_offset=class_start
        )
        
        evaluation_time = (time.time() - start_time) / 60.0
        print(f"\n✅ Evaluation completed in {evaluation_time:.2f} minutes")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return
    
    # Save results to JSON
    json_file = save_results_to_json(
        metrics=metrics,
        class_range=class_range,
        checkpoint_path=checkpoint_path,
        total_samples=total_samples,
        evaluation_time=evaluation_time
    )
    
    # Print results
    print_evaluation_results(metrics, class_range)
    
    # Save confusion matrix
    try:
        save_confusion_matrix(metrics['confusion_matrix'], class_range)
    except Exception as e:
        print(f"⚠️ Could not save confusion matrix: {e}")
    
    # Additional analysis
    print(f"\n🔬 ADDITIONAL ANALYSIS:")
    correct_predictions = np.sum(np.array(labels) == np.array(preds))
    total_predictions = len(labels)
    error_rate = (1 - metrics['accuracy'] / 100) * 100
    
    print(f"   Correct Predictions:    {correct_predictions:,} / {total_predictions:,}")
    print(f"   Error Rate:             {error_rate:.2f}%")
    print(f"   Embedding Dimension:    {embeddings.shape[1]}")
    print(f"   Average Embedding Norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"🎯 Model Performance:     {metrics['accuracy']:.2f}% accuracy")
    print(f"⏱️  Evaluation Time:      {evaluation_time:.2f} minutes")
    print(f"📊 Classes Evaluated:     {class_start}-{class_end} ({num_classes} classes)")
    print(f"📈 Test Samples:          {total_samples:,}")
    print(f"💾 Results saved to:      {json_file}")
    print(f"📊 JSON contains:         Overall accuracy + per-class metrics for all 100 classes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()