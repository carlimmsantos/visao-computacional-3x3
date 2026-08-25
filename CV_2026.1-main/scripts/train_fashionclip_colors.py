"""
Training script for FashionCLIP player color classification.

This script:
1. Splits the player dataset annotations into train/val/test sets
2. Fine-tunes FashionCLIP for player color classification
3. Evaluates performance on the test set
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    AutoProcessor,
    CLIPModel,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PlayerColorDataset(Dataset):
    """Dataset for player color classification."""

    def __init__(
        self,
        image_paths: List[str],
        texts: List[str],
        processor,
        base_dir: str = ".",
    ):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.base_dir = base_dir

        # Extract unique colors from texts and create labels
        self.color_classes = self._extract_color_classes()
        self.label_map = {color: idx for idx, color in enumerate(self.color_classes)}
        self.labels = [self.label_map[self._extract_color(text)] for text in texts]

    def _extract_color_classes(self) -> List[str]:
        """Extract unique color descriptions from texts."""
        colors = set()
        for text in self.texts:
            color = self._extract_color(text)
            colors.add(color)
        return sorted(list(colors))

    def _extract_color(self, text: str) -> str:
        """Extract color description from text."""
        if "purple and black" in text.lower():
            return "purple and black"
        elif "white" in text.lower():
            return "white"
        else:
            # Default fallback
            return text

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.base_dir, self.image_paths[idx])
        text = self.texts[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image
            image = Image.new("RGB", (224, 224), color="white")

        # Process image and text (without padding at individual level)
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=False,
        )

        return {
            "image_path": image_path,
            "text": text,
            "label": label,
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


def load_csv_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """Load image paths and texts from CSV."""
    image_paths = []
    texts = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.append(row["image_path"])
            texts.append(row["text"])

    return image_paths, texts


def collate_fn(batch):
    """Custom collate function that pads sequences to the same length."""
    image_paths = [item["image_path"] for item in batch]
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    # Pad input_ids and attention_mask to the same length
    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]

    # Find max length in batch
    max_length = max(len(ids) for ids in input_ids_list)

    # Pad sequences
    padded_input_ids = []
    padded_attention_mask = []

    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = F.pad(input_ids.unsqueeze(0), (0, padding_length), value=0).squeeze(0)
            attention_mask = F.pad(
                attention_mask.unsqueeze(0), (0, padding_length), value=0
            ).squeeze(0)
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)

    input_ids = torch.stack(padded_input_ids)
    attention_mask = torch.stack(padded_attention_mask)

    return {
        "image_path": image_paths,
        "text": texts,
        "label": labels,
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def split_data(
    image_paths: List[str],
    texts: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str]],
]:
    """Split data into train, val, test sets."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_samples = len(image_paths)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    train_paths = [image_paths[i] for i in train_indices]
    train_texts = [texts[i] for i in train_indices]

    val_paths = [image_paths[i] for i in val_indices]
    val_texts = [texts[i] for i in val_indices]

    test_paths = [image_paths[i] for i in test_indices]
    test_texts = [texts[i] for i in test_indices]

    return (
        (train_paths, train_texts),
        (val_paths, val_texts),
        (test_paths, test_texts),
    )


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    class_texts: Dict[int, str],
    processor,
):
    """Train for one epoch using CLIP contrastive learning."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Prepare class text embeddings
    class_embeddings = {}
    with torch.no_grad():
        for class_id, text in class_texts.items():
            text_inputs = processor(text=text, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items() if k in ['input_ids', 'attention_mask']}
            text_outputs = model.get_text_features(**text_inputs)
            class_embeddings[class_id] = F.normalize(text_outputs, dim=-1)

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move to device
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        # Get image embeddings
        image_outputs = model.get_image_features(pixel_values=pixel_values)
        image_embeddings = F.normalize(image_outputs, dim=-1)  # [batch_size, embedding_dim]

        # Calculate cosine similarity with all class texts
        logits = []
        for class_id in sorted(class_embeddings.keys()):
            class_embed = class_embeddings[class_id]
            # [batch_size, 1]
            similarity = torch.mm(image_embeddings, class_embed.t())
            logits.append(similarity)

        logits = torch.cat(logits, dim=1)  # [batch_size, num_classes]

        # Compute contrastive loss (cross entropy on similarities)
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | Acc: {100 * correct / total:.2f}%"
            )

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100 * correct / total

    return avg_loss, avg_acc


def evaluate(
    model,
    dataloader,
    device,
    class_texts: Dict[int, str],
    processor,
):
    """Evaluate model on a dataset using CLIP contrastive learning."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Prepare class text embeddings
    class_embeddings = {}
    with torch.no_grad():
        for class_id, text in class_texts.items():
            text_inputs = processor(text=text, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items() if k in ['input_ids', 'attention_mask']}
            text_outputs = model.get_text_features(**text_inputs)
            class_embeddings[class_id] = F.normalize(text_outputs, dim=-1)

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # Get image embeddings
            image_outputs = model.get_image_features(pixel_values=pixel_values)
            image_embeddings = F.normalize(image_outputs, dim=-1)  # [batch_size, embedding_dim]

            # Calculate cosine similarity with all class texts
            logits = []
            for class_id in sorted(class_embeddings.keys()):
                class_embed = class_embeddings[class_id]
                # [batch_size, 1]
                similarity = torch.mm(image_embeddings, class_embed.t())
                logits.append(similarity)

            logits = torch.cat(logits, dim=1)  # [batch_size, num_classes]

            # Compute contrastive loss
            loss = F.cross_entropy(logits, labels)

            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100 * correct / total

    return avg_loss, avg_acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def load_with_confidence_filter(csv_path: str, min_confidence: float = 0.7):
    """Load data with confidence filtering"""
    image_paths = []
    texts = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            confidence = float(row.get('confidence', 1.0))
            if confidence >= min_confidence:
                image_paths.append(row["image_path"])
                texts.append(row["text"])

    return image_paths, texts


def main():
    parser = argparse.ArgumentParser(
        description="Train FashionCLIP for player color classification"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="players_dataset_annotations.csv",
        help="Path to CSV file with annotations",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory for image paths",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="patrickjohncyh/fashion-clip",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fashionclip_trained",
        help="Output directory for trained model and results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of data for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio of data for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    image_paths, texts = image_paths, texts = load_with_confidence_filter(
        args.csv_path,
        min_confidence=0.7  # Filter low-confidence annotations
    )
    print(f"Loaded {len(image_paths)} samples")

    # Split data
    print("Splitting data...")
    (train_paths, train_texts), (val_paths, val_texts), (test_paths, test_texts) = (
        split_data(
            image_paths,
            texts,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    )

    print(f"Train: {len(train_paths)}")
    print(f"Val: {len(val_paths)}")
    print(f"Test: {len(test_paths)}")

    # Save split indices
    split_info = {
        "train_indices": list(range(len(train_paths))),
        "val_indices": list(range(len(train_paths), len(train_paths) + len(val_paths))),
        "test_indices": list(
            range(len(train_paths) + len(val_paths), len(image_paths))
        ),
        "train_size": len(train_paths),
        "val_size": len(val_paths),
        "test_size": len(test_paths),
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print("Split info saved to split_info.json")

    # Load model and processor
    print(f"Loading model {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device)

    # Create datasets
    print("Creating datasets...")
    train_dataset = PlayerColorDataset(
        train_paths, train_texts, processor, args.base_dir
    )
    val_dataset = PlayerColorDataset(val_paths, val_texts, processor, args.base_dir)
    test_dataset = PlayerColorDataset(test_paths, test_texts, processor, args.base_dir)

    print(f"Color classes: {train_dataset.color_classes}")

    # Save class mapping
    class_mapping = {
        idx: color for color, idx in train_dataset.label_map.items()
    }
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # Create class texts mapping (class_id -> text description)
    class_texts = {idx: color for color, idx in train_dataset.label_map.items()}

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training loop
    print("Starting training...")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, epoch + 1,
            class_texts, processor
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_dataloader, device, class_texts, processor
        )
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            model.save_pretrained(output_dir / "best_model")
            processor.save_pretrained(output_dir / "best_model")
            print(f"Best model saved with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Save final model
    model.save_pretrained(output_dir / "final_model")
    processor.save_pretrained(output_dir / "final_model")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History - Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training History - Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png")
    plt.close()
    print("Training history plot saved")

    # Load best model for testing
    print("\nLoading best model for testing...")
    best_model = CLIPModel.from_pretrained(output_dir / "best_model").to(device)
    best_processor = AutoProcessor.from_pretrained(output_dir / "best_model")

    # Test
    print("Testing on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate(
        best_model, test_dataloader, device, class_texts, best_processor
    )

    print(f"\n{'=' * 60}")
    print("TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, average=None
    )

    print("\nPer-class metrics:")
    for idx, color in enumerate(train_dataset.color_classes):
        print(
            f"{color}: Precision={precision[idx]:.4f}, Recall={recall[idx]:.4f}, "
            f"F1={f1[idx]:.4f}, Support={support[idx]}"
        )

    # Save test results
    test_results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "per_class_metrics": {
            color: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, color in enumerate(train_dataset.color_classes)
        },
    }

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print("Test results saved to test_results.json")

    # Plot confusion matrix
    plot_confusion_matrix(
        test_labels,
        test_preds,
        train_dataset.color_classes,
        output_dir / "confusion_matrix.png",
    )

    print(f"\nAll outputs saved to {output_dir}")
    print("Training completed!")


if __name__ == "__main__":
    main()
