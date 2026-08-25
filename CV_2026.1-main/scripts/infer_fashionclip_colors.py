"""
Inference script for trained FashionCLIP player color classifier.

This script loads a trained FashionCLIP model and performs inference on new images.
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import json


def load_model_and_processor(model_dir):
    """Load model and processor from directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForZeroShotImageClassification.from_pretrained(model_dir).to(device)
    
    return model, processor, device


def load_class_mapping(model_dir):
    """Load class mapping from training directory."""
    mapping_path = Path(model_dir).parent / "class_mapping.json"
    
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
            # Convert string keys to int
            return {int(k): v for k, v in mapping.items()}
    else:
        print(f"Warning: class_mapping.json not found at {mapping_path}")
        return {0: "white", 1: "purple and black"}


def infer_image(image_path, model, processor, class_mapping, device):
    """Perform inference on a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Prepare dummy text (required by the model)
    dummy_text = "a basketball player"
    
    inputs = processor(
        images=image,
        text=dummy_text,
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    with torch.no_grad():
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()
    
    pred_label = class_mapping.get(pred_class, f"Class {pred_class}")
    
    return {
        "predicted_class": pred_label,
        "confidence": confidence,
        "all_probs": {
            class_mapping.get(i, f"Class {i}"): probs[0, i].item()
            for i in range(probs.shape[1])
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Inference with trained FashionCLIP model")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained model directory (e.g., fashionclip_trained/best_model)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to image for inference",
    )
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.isdir(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    
    # Check if image exists
    if not os.path.isfile(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return
    
    # Load model and processor
    print("Loading model...")
    model, processor, device = load_model_and_processor(args.model_dir)
    
    # Load class mapping
    class_mapping = load_class_mapping(args.model_dir)
    
    # Run inference
    print(f"Running inference on {args.image_path}...")
    result = infer_image(args.image_path, model, processor, class_mapping, device)
    
    if result:
        print(f"\nPredicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nProbabilities for all classes:")
        for class_name, prob in result['all_probs'].items():
            print(f"  {class_name}: {prob:.4f}")
    else:
        print("Inference failed")


if __name__ == "__main__":
    main()
