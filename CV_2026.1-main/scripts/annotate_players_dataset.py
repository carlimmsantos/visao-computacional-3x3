from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

WHITE_TEXT = "a basketball player wearing a white outfit"
PURPLE_BLACK_TEXT = "a basketball player wearing purple and black outfit"
DEFAULT_DATASET_DIR = Path("players_dataset")
DEFAULT_CSV_PATH = Path("players_dataset_annotations.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(dataset_dir: Path) -> list[Path]:
    """Collect all images from dataset directory recursively."""
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_existing_annotations(csv_path: Path) -> dict[str, str]:
    """Load existing annotations from CSV file."""
    if not csv_path.exists():
        return {}

    annotations: dict[str, str] = {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_path = (row.get("image_path") or "").strip()
                text = (row.get("text") or "").strip()
                if image_path and text:
                    annotations[image_path] = text
    except Exception as e:
        print(f"Warning: Could not load existing annotations: {e}")

    return annotations


def save_annotations(csv_path: Path, annotations: dict[str, str]) -> None:
    """Save annotations to CSV file with backup."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Create backup if file exists
    if csv_path.exists():
        backup_path = csv_path.with_suffix(".csv.bak")
        try:
            csv_path.rename(backup_path)
        except Exception:
            pass  # Continue even if backup fails

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "text"])
            for image_path, text in sorted(annotations.items()):
                writer.writerow([image_path, text])
        print(f"✓ Annotations saved to {csv_path}")
    except Exception as e:
        print(f"✗ Error saving annotations: {e}")


def resize_to_screen(image, max_width: int = 1400, max_height: int = 900):
    """Resize image to fit screen while maintaining aspect ratio."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def create_overlay(image, alpha: float = 0.3):
    """Create semi-transparent overlay for better text visibility."""
    overlay = image.copy()
    # Dark overlay at the top for text
    overlay[:200, :] = overlay[:200, :] * (1 - alpha) + np.array([0, 0, 0]) * alpha
    return overlay.astype(np.uint8)


def add_instructions(
    image,
    current_index: int,
    total: int,
    relative_path: str,
    selected_text: str | None = None,
    stats: dict | None = None,
):
    """Create fixed UI canvas independent from image size."""

    # ===== FIXED WINDOW SIZE =====
    canvas_width = 1600
    canvas_height = 1000

    sidebar_width = 420
    top_bar_height = 170

    # Background
    canvas = np.full((canvas_height, canvas_width, 3), 25, dtype=np.uint8)

    # ===== IMAGE AREA =====
    image_area_x = sidebar_width + 20
    image_area_y = top_bar_height + 20

    image_area_width = canvas_width - sidebar_width - 40
    image_area_height = canvas_height - top_bar_height - 40

    h, w = image.shape[:2]

    scale = min(
        image_area_width / w,
        image_area_height / h
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Center image
    x_offset = image_area_x + (image_area_width - new_w) // 2
    y_offset = image_area_y + (image_area_height - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # ===== TOP BAR =====
    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas_width, top_bar_height),
        (40, 40, 40),
        -1
    )

    title = "PLAYER UNIFORM ANNOTATION TOOL"

    cv2.putText(
        canvas,
        title,
        (40, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ===== PROGRESS BAR =====
    progress = (current_index + 1) / total

    bar_x = 40
    bar_y = 80
    bar_w = 700
    bar_h = 30

    cv2.rectangle(
        canvas,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (70, 70, 70),
        -1
    )

    cv2.rectangle(
        canvas,
        (bar_x, bar_y),
        (bar_x + int(bar_w * progress), bar_y + bar_h),
        (0, 220, 0),
        -1
    )

    cv2.putText(
        canvas,
        f"{current_index + 1}/{total} ({progress*100:.1f}%)",
        (bar_x, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ===== SIDEBAR =====
    cv2.rectangle(
        canvas,
        (0, top_bar_height),
        (sidebar_width, canvas_height),
        (35, 35, 35),
        -1
    )

    y = top_bar_height + 40

    bindings = [
        ("1 / W", "White Outfit", (0, 255, 255)),
        ("2 / P", "Purple/Black", (180, 0, 180)),
        ("ENTER", "Confirm", (0, 255, 0)),
        ("C", "Clear", (0, 165, 255)),
        ("B", "Previous", (255, 165, 0)),
        ("S", "Skip", (255, 165, 0)),
        ("Q / ESC", "Quit", (0, 0, 255)),
    ]

    for key_desc, action, color in bindings:

        cv2.putText(
            canvas,
            key_desc,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            action,
            (160, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 45

    # ===== STATUS =====
    y += 20

    if selected_text is not None:

        status_color = (
            (0, 255, 0)
            if "white" in selected_text.lower()
            else (180, 0, 180)
        )

        cv2.putText(
            canvas,
            "SELECTED:",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )

        y += 40

        cv2.putText(
            canvas,
            selected_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            canvas,
            "NO SELECTION",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
            cv2.LINE_AA
        )

    # ===== FILE NAME =====
    cv2.putText(
        canvas,
        relative_path[-90:],
        (800, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    # ===== STATS =====
    if stats:

        stats_text = (
            f"Completed: {stats.get('completed', 0)} | "
            f"White: {stats.get('white', 0)} | "
            f"Purple: {stats.get('purple', 0)}"
        )

        cv2.putText(
            canvas,
            stats_text,
            (800, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return canvas


def get_annotation_stats(annotations: dict[str, str]) -> dict:
    """Calculate annotation statistics."""
    white_count = sum(1 for text in annotations.values() if "white" in text.lower())
    purple_count = sum(1 for text in annotations.values() if "purple" in text.lower())
    return {
        "completed": len(annotations),
        "white": white_count,
        "purple": purple_count,
    }


def show_summary(csv_path: Path, annotations: dict[str, str], total_images: int) -> None:
    """Display final summary of annotations."""
    print("\n" + "="*60)
    print("📊 ANNOTATION SUMMARY")
    print("="*60)
    print(f"Total images in dataset: {total_images}")
    print(f"Annotated images: {len(annotations)}")
    print(f"Remaining images: {total_images - len(annotations)}")

    if annotations:
        white_count = sum(1 for text in annotations.values() if "white" in text.lower())
        purple_count = sum(1 for text in annotations.values() if "purple" in text.lower())
        print(f"\nWhite outfit: {white_count} images")
        print(f"Purple/Black outfit: {purple_count} images")

    print(f"\n💾 Annotations saved to: {csv_path}")
    print("="*60)


def annotate_dataset(dataset_dir: Path, csv_path: Path) -> None:
    """Main annotation function with improved UX."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    images = collect_images(dataset_dir)
    if not images:
        print(f"❌ No images found in {dataset_dir}")
        return

    print(f"📸 Found {len(images)} images in {dataset_dir}")

    annotations = load_existing_annotations(csv_path)

    # Determine which images still need annotations
    pending_indices = [
        i for i, image_path in enumerate(images)
        if image_path.as_posix() not in annotations
    ]

    if not pending_indices:
        print(f"✅ All {len(images)} images are already annotated!")
        show_summary(csv_path, annotations, len(images))
        return

    print(f"📝 Need to annotate {len(pending_indices)} images")
    print(f"💡 Use keys: 1/w=White, 2/p=Purple/Black, Enter=Confirm, c=Clear, b=Back, s=Skip, q=Quit\n")

    window_name = "Players Dataset Annotation Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    try:
        pos = 0
        total_pending = len(pending_indices)
        auto_save_counter = 0

        while 0 <= pos < total_pending:
            image_index = pending_indices[pos]
            image_path = images[image_index]

            # Load image with error handling
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠️ Skipping unreadable image: {image_path.name}")
                pos += 1
                continue

            relative_path = image_path.as_posix()
            resized_image = resize_to_screen(image)
            selected_text = None
            move_to_previous = False
            go_to_next = False

            # Get current stats
            stats = get_annotation_stats(annotations)

            while True:
                display_image = add_instructions(
                    resized_image,
                    pos,
                    total_pending,
                    relative_path,
                    selected_text=selected_text,
                    stats=stats,
                )
                cv2.imshow(window_name, display_image)
                key = cv2.waitKey(0) & 0xFF

                # White outfit
                if key in (ord("1"), ord("w"), ord("W")):
                    selected_text = WHITE_TEXT
                    print(f"✓ Selected: White outfit for {image_path.name}")
                    continue

                # Purple/black outfit
                if key in (ord("2"), ord("p"), ord("P")):
                    selected_text = PURPLE_BLACK_TEXT
                    print(f"✓ Selected: Purple/Black outfit for {image_path.name}")
                    continue

                # Confirm selection
                if key in (13, 10):  # Enter key
                    if selected_text is None:
                        print("⚠️ Please select an outfit first (press 1/w or 2/p)")
                        continue

                    annotations[relative_path] = selected_text
                    save_annotations(csv_path, annotations)
                    auto_save_counter += 1
                    go_to_next = True
                    print(f"✅ Confirmed: {image_path.name} → {selected_text}")
                    break

                # Clear selection
                if key in (ord("c"), ord("C")):
                    selected_text = None
                    print(f"🔄 Cleared selection for {image_path.name}")
                    continue

                # Skip
                if key in (ord("s"), ord("S")):
                    go_to_next = True
                    print(f"⏭️ Skipped: {image_path.name}")
                    break

                # Previous
                if key in (ord("b"), ord("B")):
                    move_to_previous = True
                    print(f"⬅️ Going back to previous image")
                    break

                # Quit
                if key in (ord("q"), ord("Q"), 27):  # q or Esc
                    print("\n🛑 User requested quit")
                    show_summary(csv_path, annotations, len(images))
                    return

            if move_to_previous:
                pos = max(0, pos - 1)
                continue

            if go_to_next:
                pos += 1

                # Auto-save every 10 annotations as additional safety
                if auto_save_counter >= 10:
                    save_annotations(csv_path, annotations)
                    auto_save_counter = 0

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        save_annotations(csv_path, annotations)
        show_summary(csv_path, annotations, len(images))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🏀 Interactive annotation tool for basketball player crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default directories
  %(prog)s --dataset-dir ./my_players         # Custom dataset directory
  %(prog)s --csv-path ./annotations.csv       # Custom CSV path
  %(prog)s --dataset-dir ./data --csv-path ./out.csv
        """
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory with player crop images (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Output CSV path with image_path and text columns (default: {DEFAULT_CSV_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        annotate_dataset(args.dataset_dir, args.csv_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
