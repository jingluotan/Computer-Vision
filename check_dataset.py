"""
check_dataset.py

Purpose:
    Verify that the GoPro dataset can be loaded and that blurry images
    have matching sharp reference images before starting experiments.

Main tasks:
    1. Find blur/sharp image pairs by sequence and filename.
    2. Randomly select up to four pairs using a fixed random seed.
    3. Check that the selected images have matching dimensions.
    4. Display blurry and sharp images side by side for visual inspection.
    5. Save the comparison figure.

Inputs:
    Paired images from the train or test split under DATA_ROOT.
    Use SPLIT to select which split to inspect.

Outputs:
    - Image pair count and sampled image dimensions in the terminal.
    - A side-by-side comparison window.
    - train_pairs.png or test_pairs.png beside this script.

Functions:
    find_pairs(): Find matching blur/sharp image paths.
    main(): Load sample pairs, check dimensions, and create the figure.

Limitations:
    Only sampled images are opened and checked for matching dimensions.
    This module does not perform deblurring, training, or metric evaluation.

Run:
    py check_dataset.py
"""


from pathlib import Path
import random


from PIL import Image
import matplotlib.pyplot as plt


# Dataset path
DATA_ROOT = Path(r"D:\google\CV\project\GOPRO_Large")
SPLIT = "test"  # Choose "train" or "test".


def find_pairs(split_dir):
    """Find matching blur/sharp image pairs by sequence and filename."""
    pairs = []

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {split_dir}")

    for sequence_dir in sorted(split_dir.iterdir()):
        if not sequence_dir.is_dir():
            continue

        for blur_path in sorted((sequence_dir / "blur").glob("*.png")):
            sharp_path = sequence_dir / "sharp" / blur_path.name

            if not sharp_path.is_file():
                raise FileNotFoundError(f"Missing sharp image: {sharp_path}")

            pairs.append((blur_path, sharp_path))

    if not pairs:
        raise ValueError(f"No image pairs found in: {split_dir}")

    return pairs


def main():
    pairs = find_pairs(DATA_ROOT / SPLIT)
    print(f"Found {len(pairs)} image pairs in the {SPLIT} split")

    # Use a fixed random seed to select the same samples on each run.
    samples = random.Random(42).sample(pairs, k=min(4, len(pairs)))

    fig, axes = plt.subplots(
        len(samples), 2,
        figsize=(14, 4 * len(samples)),
        squeeze=False,
    )

    for row, (blur_path, sharp_path) in enumerate(samples):
        with Image.open(blur_path) as image:
            blur = image.convert("RGB")

        with Image.open(sharp_path) as image:
            sharp = image.convert("RGB")

        if blur.size != sharp.size:
            raise ValueError(f"Image dimensions do not match: {blur_path}")

        print(
            f"{blur_path.parent.parent.name}/{blur_path.name}"
            f" | Size: {blur.size}"
        )

        axes[row, 0].imshow(blur)
        axes[row, 0].set_title(
            f"Blur: {blur_path.parent.parent.name}/{blur_path.name}"
        )

        axes[row, 1].imshow(sharp)
        axes[row, 1].set_title(f"Sharp: {sharp_path.name}")

        for axis in axes[row]:
            axis.axis("off")

    fig.tight_layout()

    # Save the comparison figure in the same directory as this script.
    output_path = Path(__file__).resolve().parent / f"{SPLIT}_pairs.png"
    fig.savefig(output_path, dpi=150)
    print(f"Comparison figure saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()