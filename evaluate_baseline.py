"""
evaluate_baseline

Purpose: 
    Establish restoration-free reference scores for later method comparisons.

Usage: 
    Compare each original blurry test image against its sharp ground truth,
    compute RGB PSNR and SSIM, and average the per-image scores.

Inputs: 
    All test/sequence/blur/*.png images and their matching sharp images.

Outputs: 
    results/baseline_test_metrics.csv (per-image scores) and
    results/baseline_test_summary.txt (mean scores and evaluation settings).

Dependency: 
    Imports find_pairs() from check_dataset.py.

Evaluation: 
    RGB values in [0, 1], no border crop, uniform 7x7 SSIM window,
    and sample covariance. Keep these settings consistent across methods.

Scope: 
    Evaluates the original inputs; does not train or apply a deblurring model.

Run: 
    py evaluate_baseline.py
"""

from pathlib import Path
import csv

import numpy as np
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from check_dataset import find_pairs


DATA_ROOT = Path(r"D:\google\CV\project\GOPRO_Large")
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def load_rgb(path):
    """Load an RGB image as float32 values in the range [0, 1]."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def main():
    pairs = find_pairs(DATA_ROOT / "test")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "baseline_test_metrics.csv"

    psnr_values = []
    ssim_values = []

    print(f"Evaluating {len(pairs)} test pairs...")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sequence", "filename", "psnr_db", "ssim"])

        for index, (blur_path, sharp_path) in enumerate(pairs, start=1):
            blur = load_rgb(blur_path)
            sharp = load_rgb(sharp_path)

            if blur.shape != sharp.shape:
                raise ValueError(f"Image shapes do not match: {blur_path}")

            # Compute PSNR over all RGB pixels.
            psnr = peak_signal_noise_ratio(
                sharp,
                blur,
                data_range=1.0,
            )

            # Compute SSIM per RGB channel using a uniform 7x7 window.
            ssim = structural_similarity(
                sharp,
                blur,
                data_range=1.0,
                channel_axis=-1,
                win_size=7,
                gaussian_weights=False,
                use_sample_covariance=True,
            )

            psnr_values.append(float(psnr))
            ssim_values.append(float(ssim))

            writer.writerow([
                blur_path.parent.parent.name,
                blur_path.name,
                f"{psnr:.6f}",
                f"{ssim:.6f}",
            ])

            if index % 50 == 0 or index == len(pairs):
                print(f"Processed {index}/{len(pairs)} pairs")

    # Average the per-image scores across the test set.
    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    summary = (
        "Method: Original blurry input\n"
        "Split: test\n"
        f"Image pairs: {len(pairs)}\n"
        "Input folder: blur\n"
        "Reference folder: sharp\n"
        "Color space: RGB\n"
        "Data range: [0, 1]\n"
        "Border crop: None\n"
        "SSIM: uniform 7x7 window, sample covariance\n"
        "Aggregation: Mean of per-image scores\n"
        f"Mean PSNR: {mean_psnr:.6f} dB\n"
        f"Mean SSIM: {mean_ssim:.6f}\n"
    )

    summary_path = OUTPUT_DIR / "baseline_test_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"Per-image metrics saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()