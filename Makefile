# Motion deblurring project tasks
# Usage: make check | make evaluate
# Override the Python launcher if needed: make check PYTHON=python
# Recipe lines must start with a tab.

PYTHON = py
.DEFAULT_GOAL := help
.PHONY: help check evaluate

# Display available tasks without running experiments.
help:
	@echo make check     - Step 1: inspect paired dataset images
	@echo make evaluate  - Step 2: compute baseline PSNR and SSIM

# Step 1: verify dataset loading and inspect sampled blur/sharp pairs.
# Output: train_pairs.png or test_pairs.png, selected in check_dataset.py.
check:
	$(PYTHON) -m pip install pillow matplotlib
	$(PYTHON) check_dataset.py

# Step 2: evaluate original blurry test images against sharp references.
# Outputs: results/baseline_test_metrics.csv and baseline_test_summary.txt.
# Matplotlib is required because evaluate_baseline imports check_dataset.
evaluate:
	$(PYTHON) -m pip install pillow matplotlib numpy scikit-image
	$(PYTHON) evaluate_baseline.py
