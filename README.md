# Motion Deblurring: Traditional Methods and Deep Learning

Computer vision project by Keyu Chen (kchen190), Yukun Zhong (yzhong54), and Siyang Qin (sqin12).

## Project goal

Explore whether combining physics-based deconvolution with a neural refinement network can improve motion deblurring across different blur strengths.

Planned methods include Wiener deconvolution, Richardson-Lucy deconvolution, a pretrained NAFNet reference, and a hybrid model that takes both the blurry image and a traditional restoration as input.

**Current implementation:** dataset inspection and evaluation of the original blurry inputs. Deblurring methods and model training are planned work.

## Dataset and download

We use the GoPro dataset introduced by Nah et al. (CVPR 2017).

- [Official dataset page](https://seungjunnah.github.io/Datasets/gopro.html)
- [Download GOPRO_Large.zip: paired blurry and sharp images](https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip)
- [Download page and file information](https://huggingface.co/datasets/snah/GOPRO_Large/blob/main/GOPRO_Large.zip)

**Download `GOPRO_Large.zip` for the current scripts.** The download is approximately 9.54 GB. `GOPRO_Large_all.zip` contains all sharp frames used to generate blur; it is a different version and is not directly compatible with the paired-data workflow below.

| Split | Sequences | Blur/sharp pairs |
| --- | ---: | ---: |
| Training | 22 | 2,103 |
| Test | 11 | 1,111 |
| Total | 33 | 3,214 |

Images have a resolution of 1280 x 720. Extract the archive so the directory layout is:

```text
GOPRO_Large/
  train/
    GOPR0372_07_00/
      blur/
        000047.png
      blur_gamma/
        000047.png
      sharp/
        000047.png
    ...
  test/
    GOPR0384_11_00/
      blur/
        000001.png
      blur_gamma/
        000001.png
      sharp/
        000001.png
    ...
```

The current scripts use `blur` as input and the matching file in `sharp` as ground truth. They do not use `blur_gamma`.

Dataset files are excluded from Git. Download them separately. The dataset is released under CC BY 4.0; see the official page for licensing and attribution.

## Setup

Clone the repository and enter its directory:

```powershell
git clone https://github.com/jingluotan/Computer-Vision.git
cd Computer-Vision
```

Create a virtual environment on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pillow matplotlib numpy scikit-image
```

Set `DATA_ROOT` in **both** `check_dataset.py` and `evaluate_baseline.py` to your extracted dataset directory. The current scripts contain a machine-specific path, which must be changed if your dataset is elsewhere. For example:

```python
DATA_ROOT = Path(r"D:\datasets\GOPRO_Large")
```

## Step 1: Inspect dataset pairs

**Purpose:** confirm image pairing and visually inspect representative blurry/sharp images before running experiments.

In `check_dataset.py`, choose the split:

```python
SPLIT = "test"  # Use "train" to inspect training data.
```

Run:

```powershell
python check_dataset.py
```

The script prints the number of pairs, opens up to four randomly selected pairs, checks their dimensions, and saves `test_pairs.png` or `train_pairs.png` beside the script. The random seed is fixed for repeatable sampling. It does not open and validate every image.

## Step 2: Evaluate the original blurry inputs

**Purpose:** establish reference PSNR and SSIM scores before applying any deblurring method.

Run:

```powershell
python evaluate_baseline.py
```

The script compares all 1,111 blurry test images against their sharp references and saves:

```text
results/
  baseline_test_metrics.csv
  baseline_test_summary.txt
```

Evaluation settings:

- RGB images with pixel values scaled to [0, 1].
- No border cropping.
- PSNR over RGB pixels, reported in dB.
- SSIM computed per channel with a uniform 7 x 7 window and sample covariance.
- Dataset scores are the arithmetic means of per-image scores.

### Recorded baseline

| Method | Mean PSNR (dB) | Mean SSIM |
| --- | ---: | ---: |
| Original blurry input | 25.640138 | 0.779467 |

These are scores for the unprocessed inputs, not a deblurring model. Higher scores indicate closer agreement with the reference under these metrics. Use the same settings when comparing later methods; published results may use different evaluation protocols.

## Makefile shortcuts

GNU Make must be installed separately. On Windows, it can be installed with:

```powershell
winget install --id ezwinports.make --exact --source winget
```

Restart VS Code after installation and check `make --version`. Run the tasks from the repository directory:

```powershell
make
make check
make evaluate
```

Each experiment target installs its dependencies before running its Python script. The Makefile defaults to the Windows `py` launcher. To explicitly use the active virtual environment's Python:

```powershell
make check PYTHON=python
make evaluate PYTHON=python
```

## Files

| File | Purpose |
| --- | --- |
| `check_dataset.py` | Discover blur/sharp pairs and visualize sampled images. |
| `evaluate_baseline.py` | Compute per-image and mean PSNR/SSIM for blurry test inputs. |
| `Makefile` | Provide named commands for the two steps. |
| `.gitignore` | Exclude datasets, virtual environments, caches, and generated outputs. |

## Next experiments

1. Generate controlled motion blur from training sharp images using known kernels.
2. Evaluate Wiener and Richardson-Lucy deconvolution on synthetic examples.
3. Run pretrained NAFNet on the GoPro test set.
4. Train and evaluate a hybrid refinement model with appropriate ablations.

Reserve a validation split from training sequences for tuning. Keep the test split separate from training and parameter selection.

## Dataset reference

S. Nah, T. H. Kim, and K. M. Lee, "Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring," CVPR, 2017, pp. 3883-3891.

[Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.html)
