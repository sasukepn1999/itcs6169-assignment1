# ITCS 6169/8169 Assignment 1: The CNN Challenge

16-class scene classification from 2,400 labelled training images with a CNN.
Report: `report.pdf` (2 pages). AI usage: [`AI_USAGE.md`](AI_USAGE.md).

## Results

| Model | Labelled data | Unlabelled data (FixMatch) | Validation (last-10) | **Test** |
|---|---|---|---|---|
| Starter TNet (64×64) | train split | none | 49.8% (best epoch) | 42.3% |
| EfficientNet-B0 from scratch | train split | none | 83.5 ± 0.5% | 81.8 ± 1.1% |
| EfficientNet-B0, ImageNet-pretrained (no FixMatch) | train split | none | 93.7 ± 0.5% | (not evaluated) |
| **Final recipe**: EfficientNet-B0, ImageNet-pretrained | train+val | test2 | (val used for training) | **93.1 ± 0.8%** |
| **Final submission model**: same, ImageNet-pretrained | train+val+test | test2 | n/a | n/a (no held-out data) |

All numbers are mean ± std over 3 seeds. The "test" column comes only from models that never saw the test
images (neither labels nor pixels). The final submission model also trains on the labelled test images,
as allowed by the instructor, so it has no test score; it produces the leaderboard predictions for `test2`.

## Final recipe

EfficientNet-B0 (torchvision), ImageNet-1k initialisation (`IMAGENET1K_V1`), all layers fine-tuned, new
16-way linear head; 200×200 grayscale replicated to 3 channels (ImageNet normalisation); augmentation:
random-resized crop (scale 0.5–1), reflect-pad translation (size/8), horizontal flip; Adam, peak lr 1e-4 (selected on validation),
per-iteration cosine decay to 0, batch 64, 200 epochs, cross-entropy, bf16 autocast; FixMatch on the
unlabelled images (confidence threshold 0.95, λ_u = 1, RandAugment(2, 10) + Cutout as the strong view,
64 labelled + 64 unlabelled images per step); final-epoch weights.

## Setup

```bash
conda create -n cv-hw1 python=3.11 -y && conda activate cv-hw1
pip install -r requirements.txt
```

Download the dataset from the link in the assignment PDF and place it as

```text
data/train/<class>/*.jpg   (2,400 images, 16 classes x 150)
data/test/<class>/*.jpg    (400 images, 16 classes x 25)
data/test2/*.jpg           (400 unlabelled images)
```

## Reproduce

```bash
# final submission model (3 seeds) and test2 predictions
python train.py --config configs/final.json
python predict.py --checkpoint results/runs/final_recipe/6c8865075b_s0.pt --images data/test2 --out predictions_test2.csv

# clean test estimate of the final recipe, and the from-scratch model
python train.py --config configs/final_clean_test.json
python train.py --config configs/scratch_effnet.json
python evaluate.py --checkpoint results/runs/final_recipe/440e59840c_s{0,1,2}.pt
```

`train.py` prints the checkpoint paths. Every experiment in the report can be re-run from the notebook
`ITCS_6169_8169_Assignment1_2026_Starter.ipynb` (one section per experiment). Finished runs are cached in
`results/runs/`, so executing the notebook end to end only trains what is missing. The executed copy with
all outputs is `Starter_executed.ipynb`.

Seeds are fixed (data split: seed 0; training seeds 0, 1, 2) and training is deterministic for a given
GPU type: re-running a configuration reproduces its accuracy curve exactly.

## Leaderboard predictions

`predictions_test2.csv` contains the final model's predictions (seed 0) for the 400 `test2` images;
`predictions_test2_ensemble.csv` averages the 3 seeds (they agree on 97.75% of the images).

## Checkpoints

The checkpoints (`*.pt`, about 16 MB each, with weights and the full training config) are attached to the
GitHub release **TBD-link**. Download them into `results/runs/<experiment>/` to use `evaluate.py` and
`predict.py` without retraining.

**File names.** `<config-id>_s<seed>.pt`: the first part is a short hash of the training configuration
(see `configs/` and the `config` stored inside each file); `_s0`, `_s1`, `_s2` are **three independent
training runs of the same configuration with random seeds 0, 1 and 2**. The seed changes only the random
initialisation of the new layers, the order of the training data, the augmentation draws and FixMatch's
sampling. All accuracies are reported as mean ± std over these 3 runs. Each `.pt` has a `.json` next to
it with that run's training curves and validation predictions.

| File | Model | Labelled training data | Unlabelled (FixMatch) | Result |
|---|---|---|---|---|
| `final_recipe/6c8865075b_s{0,1,2}.pt` | **Submitted model** (final recipe) | train + val + test (2,800) | test2 | produces `predictions_test2*.csv`; no score (no labelled data left) |
| `final_recipe/440e59840c_s{0,1,2}.pt` | **Clean-test model** (same recipe) | train + val (2,400) | test2 | **93.1 ± 0.8% test**, the accuracy in the report |
| `final/a94cf549e2_s{0,1,2}.pt` | EfficientNet-B0 from scratch (Exp. 4) | train split (1,920) | none | 83.5 ± 0.5% val (last-10), 81.8 ± 1.1% test |
| `final/f44392305b_s{0,1,2}.pt` | from scratch + rotation SSL (Exp. 6) | train split (1,920) | none (SSL on train images) | 84.4 ± 0.9% val (last-10), 81.9 ± 0.2% test |

The submitted and clean-test models use the identical recipe (ImageNet-pretrained EfficientNet-B0, lr 1e-4,
FixMatch on test2); they differ only in whether the labelled test images are part of training. The test set
cannot be used both for training and for measuring accuracy, so the clean-test models give the honest
accuracy and the submitted model uses all labels for the leaderboard. `predictions_test2.csv` comes from the
submitted model's seed 0, and `predictions_test2_ensemble.csv` averages its three seeds.

## Files

| File | Purpose |
|---|---|
| `cvexp.py` | experiment library: data, augmentation, models, schedules, SSL/FixMatch, training loop, parallel multi-GPU runner |
| `ssl_aux.py` | self-supervised auxiliary tasks (rotation, SimCLR, SimSiam, masked image modelling) |
| `train.py` | train one configuration from `configs/*.json` and save checkpoints |
| `evaluate.py` | accuracy and per-class accuracy of checkpoints on test (or validation) |
| `predict.py` | predictions for an unlabelled folder (e.g. `test2`) as CSV; several checkpoints are ensembled |
| `configs/` | configurations of the final and reference models |
| `ITCS_6169_8169_Assignment1_2026_Starter.ipynb` | the starter notebook extended with every experiment (Experiments 1–9) |
| `analysis/` | data analysis (`eda.py`) and the stand-alone starter baseline (`baseline_starter.py`) |
| `results/` | per-experiment summaries (`*.json`) and per-run histories (`runs/`) |

## Environment

Python 3.11.17, PyTorch 2.14.1 + torchvision 0.29.1 (CUDA 13.0), NVIDIA H200 (driver 580.173.02).
See `requirements.txt`.
