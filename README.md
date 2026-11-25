# Which Layer to Read? Adaptive Depth for Training-Free Few-Shot Open-Set Recognition

This repository contains a training-free method for few-shot open-set recognition (FSOSR): each episode gives a few
labelled images of a handful of known classes, and the model must classify queries from those classes while rejecting
queries from classes it has never seen. Methods built on frozen foundation models usually read features from the last
layer, but the best depth depends on the dataset and on the backbone. Our method selects the layer automatically and
works on a frozen DINOv2 ViT-B/14 with registers, with no gradients and no per-dataset tuning.

> This work has been submitted to the **European Symposium on Artificial Neural Networks (ESANN 2027)**.

## Highlights

- **The last layer is not always the right one**
  With DINOv2, reading the last layer loses up to 25 accuracy points on EuroSAT and 11 on Omniglot against the best layer.

- **Depth selection without query labels**
  A round-trip consistency test chooses the layer, either in every episode from its supports alone, or once per dataset
  from data not used at test.

- **Two complementary experts**
  A patch-matching expert at the selected layer and OSLO on the selected and last layers, fused with no tuned weights.

- **Training-free**
  A frozen backbone, no gradients, no fine-tuning and no per-dataset tuning.

- **Best average results on four benchmarks**
  Large gains far from the pretraining domain (EuroSAT, Omniglot), and on par or better close to it (CIFAR-FS, DTD).

## Setup

1) Clone the repository
    ```bash
    git clone https://github.com/DanielVilaCruz/adaptive-depth-fsosr.git
    cd which-layer-fsosr
    ```
2) Install PyTorch

   This project requires PyTorch (with CUDA or CPU support).

   Example (CUDA 12.1):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   Example (CPU):
   ```bash
   pip install torch torchvision
   ```
3) Install project dependencies
    ```bash
    pip install -r requirements.txt
    ```
4) Download the datasets and place them under the `data/` folder, with one folder of images per class.
   Supported benchmarks:
    - CIFAR-FS (standard split: `train/` and `test/` with disjoint classes)
    - Omniglot (background alphabets as `train/`, evaluation alphabets as `test/`)
    - DTD
    - EuroSAT

   ```
   data/cifar_fs/{train,test}/<class>/*
   data/omniglot/{train,test}/<class>/*
   data/dtd/<class>/*
   data/eurosat/<class>/*
   ```
5) Hold out data to choose the depth (DTD and EuroSAT only)

   CIFAR-FS and Omniglot choose the depth on their training classes. DTD and EuroSAT have no training classes, so
   `make_holdout.py` splits them into a held-out part (`train/`, used only to choose the depth) and the test part
   (`test/`, where the episodes are drawn). For DTD, 17 of the 47 classes are held out, so the depth is chosen on
   classes disjoint from the test ones, as for CIFAR-FS and Omniglot. EuroSAT has only 10 classes, so 5% of the images of
   each class are held out instead:
   ```bash
   python make_holdout.py --src data/dtd --out data/dtd_ho --classes 17
   python make_holdout.py --src data/eurosat --out data/eurosat_ho --frac 0.05
   ```

## Run Experiments

```bash
python fsosr.py --data_root data/<dataset> [options]
```

Main parameters:

| parameter | default | description |
|---|---|---|
| `--data_root` | (required) | one or more dataset folders; episodes are drawn from `test/` if present |
| `--src` | `auto` | how the layer is chosen: `auto` (round trip in every episode), `dataset` (once per dataset, from `--depth_root`) or a fixed layer number, e.g. `12` |
| `--depth_root` | | with `--src dataset`: the data used to choose the layer, one folder per `--data_root` (e.g. `data/cifar_fs/train`) |
| `--depth_same_classes` | off | with `--src dataset`: allow `--depth_root` to share classes with the test set |
| `--depth_episodes` | `50` | with `--src dataset`: episodes used to choose the layer |
| `--src_candidates` | `4-12` | candidate layers |
| `--configs` | `5w1s,5w5s` | episode settings, `<N>w<K>s` (N known classes, K shots) |
| `--n_query` | `10` | queries per known class |
| `--n_unknown_classes` | `5` | unknown classes per episode |
| `--n_unknown_query` | `5` | queries per unknown class |
| `--episodes` | `300` | test episodes per setting |
| `--baselines` | `maxlogit,knn,ostim,oslo` | baselines run on the same episodes (`""` for none) |
| `--bl_layers` | `12` | layers the baselines read, e.g. `12` or `8,12` |
| `--lsel_baselines` | off | also run the baselines at the selected layer |
| `--seed` | `0` | the same seed gives the same episodes for every method |
| `--out_dir` | `results` | where `summary.json` and `table.md` are written |


## Benchmark results

Average over CIFAR-FS, Omniglot, DTD and EuroSAT (mean ± 95% CI over 300 episodes):

| Method | Accuracy 1-shot | Accuracy 5-shot | OSCR 1-shot | OSCR 5-shot |
|---|---|---|---|---|
| MaxLogit | 76.0±0.6 | 88.8±0.4 | 63.0±0.6 | 79.0±0.4 |
| kNN | 76.0±0.6 | 88.2±0.4 | 63.0±0.6 | 64.9±0.6 |
| OSTIM-EM | 80.2±0.6 | 88.3±0.4 | 64.9±0.8 | 75.8±0.6 |
| OSLO | 81.4±0.6 | 89.2±0.4 | 68.2±0.7 | 81.3±0.5 |
| Ours, depth per episode | 85.6±0.6 | 93.4±0.3 | 72.1±0.7 | 87.0±0.4 |
| **Ours, depth per dataset** | **87.7±0.5** | **94.0±0.3** | **75.1±0.6** | **88.3±0.4** |
