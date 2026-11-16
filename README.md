# 🔍 UEFCCS
UEFCCS was developed for collision cross section (CCS) prediction

<img src="./images/ToC_.png" alt="Logo" width="600">

## 📁 Git clone repository

    git clone https://github.com/ypsh1/UEFCCS.git
    cd UEFCCS

## 📦 Set up the environment

The original README specifies Python 3.10, PyTorch 2.9.1, and CUDA 12.8. The scripts import NumPy, pandas, scikit-learn, RDKit, and `unimol_tools` in addition to PyTorch.

```bash
conda env create -f environment.yml
conda activate uefccs
```

`predict.py` and `train.py` default to `--device cuda:0`; use `--device cpu` for their tensors and MLP when CUDA is unavailable.

## 📁 Data and model weights

CSV column names are case-sensitive:

| Script | Required columns | Optional columns |
| --- | --- | --- |
| `finetune.py` | `smiles`, `Adduct`, `CCS_AVG` | - |
| `extract_CLS_token.py` | `smiles` | - |
| `train.py` | `smiles`, `Adduct`, `CCS_AVG` | - |
| `predict.py` | `smiles`, `Adduct` | `CCS`: reference values used to evaluate predicted CCS |

Use valid, nonempty SMILES and numeric CCS labels when labels are provided. `CCS_AVG` is the training target; prediction evaluation reads `CCS`.

### Download precomputed [CLS] representations

Precomputed [CLS] representation files are also available on Hugging Face:

| File | Local path | Description |
| --- | --- | --- |
| [CLS.pt](https://huggingface.co/ypsh1/UEFCCS/resolve/main/CLS.pt) | `./data/CLS.pt` | Representations for the full dataset |
| [CLS_train.pt](https://huggingface.co/ypsh1/UEFCCS/resolve/main/CLS_train.pt) | `./data/CLS_train.pt` | Representations for the training subset |
| [CLS_test.pt](https://huggingface.co/ypsh1/UEFCCS/resolve/main/CLS_test.pt) | `./data/CLS_test.pt` | Representations for the test subset |

General usage:

```bash
curl -fL "<cls_download_url>" -o <local_cls_path>
```

e.g. from the repository root:

```bash

curl -fL "https://huggingface.co/ypsh1/UEFCCS/resolve/main/CLS.pt" \
    -o ./data/CLS.pt

curl -fL "https://huggingface.co/ypsh1/UEFCCS/resolve/main/CLS_train.pt" \
    -o ./data/CLS_train.pt

curl -fL "https://huggingface.co/ypsh1/UEFCCS/resolve/main/CLS_test.pt" \
    -o ./data/CLS_test.pt
```

Use each representation file with its corresponding CSV, preserving the exact row order. See the [current training interface limitation](#3-train-the-uefccs-regressor) before using these files with `train.py`.

For prediction, the supported adducts are `[M+H]`, `[M+Na]`, `[M-H]`, and `[M+K]`. Their charged forms (`[M+H]+`, `[M+Na]+`, `[M-H]-`, and `[M+K]+`) are normalized automatically. Rows with other adducts are excluded. Fine-tuning selects only rows whose `Adduct` is exactly `[M+H]`.    

Prediction requires two weight files:

| Weight file | Default path | Purpose |
| --- | --- | --- |
| UEFCCS regressor checkpoint | `./ckpt/weights.pth` | Loaded by `predict.py --ckpt`; must contain `model_state_dict` |
| Fine-tuned Uni-Mol checkpoint | `./finetune_unimol/model_2.pth` | Loaded by `predict.py --ft_path` to generate [CLS] representations |

### Download pretrained weights

The pretrained weights are available in the [UEFCCS model repository on Hugging Face](https://huggingface.co/ypsh1/UEFCCS). Download both files to the default paths listed above.

General usage:

```bash
curl -fL "<weight_download_url>" -o <local_checkpoint_path>
```

e.g. from the repository root:

curl -fL "https://huggingface.co/ypsh1/UEFCCS/resolve/main/weights.pth" \
    -o ./ckpt/weights.pth

curl -fL "https://huggingface.co/ypsh1/UEFCCS/resolve/main/model_2.pth" \
    -o ./finetune_unimol/model_2.pth
```

Use the file actually saved by your Uni-Mol run. Prediction should use the same Uni-Mol weights used to generate the regressor's training features.

## 🎯 Prediction

```bash
python predict.py \
    --input_csv <input.csv> \
    --output_csv <predictions.csv> \
    --ckpt <uefccs_checkpoint.pth> \
    --ft_path <unimol_checkpoint.pth> \
    --device <device> \
    --batch_size <batch_size>
```

e.g. using the default weight paths, device, and batch size:

```bash
python predict.py --input_csv ./data/test.csv --output_csv ./results.csv
```

## ⚡ Reproduction

For evaluation on an independent test set, prepare separate training and test CSV files first, and fine-tune Uni-Mol using only the training CSV.

In ccs_data, `METLIN_CCS_monomer_train.csv` and `METLIN_CCS_monomer_test.csv` were generated using `train_test_split(df, test_size=0.2, random_state=42, shuffle=True)` #
 df refers to METLIN_CCS_monomer.csv
 
### 1. Fine-tune Uni-Mol

Fine-tune Uni-Mol V1 on CCS values for the `[M+H]` subset. The script removes salts, canonicalizes SMILES, and excludes unparseable SMILES from the fine-tuning pairs. 

General usage:

```bash
python finetune.py \
    --csv <training.csv> \
    --save_path <unimol_output_directory> \
    --epochs <epochs> \
    --lr <learning_rate> \
    --batch_size <batch_size> \
    --early_stopping <patience>
```

e.g.:

```bash
python finetune.py \
    --csv ./data/METLIN_CCS_monomer_train.csv \
    --save_path ./finetune_unimol \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 64 \
    --early_stopping 10
```

All arguments above are optional; the example shows their defaults, so `python finetune.py` runs with the same settings.

### 2. Export [CLS] representations

Extract [CLS] representations using the fine-tuned Uni-Mol weights. Each saved file contains `{"cls_repr": tensor}` with shape `(N, 512)`. With `--split false`, rows follow the input CSV order.

General usage:

```bash
python extract_CLS_token.py \
    --csv <input.csv> \
    --UMpt_path <unimol_checkpoint.pth> \
    --split <true_or_false> \
    --out_pt <output.pt> \
    --test_size <test_fraction> \
    --seed <random_seed>
```

e.g. exporting representations for one CSV with splitting:

This generates ./data/CLS_train.pt and ./data/CLS_test.pt. The representations are split using 'train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)'. This assigns 80% of the samples to the training set and 20% to the test set.

```bash
python extract_CLS_token.py \
    --csv ./data/METLIN_CCS_monomer.csv \
    --UMpt_path ./finetune_unimol/model_2.pth \
    --split True \
    --out_pt ./data/CLS.pt
```

| Argument | Default | Description |
| --- | --- | --- |
| `--csv` | Required | CSV containing a `smiles` column |
| `--UMpt_path` | Required | Fine-tuned Uni-Mol checkpoint |
| `--split` | `false` | Save one file, or randomly split representations into train/test files |
| `--out_pt` | `./data/CLS.pt` | Output path, or base name when splitting |
| `--test_size` | `0.2` | Test fraction when splitting |
| `--seed` | `42` | Random seed for splitting |

With --split False --out_pt ./data/CLS.pt, the script saves all representations to ./data/CLS.pt without splitting. Separately prepared train/test CSV files need their own extraction runs with --split False to preserve row alignment

### 3. Train the UEFCCS regressor

**Current interface limitation:** `train.py` describes `--cls_tensor`, `--train_cls_tensor`, and `--test_cls_tensor` as paths to precomputed representations. 

#### Separate training and test CSV files (`pre_split`)

The training CSV is split into training/validation subsets (90%/10%, `random_state=42`); the test CSV is reserved for final evaluation. Use the same Uni-Mol checkpoint for both feature sets.

General usage:

```bash
python train.py \
    --data_mode pre_split \
    --train_csv_path <training.csv> \
    --test_csv_path <test.csv> \
    --train_cls_tensor <unimol_checkpoint.pth> \
    --test_cls_tensor <unimol_checkpoint.pth> \
    --checkpoint_save_path <uefccs_checkpoint.pth> \
    --results_path <test_predictions.csv> \
    --device <device>
```

e.g.:

```bash
python train.py \
    --data_mode pre_split \
    --train_csv_path ./data/METLIN_CCS_monomer_train.csv \
    --test_csv_path ./data/METLIN_CCS_monomer_test.csv \
    --train_cls_tensor ./data/CLS_train.pt \
    --test_cls_tensor ./data/CLS_test.pt \
    --checkpoint_save_path ./ckpt/weights.pth \
    --results_path ./results.csv \
    --device cuda:0
```

#### One CSV with an internal split (`split`)

This mode splits the CSV into approximately 72% training, 8% validation, and 20% test data, using `random_state=42` for both splits.

General usage:

```bash
python train.py \
    --data_mode split \
    --csv_path <input.csv> \
    --cls_tensor <unimol_checkpoint.pth> \
    --checkpoint_save_path <uefccs_checkpoint.pth> \
    --results_path <test_predictions.csv> \
    --device <device>
```

e.g.:

```bash
python train.py \
    --data_mode split \
    --csv_path ./data/METLIN_CCS_monomer.csv \
    --cls_tensor `./data/CLS.pt` \
    --checkpoint_save_path ./ckpt/weights.pth \
    --results_path ./results.csv \
    --device cuda:0
```

`--data_mode` defaults to `split`, `--checkpoint_save_path` to `./ckpt/weights.pth`, `--results_path` to `./results.csv`, and `--device` to `cuda:0`.

Training saves a checkpoint containing `model_state_dict` and a test-results CSV containing `preds` and `targets`. The script prints MSE, MAE, R², median absolute error, and median relative error. Relative error is reported as a fraction.

   
## 📖 Citation

If you use this code, please cite:

    @article{ypsh1_2026UEFCCS,
          title={}, 
          author={},
          journal={},
          year={2026}
    }

## License
MIT
