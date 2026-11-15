import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from features_extraction import prepare_features4train, MultiFeatureDataset, MLPRegressor

def main(csv_path=None, checkpoint_save_path=None, results_path=None,
         cls_tensor=None, device=None, data_mode="split", train_csv_path=None,
         test_csv_path=None, train_cls_tensor=None, test_cls_tensor=None):
    """Train a CCS regression model and evaluate its best validation weights.

    Args:
        csv_path: Input CSV path for split mode. Required columns are
            'smiles', 'Adduct', and 'CCS_AVG'.
        checkpoint_save_path: Output path for the model checkpoint.
        results_path: Output CSV path for test predictions and targets.
        cls_tensor: Path to precomputed CLS representations for csv_path.
        device: PyTorch device or device name used for tensors and the model.
        data_mode: 'split' to divide one CSV, or 'pre_split' to use separate
            training and test CSV files.
        train_csv_path: Training CSV path for pre_split mode. This dataset is
            further divided into training and validation subsets.
        test_csv_path: Test CSV path for pre_split mode.
        train_cls_tensor: Precomputed CLS representation path for train_csv_path.
        test_cls_tensor: Precomputed CLS representation path for test_csv_path.

    CLS representations must follow the row order of their corresponding CSV.
    The parent directory of checkpoint_save_path must already exist.
    """
    device = torch.device(device)
    required_cols = ['smiles', 'Adduct', 'CCS_AVG']

    # Keep this element order consistent with the feature-vector construction.
    elements = ["H", "Li", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Se", "Br", "I"]

    def load_csv(path):
        """Load a CSV and check that all required input and target columns exist."""
        df = pd.read_csv(path)
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in CSV: {path}")
        return df

    # Split rows while retaining their original indices. MultiFeatureDataset
    # uses these indices to retrieve aligned features and targets.
    if data_mode == "split":
        if csv_path is None or cls_tensor is None:
            raise ValueError("Split mode requires csv_path and cls_tensor.")
        df = load_csv(csv_path)

        # Reserve 20% for testing, then use 10% of the remaining 80% for validation.
        # The resulting train/validation/test proportions are about 72%/8%/20%.
        train_val, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        train, val = train_test_split(train_val, test_size=0.1, random_state=42, shuffle=True)

        # Build features in the full CSV's original row order, then select subsets.
        # CLS representations must use this same order, rather than a split order.
        features = prepare_features4train(df, elements, device, cls_tensor)
        train_dataset = MultiFeatureDataset(list(train.index), *features)
        val_dataset = MultiFeatureDataset(list(val.index), *features)
        test_dataset = MultiFeatureDataset(list(test.index), *features)
    elif data_mode == "pre_split":
        if any(value is None for value in (train_csv_path, test_csv_path,
                                          train_cls_tensor, test_cls_tensor)):
            raise ValueError("Pre-split mode requires training/test CSV and CLS representation paths.")
        train_val = load_csv(train_csv_path)
        test = load_csv(test_csv_path)

        # Use 90%/10% of the supplied training CSV for training and validation.
        # Keep the separately supplied test dataset for the final evaluation.
        train, val = train_test_split(train_val, test_size=0.1, random_state=42, shuffle=True)

        # Each CSV has its own row indices and matching CLS representation file.
        # Training and validation share one feature set; testing uses another.
        train_features = prepare_features4train(train_val, elements, device, train_cls_tensor)
        test_features = prepare_features4train(test, elements, device, test_cls_tensor)
        train_dataset = MultiFeatureDataset(list(train.index), *train_features)
        val_dataset = MultiFeatureDataset(list(val.index), *train_features)
        test_dataset = MultiFeatureDataset(list(test.index), *test_features)
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")

    # Shuffle training samples each epoch; preserve validation and test order.
    # prepare_features4train has already placed all tensors on the chosen device.
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # The regressor combines CLS, fingerprint, element-count, and adduct features.
    model = MLPRegressor().to(device)

    # Huber loss is quadratic for small residuals and linear for large residuals.
    # Adam updates the parameters, with weight decay providing regularization.
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Reduce the learning rate when validation loss stops improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # Train for at most 500 epochs, stopping after 15 consecutive epochs without
    # a strictly lower validation loss. The scheduler tracks its own improvements.
    max_iter = 500
    best_loss = float('inf')
    patience = 15
    counter = 0
    best_model_state = None

    for epoch in range(max_iter):
        # Enable training behavior, including dropout, and update model parameters.
        model.train()
        train_loss = 0.0
        for batch_dof, batch_fp, batch_fe, batch_adduct, batch_y in train_loader:
            # batch_dof contains CLS representations. Flatten predictions from
            # (batch_size, 1) to (batch_size,) to match the target shape.
            optimizer.zero_grad()
            outputs = model(batch_dof, batch_fp, batch_fe, batch_adduct)
            loss = criterion(outputs.reshape(-1), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Disable dropout and gradient tracking during validation.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_dof, batch_fp, batch_fe, batch_adduct, batch_y in val_loader:
                outputs = model(batch_dof, batch_fp, batch_fe, batch_adduct)
                val_loss += criterion(outputs.reshape(-1), batch_y).item()

        # Use the mean batch loss for scheduling and early stopping. Each batch
        # has equal weight here, including a potentially smaller final batch.
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0

            # Deep-copy the tensors so subsequent updates cannot modify the
            # saved parameters from the best validation epoch.
            best_model_state = model.state_dict().copy()
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Epoch numbers are zero-based. Log the mean training and validation
        # batch losses periodically rather than printing every epoch.
        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

    # Restore the best validation weights before saving and testing the model.
    model.load_state_dict(best_model_state)

    # Model weights and loss correspond to the best validation epoch. The epoch
    # counter and optimizer state correspond to the final training epoch.
    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
    }, checkpoint_save_path)

    # Evaluate the selected model on the test set without updating its parameters.
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch_dof, batch_fp, batch_fe, batch_addt, batch_y in test_loader:
            outputs = model(batch_dof, batch_fp, batch_fe, batch_addt)
            test_preds.extend(outputs.reshape(-1).cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())

    # Convert all batches to NumPy arrays for evaluation with scikit-learn.
    test_targets = np.array(test_targets)
    test_preds = np.array(test_preds)

    # Report MSE, MAE, R2, and median absolute and relative errors.
    mse = mean_squared_error(test_targets, test_preds)
    mae = mean_absolute_error(test_targets, test_preds)
    r2 = r2_score(test_targets, test_preds)
    median_ae = median_absolute_error(test_targets, test_preds)

    # Relative errors are fractions, not percentages, and require nonzero targets.
    relative_errors = np.abs(test_targets - test_preds) / np.abs(test_targets)
    median_re = np.median(relative_errors)

    print("\nTest Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Median Absolute Error: {median_ae:.4f}")
    print(f"Median Relative Error: {median_re:.4f}")

    # Predictions and targets remain paired in test-loader order. In split mode,
    # this is the order produced by train_test_split, not the original CSV order.
    results = pd.DataFrame({"preds": test_preds, "targets": test_targets})
    results.to_csv(results_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a CCS regression model")
    parser.add_argument("--data_mode", choices=["split", "pre_split"], default="split",
                        help="Split one dataset internally (default), or use separate training and test datasets")
    parser.add_argument("--csv_path", type=str,
                        help="Path to the input CSV with 'smiles', 'Adduct', and 'CCS_AVG' columns (split mode)")
    parser.add_argument("--checkpoint_save_path", type=str, default="./ckpt/weights.pth",
                        help="Output path for the model checkpoint")
    parser.add_argument("--results_path", type=str, default="./results.csv",
                        help="Output CSV path for test predictions and targets")
    parser.add_argument("--cls_tensor", type=str,
                        help="Path to precomputed CLS representations (.pt, .pth, .ckpt, or .csv; split mode)")
    parser.add_argument("--train_csv_path", type=str,
                        help="Path to the training CSV with 'smiles', 'Adduct', and 'CCS_AVG' columns (pre_split mode)")
    parser.add_argument("--test_csv_path", type=str,
                        help="Path to the test CSV with 'smiles', 'Adduct', and 'CCS_AVG' columns (pre_split mode)")
    parser.add_argument("--train_cls_tensor", type=str,
                        help="Path to precomputed CLS representations matching the training CSV row order (pre_split mode)")
    parser.add_argument("--test_cls_tensor", type=str,
                        help="Path to precomputed CLS representations matching the test CSV row order (pre_split mode)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device used for computation (e.g., 'cuda:0' or 'cpu')")

    args = parser.parse_args()

    # Validate mode-specific inputs before reading files or building features.
    required = ("csv_path", "cls_tensor") if args.data_mode == "split" else (
        "train_csv_path", "test_csv_path", "train_cls_tensor", "test_cls_tensor"
    )
    missing = [f"--{name}" for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"--data_mode {args.data_mode} requires: {', '.join(missing)}")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Set random seeds for model initialization, dropout, and training shuffling.
    # train_test_split uses its own explicit random_state=42 above.
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # Forward parsed arguments to the shared training and evaluation pipeline.
    main(
        csv_path=args.csv_path,
        checkpoint_save_path=args.checkpoint_save_path,
        results_path=args.results_path,
        cls_tensor=args.cls_tensor,
        device=device,
        data_mode=args.data_mode,
        train_csv_path=args.train_csv_path,
        test_csv_path=args.test_csv_path,
        train_cls_tensor=args.train_cls_tensor,
        test_cls_tensor=args.test_cls_tensor,
    )
