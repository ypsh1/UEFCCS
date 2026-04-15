import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from features import prepare_features4train, MultiFeatureDataset, MLPRegressor

def main(csv_path, checkpoint_save_path, cls_checkpoint=None, device=None):
    device = torch.device(device)
    
    df = pd.read_csv(csv_path)
    required_cols = ['smiles', 'Adduct', 'CCS_AVG', 'Molecular Formula']
    for col in required_cols:
        if col not in df_all.columns:
            raise KeyError(f"Column '{col}' not found in CSV")

    train_val, test = train_test_split(df_all, test_size=0.2, random_state=42, shuffle=True)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42, shuffle=True)

    train_indices = list(train.index)
    val_indices   = list(val.index)
    test_indices  = list(test.index)

    elements = ["H", "Li", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Se", "Br", "I"]

    ion_repr, fp_features, vec_features, adduct_features, y = prepare_features4train(
        df, elements, device, unirepr_checkpoint
    )

    train_dataset = MultiFeatureDataset(train_indices, ion_repr, fp_features, vec_features, adduct_features, y)
    val_dataset   = MultiFeatureDataset(val_indices,   ion_repr, fp_features, vec_features, adduct_features, y)
    test_dataset  = MultiFeatureDataset(test_indices,  ion_repr, fp_features, vec_features, adduct_features, y)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    model = MLPRegressor().to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    max_iter = 500
    best_loss = float('inf')
    patience = 15
    counter = 0
    best_model_state = None

    for epoch in range(max_iter):
        model.train()
        train_loss = 0.0
        for batch_dof, batch_fp, batch_vec, batch_adduct, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_dof, batch_fp, batch_vec, batch_adduct)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_dof, batch_fp, batch_vec, batch_adduct, batch_y in val_loader:
                outputs = model(batch_dof, batch_fp, batch_vec, batch_adduct)
                val_loss += criterion(outputs.squeeze(), batch_y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

    model.load_state_dict(best_model_state)

    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
    }, checkpoint_save_path)

    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch_dof, batch_fp, batch_vec, batch_addt, batch_y in test_loader:
            outputs = model(batch_dof, batch_fp, batch_vec, batch_addt)
            test_preds.extend(outputs.squeeze().cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())

    test_targets = np.array(test_targets)
    test_preds = np.array(test_preds)
    test_preds = np.round(test_preds, 2)

    mse = mean_squared_error(test_targets, test_preds)
    mae = mean_absolute_error(test_targets, test_preds)
    r2 = r2_score(test_targets, test_preds)
    median_ae = median_absolute_error(test_targets, test_preds)
    relative_errors = np.abs(test_targets - test_preds) / np.abs(test_targets)
    median_re = np.median(relative_errors)

    print("\nTest Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Median Absolute Error: {median_ae:.4f}")
    print(f"Median Relative Error: {median_re:.4f}")

    results = pd.DataFrame({"preds": test_preds, "targets": test_targets})
    # results.to_csv("./results.csv", index=False)   
    return model, results

if __name__ == "__main__":
    
    device = torch.device("cuda:0")  # according to your device
    print(f"Using device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
 
    main(
        csv_path="./data/METLIN_CCS_monomer.csv",
        checkpoint_save_path="./train.pth",
        cls_checkpoint="./data/CLS_repr.pt",  
        device=device
    )
