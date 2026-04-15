import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from prepare import prepare_features, MultiFeatureDataset, MLPRegressor, predict_and_evaluate

def main(csv_path, results_csv_path, device, batch_size=128):
    # Prepare features
    ion_repr, fp_features, fe_features, adduct_features, y = prepare_features(csv_path, device)

    # DataLoader
    all_indices = np.arange(len(y))
    test_dataset = MultiFeatureDataset(
        indices=all_indices,
        ion_repr=ion_repr,
        fp_features=fp_features,
        fe_features=fe_features,
        adduct_features=adduct_features,
        y=y.cpu().numpy()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load pretrained model
    model = MLPRegressor().to(device)
    checkpoint = torch.load("./ckpt/predict.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Predict and evaluate
    test_preds, test_targets, metrics = predict_and_evaluate(model, test_loader, device, csv_path)

    # Save prediction results
    results = pd.DataFrame({"CCS": test_targets, "preds": test_preds})
    results.to_csv(results_csv_path, index=False)
    print(f"Prediction results saved to: {results_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction with pretrained model")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output predictions CSV")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for DataLoader")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Fix random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Call main function
    main(args.input_csv, args.output_csv, device, args.batch_size)
   
