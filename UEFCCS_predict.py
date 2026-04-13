import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
from uefccs import features
from features import prepare_features, MultiFeatureDataset, MLPRegressor,predict_and_evaluate

def main():

    csv_path = "./your_path"
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
    batch_size = 128
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #load pretrained model
    model = MLPRegressor().to(device)
    checkpoint = torch.load("./weights4predict.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_preds, test_targets, metrics = predict_and_evaluate(model, test_loader, device)

    # results = pd.DataFrame({"preds": test_preds, "targets": test_targets})
    # results.to_csv("/path/to/results.csv", index=False)


if __name__ == "__main__":
    
    device = torch.device("cuda:1") #according to your devices
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    main()
