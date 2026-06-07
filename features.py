import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import re
import ast
from pathlib import Path
from unimol_tools import UniMolRepr
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

def remove_salts(mol):
    """
    Remove salt ions, keep the fragment with the largest number of atoms in the molecule
    """
    
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if frags:
        largest_frag = max(frags, key=lambda x: x.GetNumAtoms())
        return largest_frag
    return mol

def smiles_2_formula(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol is None:
            return ""
        formula = CalcMolFormula(mol, separateIsotopes=False, abbreviateHIsotopes=False)

        return formula
    except Exception:
        return ""

def canonicalize_smiles(smi_list):
    """
    Convert a list of SMILES to canonical SMILES (retaining stereochemical information)
    """
    
    def canonicalize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
    return [canonicalize(smiles) for smiles in smi_list]


def formula_to_vector(formula, elements, adjust=None):
    """
    Convert molecular formula to element count vector, 
    with optional adjustment for atomic gains/losses corresponding to charge
    """
    
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts = {elem: int(num) if num else 1 for elem, num in tokens}
    if adjust:
        for elem, delta in adjust.items():
            counts[elem] = counts.get(elem, 0) + delta
    return [counts.get(elem, 0) for elem in elements]


def get_adjust_from_adduct(adduct):
    """
    Return elements and their counts to be adjusted according to the adduct ion type
    """    
    if adduct == "[M+H]":
        return {"H": 1}
    elif adduct == "[M-H]":
        return {"H": -1}
    elif adduct == "[M+Na]":
        return {"Na": 1}
    else:
        return {}

class MultiFeatureDataset:
    def __init__(self, indices, cls_repr, fp_features, fe_features, adduct_features, y):
        self.indices = indices
        self.cls_repr = cls_repr
        self.fp_features = fp_features
        self.fe_features = fe_features
        self.adduct_features = adduct_features
        self.y = y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            self.cls_repr[i],
            self.fp_features[i],
            self.fe_features[i],
            self.adduct_features[i],
            self.y[i],
        )

def prepare_features(csv_path, device):
    """
    Read CSV and generate all feature tensors, 
    returns (cls_repr, fp_features, fe_features, adduct_features, y)
    """
    
    df = pd.read_csv(csv_path)
    df['Adduct'] = df['Adduct'].astype(str).str.strip()
    df = df[df['Adduct'].isin(['[M+H]', '[M+Na]', '[M-H]'])]

    smiles_list = canonicalize_smiles(df['smiles'].tolist())

    # Calculate molecular formula (only executed when there is no 'Formula' column in the CSV)
    if 'Formula' not in df.columns:
        formulas = []
        for smi in smiles_list:
            formulas.append(smiles_2_formula(smi))
        df['Formula'] = formulas
 
    # Prepare target values
    # Generate Morgan fingerprint (2048 bits, radius=2)
    
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Failed to parse SMILES: {smi}")
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048,
            useFeatures=False, useChirality=True, useBondTypes=True
        )
        fp_str = fp.ToBitString()
        fp_int_list = [int(i) for i in fp_str]
        fingerprints.append(fp_int_list)
    print(f"Number of fingerprints: {len(fingerprints)}")

    # Generating element count vector (considering adduct ions)
    elements = ["H", "Li", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Se", "Br", "I"]
    rows = []
    for idx, row in df.iterrows():
        formula = row['Formula']
        adduct = row['Adduct']
        adjust = get_adjust_from_adduct(adduct)
        vector = formula_to_vector(formula, elements, adjust)
        rows.append(vector)

    # generating UniMol representation (CLS token)
    clf = UniMolRepr(data_type='molecule', model_name='unimolv1', remove_hs=False)
    M_CLS_repr = clf.get_repr(smiles_list, return_atomic_reprs=False)
    cls = [arr.tolist() for arr in M_CLS_repr['cls_repr']]

    # Adduct ion encoding mapping
    mapping = {"[M+H]": 32, "[M+Na]": 155, "[M-H]": -32}
    Adduct_lst = df["Adduct"].map(mapping).tolist()
    print(f"Adduct list length: {len(Adduct_lst)}")

    # y values

    if 'CCS' not in df.columns:
        CCS = np.full(len(smiles_list), np.nan, dtype=np.float64)
        targets = CCS
    else:
        targets = df['CCS'].tolist()
        

    # Convert to PyTorch tensor
    cls_repr = torch.FloatTensor(np.array(cls)).to(device)           # (N, 512)
    
    fp_features = torch.FloatTensor(np.array(fingerprints)).to(device)  # (N, 2048)
    fe_features = torch.FloatTensor(np.array(rows)).to(device)      # (N, len(elements))
    adduct_features = torch.FloatTensor(np.array(Adduct_lst).reshape(-1, 1)).to(device)  # (N, 1)
    y = torch.FloatTensor(np.array(targets)).to(device)        # (N,)

    return cls_repr, fp_features, fe_features, adduct_features, y

class MLPRegressor(nn.Module):
    def __init__(self, ion_dim=513, fp_dim=2048, fe_dim=16, hidden_dim=512, output_dim=1):
        super(MLPRegressor, self).__init__()
        self.fp_projection = nn.Linear(fp_dim, hidden_dim)
        self.vec_projection = nn.Linear(fe_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.regressor = nn.Sequential(
            nn.Linear(ion_dim + hidden_dim + hidden_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.003),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.001),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, ion, fp, fe, addt):
        fp_proj = self.fp_projection(fp)
        fp_proj = self.relu(fp_proj)
        fe_proj = self.vec_projection(fe)
        fe_proj = self.relu(fe_proj)
        combined = torch.cat([addt, ion, fe_proj, fp_proj], dim=1)  
        output = self.regressor(combined)
        return output

def predict_and_evaluate(model, test_loader, device, csv_path):
 
    """
       Make predictions on the test set and compute evaluation metrics, returns (test_preds, test_targets, metrics_dict)

       Args:
       model: trained model
       test_loader: test data loader (still returns batch_y, but ignored if there is no CCS column)
       device: computation device
       csv_path: CSV file path of the test data (used to check whether the 'CCS' column exists)
    """
    
    # Read CSV and check whether the 'CCS' column exists
    
    test_df = pd.read_csv(csv_path)
    has_ccs = 'CCS' in test_df.columns

    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch_ion, batch_fp, batch_vec, batch_addt, batch_y in test_loader:
            batch_ion = batch_ion.to(device)
            batch_fp = batch_fp.to(device)
            batch_vec = batch_vec.to(device)
            batch_addt = batch_addt.to(device)
            outputs = model(batch_ion, batch_fp, batch_vec, batch_addt)
            test_preds.extend(outputs.squeeze().cpu().numpy())

            if has_ccs:
                test_targets.extend(batch_y.cpu().numpy())
            else:
                # When there is no CCS column, fill with None (length consistent with batch)
                
                test_targets.extend([None] * len(batch_y))

    test_preds = np.array(test_preds)

    if has_ccs:
        test_targets = np.array(test_targets)
        
        mse = mean_squared_error(test_targets, test_preds)
        mae = mean_absolute_error(test_targets, test_preds)
        r2 = r2_score(test_targets, test_preds)
        median_ae = median_absolute_error(test_targets, test_preds)
        relative_errors = np.abs(test_targets - test_preds) / np.abs(test_targets)
        median_re = np.median(relative_errors)

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Median Absolute Error': median_ae,
            'Median Relative Error': median_re
        }

        print("Test Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
    else:
        # No ground truth labels: test_targets are all None, all metric values are "not applicable"
        
        test_targets = np.array([None] * len(test_preds))
        metrics = {
            'MSE': "not applicable",
            'MAE': "not applicable",
            'R2': "not applicable",
            'Median Absolute Error': "not applicable",
            'Median Relative Error': "not applicable"
        }
        print("No 'CCS' column found in test CSV. Returning predictions only, metrics are 'not applicable'.")

    return test_preds, test_targets, metrics


 def prepare_features4train(df, elements, device, cls_tensor=None):
    """
       Extract all features from the DataFrame and return tensors.

       Parameters:
       df: pandas DataFrame, must contain columns 'smiles', 'Adduct', 'CCS_AVG', 'Molecular Formula'
       elements: list of elements
       device: torch device
       unirepr_checkpoint: optional, precomputed UniMol representation checkpoint path. If None, dynamically compute

       Returns:
       cls_repr, fp_features, vec_features, adduct_features, y
    """
    #  canonicalize smiles
    smiles_list = canonicalize_smiles(df['smiles'].tolist())

    # ECFP fingerprints
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048,
            useFeatures=False, useChirality=True, useBondTypes=True
        )
        fp_str = fp.ToBitString()
        fp_int_list = [int(i) for i in fp_str]
        fingerprints.append(fp_int_list)
    fp_features = torch.FloatTensor(np.array(fingerprints)).to(device)

    # elements counting
    rows = []
    for idx, row in df.iterrows():
        formula = row['Molecular Formula']
        adduct = row['Adduct']
        adjust = get_adjust_from_adduct(adduct)
        vector = formula_to_vector(formula, elements, adjust)
        rows.append(vector)
    fe_features = torch.FloatTensor(np.array(rows)).to(device)

    # mapping adduct types
    mapping = {"[M+H]": 32, "[M+Na]": 155, "[M-H]": -32}
    adduct_vals = df["Adduct"].map(mapping).values.reshape(-1, 1)
    adduct_features = torch.FloatTensor(adduct_vals).to(device)
 
    path = Path(cls_tensor)
    suffix = path.suffix.lower()
    
    if suffix in ['.pt', '.pth', '.ckpt']:
        # PyTorch checkpoint fashion
        checkpoint = torch.load(cls_tensor, map_location=device)
        cls_repr = checkpoint["cls_repr"].to(device)
        print(f"Loaded cls_repr from checkpoint: shape {cls_repr.shape}")
        
    elif suffix == '.csv':
        # CSV fashion
        cls = pd.read_csv(cls_tensor)

        str_vectors = cls['CLS_repr'].tolist()
       
        vectors = [ast.literal_eval(vec_str) for vec_str in str_vectors]
        cls_repr = torch.FloatTensor(np.array(vectors)).to(device)
        print(f"Loaded cls_repr from CSV: shape {cls_repr.shape}")
        
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .pt/.pth/.ckpt or .csv")
    
    y = torch.FloatTensor(df['CCS_AVG'].values).to(device)

    return cls_repr, fp_features, fe_features, adduct_features, y

