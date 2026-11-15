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
from rdkit.Chem.SaltRemover import SaltRemover

def desalt_smiles_list(smiles_list, remover=None):
    """
    Remove salts from each SMILES string and return canonical SMILES.
    - Replace unparseable SMILES strings with None and print an error.
    - Return a list with the same length as the input list.
    """
    remover = SaltRemover()

    desalted = []
    for idx, smiles in enumerate(smiles_list, start=1):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"Error: Failed to parse SMILES at position {idx}: {smiles}")
            desalted.append(None)
            continue

        desalted_mol = remover.StripMol(mol)
        desalted_smile = (
            Chem.MolToSmiles(desalted_mol, canonical=True, isomericSmiles=True)
            if desalted_mol else None
        )
        desalted.append(desalted_smile)

    return desalted

def remove_salts_heavy(mol):
    """Remove salt fragments by retaining the fragment with the most atoms."""
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if frags:
        largest_frag = max(frags, key=lambda x: x.GetNumAtoms())
        return largest_frag
    return mol


def desalt_smiles_list_heavy(smileslist):
    """
    Retain the largest fragment in each SMILES string and canonicalize it.

    Args:
        smileslist: List of input SMILES strings.

    Returns:
        A list of canonical SMILES strings after salt removal.
    """
    desalted_smiles = []

    for idx, smiles in enumerate(smileslist, start=1):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Failed to parse SMILES at position {idx}: {smiles}")

        desalted_mol = remove_salts_heavy(mol)

        if desalted_mol is None:
            raise ValueError(f"Failed to remove salts from SMILES at position {idx}: {smiles}")

        desalted_smile = Chem.MolToSmiles(desalted_mol)
        desalted_smiles.append(desalted_smile)

    return desalted_smiles

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
    """Canonicalize SMILES strings while preserving stereochemical information."""
    def canonicalize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
    return [canonicalize(smiles) for smiles in smi_list]


def formula_to_vector(formula, elements, adjust=None):
    """Convert a molecular formula to an element count vector with optional atom count adjustments."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts = {elem: int(num) if num else 1 for elem, num in tokens}
    if adjust:
        for elem, delta in adjust.items():
            counts[elem] = counts.get(elem, 0) + delta
    return [counts.get(elem, 0) for elem in elements]


def get_adjust_from_adduct(adduct):
    """Return element count adjustments for the specified adduct type."""
    if adduct == "[M+H]":
        return {"H": 1}
    elif adduct == "[M-H]":
        return {"H": -1}
    elif adduct == "[M+Na]":
        return {"Na": 1}
    elif adduct == "[M+K]":
        return {"K": 1}
    elif adduct == "[M+Cl]":
        return {"Cl": 1}
    elif adduct == "[M]":
        return {"H": 0}
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

def get_cls_repr(smiles_list, ckpt_path, remove_hs=False):
    """
    Extract molecular representations using custom Uni-Mol V1 weights.

    Args:
        smiles_list: List of canonical SMILES strings.
        ckpt_path: Path to the custom checkpoint file.
        remove_hs: Whether to remove hydrogen atoms. Must match the training setting.

    Returns:
        The representation results returned by get_repr().
    """
    clf = UniMolRepr(
        data_type="molecule",
        model_name="unimolv1",
        remove_hs=remove_hs,
    )

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )

    # Support common checkpoint formats.
    state_dict = checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break

    # Remove the module. prefix added by DataParallel or DDP.
    state_dict = {
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    # Exclude the prediction head when extracting CLS representations.
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("classification_head.")
    }

    result = clf.model.load_state_dict(state_dict, strict=False)

    # Check that all backbone weights have been loaded.
    missing_backbone = [
        k for k in result.missing_keys
        if not k.startswith("classification_head.")
    ]
    if missing_backbone:
        raise RuntimeError(
            f"Backbone weights were not fully loaded: {missing_backbone}"
        )

    print("Unexpected checkpoint keys:", result.unexpected_keys)
    print("Loaded custom weights from:", ckpt_path)

    clf.model.eval()

    with torch.no_grad():
        M_CLS_repr = clf.get_repr(
            smiles_list,
            return_atomic_reprs=False,
        )
    return M_CLS_repr

def prepare_features(csv_path, device, ft_path):
    """Read a CSV file and generate feature and target tensors.

    Returns:
        A tuple of (cls_repr, fp_features, fe_features, adduct_features, y).
    """
    df = pd.read_csv(csv_path)
    print(f"Input CSV shape: {df.shape}")


    df['Adduct'] = df['Adduct'].astype(str).str.strip()

    adduct_map = {
        '[M+H]+':'[M+H]',
        '[M+Na]+':'[M+Na]',
        '[M-H]-':'[M-H]',
        '[M+K]+':'[M+K]',
        }

    df['Adduct'] = df['Adduct'].replace(adduct_map)
    df = df[df['Adduct'].isin(['[M+H]', '[M+Na]', '[M-H]','[M+K]'])]
    #df = df[df['Adduct'].isin(['[M+H]', '[M+Na]', '[M-H]'])]
    #df = df[df['Adduct'].isin(['[M+K]'])]

    print(f"Filtered data shape: {df.shape}")

    desalt_smiles = desalt_smiles_list_heavy(df['smiles'].tolist())

    smiles_list = canonicalize_smiles(desalt_smiles)

    # Calculate molecular formulas only if the CSV has no 'Formula' column.
    if 'Formula' not in df.columns:
        formulas = []
        for smi in smiles_list:
            formulas.append(smiles_2_formula(smi))
        df['Formula'] = formulas

    # Generate Morgan fingerprints (2048 bits, radius=2).
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

    # Generate element count vectors with adduct adjustments.
    elements = ["H", "Li", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Se", "Br", "I"]
    rows = []
    for idx, row in df.iterrows():
        formula = row['Formula']
        adduct = row['Adduct']
        adjust = get_adjust_from_adduct(adduct)
        vector = formula_to_vector(formula, elements, adjust)
        rows.append(vector)

    M_CLS_repr = get_cls_repr(
        smiles_list,
        ckpt_path=ft_path,
        remove_hs=False
    )

    cls = [arr.tolist() for arr in M_CLS_repr['cls_repr']]

    # Map adduct types to scalar encodings.
    mapping = {"[M+H]": 32, "[M+Na]": 155, "[M-H]": -32, "[M]": 0, "[M+K]": 196, "[M+Cl]": 99}
    Adduct_lst = df["Adduct"].map(mapping).tolist()
    print(f"Adduct list length: {len(Adduct_lst)}")

    # Prepare target values; use NaN if CCS labels are unavailable.

    if 'CCS' not in df.columns:
        CCS = np.full(len(smiles_list), np.nan, dtype=np.float64)
        targets = CCS
    else:
        targets = df['CCS'].tolist()


    # Convert features and targets to PyTorch tensors.
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
    """Predict CCS values and evaluate them when reference CCS values are available.

    Args:
        model: Trained regression model.
        test_loader: Test data loader. Each batch includes targets, which are
            ignored if the input CSV has no 'CCS' column.
        device: Device used for computation.
        csv_path: Path to the test CSV file, used to check for a 'CCS' column.

    Returns:
        A tuple of (test_preds, test_targets, metrics). If the CSV has no
        'CCS' column, test_targets is an array of None values and metric
        values are "not applicable".
    """
    # Read the CSV file and check for a 'CCS' column.
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
                # Fill targets with None when CCS labels are unavailable.
                test_targets.extend([None] * len(batch_y))

    test_preds = np.array(test_preds)

    if has_ccs:
        test_targets = np.array(test_targets)

        t = pd.Series(test_targets)
        p = pd.Series(test_preds)

        print("NaN count in test_targets:", t.isna().sum(), "/", len(t))
        print("NaN count in test_preds:", p.isna().sum(), "/", len(p))
        # Calculate evaluation metrics.
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
        # Use None for targets and "not applicable" for metrics when labels are unavailable.
        test_targets = np.array([None] * len(test_preds))
        metrics = {
            'MSE': "not applicable",
            'MAE': "not applicable",
            'R2': "not applicable",
            'Median Absolute Error': "not applicable",
            'Median Relative Error': "not applicable"
        }
        print("No 'CCS' column found in the test CSV. Targets are filled with None; metrics are 'not applicable'.")

    return test_preds, test_targets, metrics


# ==================== Feature preparation for training ====================
def prepare_features4train(df, elements, device, cls_tensor=None):
    """
    Extract features from a DataFrame and return feature and target tensors.

    Args:
        df: pandas DataFrame containing 'smiles', 'Adduct', and 'CCS_AVG'.
        elements: Ordered list of elements used for element count vectors.
        device: Device used for the output tensors.
        cls_tensor: Path to precomputed CLS representations in a PyTorch
            checkpoint (.pt, .pth, or .ckpt) or a CSV file (.csv). A path must
            be provided even though the default value is None.

    Returns:
        A tuple of (cls_repr, fp_features, fe_features, adduct_features, y).
    """
    # 1. Normalize adduct labels.

    df['Adduct'] = df['Adduct'].astype(str).str.strip()
    adduct_map = {
        '[M+H]+':'[M+H]',
        '[M+Na]+':'[M+Na]',
        '[M-H]-':'[M-H]'}

    df['Adduct'] = df['Adduct'].replace(adduct_map)

    # 2. Remove salts and canonicalize SMILES strings.
    desalt_smiles = desalt_smiles_list(df['smiles'].tolist())

    smiles_list = canonicalize_smiles(desalt_smiles)

    if len(smiles_list) != len(df):
        raise ValueError("The number of SMILES strings does not match the number of DataFrame rows.")

    if any(smi is None for smi in smiles_list):
        raise ValueError("The canonicalized SMILES list contains None values.")

    # 3. Calculate molecular formulas if they are not provided.
    if 'Molecular Formula' not in df.columns:
        formulas = []

        for smi in smiles_list:
            formulas.append(smiles_2_formula(smi))

        df['Molecular Formula'] = formulas

    # 4. Generate Morgan fingerprints.
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

    # 5. Generate element count vectors with adduct adjustments.
    rows = []
    for idx, row in df.iterrows():
        formula = row['Molecular Formula'] # Molecular formula
        adduct = row['Adduct']
        adjust = get_adjust_from_adduct(adduct)
        vector = formula_to_vector(formula, elements, adjust)
        rows.append(vector)
    fe_features = torch.FloatTensor(np.array(rows)).to(device)

    # 6. Encode adduct types using scalar values.
    mapping = {"[M+H]": 32, "[M+Na]": 155, "[M-H]": -32, "[M+0]": 0, "[M+K]": 196, "[M+Cl]": 99}
    adduct_vals = df["Adduct"].map(mapping).values.reshape(-1, 1)
    adduct_features = torch.FloatTensor(adduct_vals).to(device)

    # 7. Load precomputed CLS representations.
    path = Path(cls_tensor)
    suffix = path.suffix.lower()

    if suffix in ['.pt', '.pth', '.ckpt']:
        # Load CLS representations from a PyTorch checkpoint.
        checkpoint = torch.load(cls_tensor, map_location=device)
        cls_repr = checkpoint["cls_repr"].to(device)
        print(f"Loaded cls_repr from checkpoint with shape: {cls_repr.shape}")

    elif suffix == '.csv':
        # Load CLS representations from a CSV file.
        cls = pd.read_csv(cls_tensor)
        # The CSV must contain a 'CLS_repr' column.
        str_vectors = cls['CLS_repr'].tolist()
        # Parse string representations of vectors into Python lists.
        vectors = [ast.literal_eval(vec_str) for vec_str in str_vectors]
        cls_repr = torch.FloatTensor(np.array(vectors)).to(device)
        print(f"Loaded cls_repr from CSV with shape: {cls_repr.shape}")

    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .pt, .pth, .ckpt, or .csv.")

    # 8. Prepare target values.
    y = torch.FloatTensor(df['CCS_AVG'].values).to(device)

    return cls_repr, fp_features, fe_features, adduct_features, y
