import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.model_selection import train_test_split
from unimol_tools import UniMolRepr


# Utility functions
def canonicalize_smiles(smiles_list):
    def canonicalize(smiles):
        if smiles is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        return (
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if mol
            else None
        )

    return [canonicalize(smiles) for smiles in smiles_list]


def desalt_smiles(smiles_list):
    remover = SaltRemover()
    desalted_smiles = []
    for idx, smiles in enumerate(smiles_list, start=1):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Error: SMILES at index {idx} could not be parsed: {smiles}")
            desalted_smiles.append(None)
            continue
        desalted_mol = remover.StripMol(mol)
        desalted_smiles.append(Chem.MolToSmiles(desalted_mol) if desalted_mol else None)
    return desalted_smiles


def load_checkpoint_compat(UMpt_path):
    """Load a checkpoint with compatibility for common save formats,
    and strip the classification head / DataParallel prefix."""
    checkpoint = torch.load(UMpt_path, map_location="cpu", weights_only=True)

    state_dict = checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break

    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("classification_head.")
    }
    return state_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {v}")


# Main pipeline
def main(args):
    # 1. Load data
    df = pd.read_csv(args.csv)
    all_smiles = df["smiles"].tolist()

    # 2. Desalt and canonicalize
    all_smiles = desalt_smiles(all_smiles)
    all_smiles = canonicalize_smiles(all_smiles)

    # 3. Build the UniMol model
    clf = UniMolRepr(
        data_type="molecule",
        model_name="unimolv1",
        remove_hs=False,
    )

    # 4. Load custom weights
    state_dict = load_checkpoint_compat(args.UMpt_path)
    result = clf.model.load_state_dict(state_dict, strict=False)

    missing_backbone = [
        k for k in result.missing_keys
        if not k.startswith("classification_head.")
    ]
    if missing_backbone:
        raise RuntimeError(f"Backbone weights not fully loaded: {missing_backbone}")

    print("Unexpected checkpoint keys:", result.unexpected_keys)
    print("Loaded custom weights from:", args.UMpt_path)

    clf.model.eval()

    # 5. Extract CLS representations
    M_CLS_repr = clf.get_repr(all_smiles, return_atomic_reprs=False)
    CLS = torch.from_numpy(np.stack(M_CLS_repr["cls_repr"])).float()

    # 6. Save
    out_path = Path(args.out_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.split:
        indices = np.arange(len(CLS))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.seed,
            shuffle=True,
        )
        train_CLS = CLS[train_idx]
        test_CLS = CLS[test_idx]

        train_path = out_path.with_name(f"{out_path.stem}_train{out_path.suffix}")
        test_path = out_path.with_name(f"{out_path.stem}_test{out_path.suffix}")

        torch.save({"cls_repr": train_CLS}, train_path)
        torch.save({"cls_repr": test_CLS}, test_path)

        print(f"train: {tuple(train_CLS.shape)} -> {train_path}")
        print(f"test : {tuple(test_CLS.shape)} -> {test_path}")
    else:
        torch.save({"cls_repr": CLS}, out_path)
        print(f"saved: {tuple(CLS.shape)} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLS representations with UniMol and save them as pt files"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Input CSV path (must contain a 'smiles' column)")
    parser.add_argument("--UMpt_path", type=str, required=True,
                        help="Path to the custom UniMol checkpoint")
    parser.add_argument("--split", type=str2bool, default=False,
                        help="true: use train_test_split and save train/test pt files; "
                             "false: save a single pt file")
    parser.add_argument("--out_pt", type=str, default="./data/CLS.pt",
                        help="Output pt path. When split=true, it is used as the base name "
                             "to generate <stem>_train.pt and <stem>_test.pt")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set ratio when split=true (default 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting when split=true (default 42)")

    args = parser.parse_args()
    main(args)
