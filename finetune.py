import argparse
import pandas as pd
import numpy as np
from unimol_tools import MolTrain, MolPredict
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.SaltRemover import SaltRemover

RDLogger.DisableLog('rdApp.warning')


def canonicalize_smiles(smiles_list):
    def canonicalize(smiles):
        if smiles is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None

    return [canonicalize(smiles) for smiles in smiles_list]


def desalt_smiles_list(smileslist, remover):
    desalted_smiles = []
    for idx, smiles in enumerate(smileslist, start=1):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"error: {idx} SMILES cannot be parsed: {smiles}")
            desalted_smiles.append(None)
            continue
        desalted_mol = remover.StripMol(mol)
        desalted_smile = Chem.MolToSmiles(desalted_mol) if desalted_mol else None
        desalted_smiles.append(desalted_smile)
    return desalted_smiles


def main(args):

    df = pd.read_csv(args.csv)

    # Desalt and canonicalize SMILES
    remover = SaltRemover()
    smileslist = df['smiles'].tolist()
    desalted_smiles = desalt_smiles_list(smileslist, remover)
    df['smiles'] = canonicalize_smiles(desalted_smiles)

    # Filter for [M+H] adduct
    train = df[df["Adduct"] == "[M+H]"].copy()
    train_targets = train["CCS_AVG"].tolist()
    train_smiles = canonicalize_smiles(train["smiles"].tolist())

    # Drop unparseable SMILES so that SMILES and TARGET lengths stay aligned
    pairs = [(s, t) for s, t in zip(train_smiles, train_targets) if s is not None]
    train_smiles = [p[0] for p in pairs]
    train_targets = [p[1] for p in pairs]

    print("train_smiles:", len(train_smiles))
    print("train_targets:", len(train_targets))

    # Train the model
    train_data = {'SMILES': train_smiles, 'TARGET': train_targets}

    clf = MolTrain(
        task='regression',
        data_type='molecule',
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        model_name='unimolv1',
        metrics='mse',
        split='random',
        save_path=args.save_path,
    )

    clf.fit(train_data)
    print(f"model saved to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Uni-Mol model on METLIN CCS data (input is a prepared train CSV)"
    )
    parser.add_argument(
        "--csv", type=str, default="./data/METLIN_CCS_monomer_train.csv",
        help="input train CSV file path"
    )
    parser.add_argument(
        "--save_path", type=str, default="./finetune_unimol",
        help="path to save the finetuned model"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--early_stopping", type=int, default=10)

    args = parser.parse_args()
    main(args)
