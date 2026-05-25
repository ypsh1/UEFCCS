import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unimol_tools_edt014post1 import UniMolRepr
import ast
from rdkit import Chem

def canonicalize_smiles(smiles_list):
    def canonicalize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
    
    return [canonicalize(smiles) for smiles in smiles_list]

df = pd.read_csv('./data/METLIN_CCS_monomer.csv')

all_smiles = df["smiles"].tolist()

clf = UniMolRepr(data_type='molecule', model_name='unimolv1', remove_hs=False)

M_CLS_repr = clf.get_repr(all_smiles, return_atomic_reprs=False)

datas = []

for mol_idx, arr in enumerate(M_CLS_repr['cls_repr']):
    datas.append(str(arr.tolist())) 

CLS = pd.DataFrame(datas, columns=['CLS_repr'])

CLS.to_csv("data/CLS_token_METLINCCS.csv", index=False)

 

