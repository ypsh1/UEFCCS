import pandas as pd
import numpy as np
from unimol_tools import MolTrain,MolPredict
from sklearn.model_selection import train_test_split
from rdkit import Chem

def canonicalize_smiles(smiles_list):
    def canonicalize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
    
    return [canonicalize(smiles) for smiles in smiles_list]

df = pd.read_csv('./data/METLIN_CCS_monomer.csv')

train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

train = train[train["Adduct"] == "[M+H]"]
train_targets = train["CCS_AVG"]

train_smiles = canonicalize_smiles(train["smiles"].tolist())

X = list(train_smiles)
y = list(train_targets)

print(len(X))
print(len(y))
train_data = {'SMILES': X, 'TARGET': y}

clf = MolTrain(task='regression',    
               data_type='molecule',
               epochs=50,                                 
               learning_rate=0.0001,
               batch_size=64,
               early_stopping=10,
               model_name='unimolv1',
               #model_size='570m',
               metrics='mse',
               split='random',
               save_path=f'./funetune_checkpoints', 
               )
    
clf.fit(train_data)    
