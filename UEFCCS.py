import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import re
warnings.filterwarnings('ignore')
from rdkit import RDLogger
from unimol_tools_edt014post1 import UniMolRepr

RDLogger.DisableLog('rdApp.warning')

device = torch.device("cuda:0")  #0, 1, 2...
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

df1 = pd.read_csv("/home/yaping/projects/regre/drugs/ac7b01709_si_003_22.csv")

df1['Adduct'] = df1['Adduct'].str.replace(r'[\+\-]$', '', regex=True)

adduct_map = {
    'M+H':'[M+H]',
    'M+Na':'[M+Na]',
    'M-H':'[M-H]'
}

df1['Adduct'] = df1['Adduct'].astype(str).str.strip()
df1['Adduct'] = df1['Adduct'].replace(adduct_map)

df1 = df1[df1['Adduct'].isin(['[M+H]', '[M+Na]', '[M-H]'])]

smileslist0 = df1["smiles"].tolist()

def remove_salts(mol):
    
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    if frags:
        largest_frag = max(frags, key=lambda x: x.GetNumAtoms())
        return largest_frag
    return mol


desalted_smiles = []
for idx, smiles in enumerate(smileslist0, start=1):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"错误: 第 {idx} 个 SMILES 无法解析: {smiles}")
        desalted_smiles.append(None)  # 存空值
        continue

    desalted_mol = remove_salts(mol)
    desalted_smile = Chem.MolToSmiles(desalted_mol) if desalted_mol else None
    desalted_smiles.append(desalted_smile)

    #print(f"{smiles}")
    #print(f"  -> {desalted_smile}")
    #print()

#print(f"results: {desalted_smiles}")

df1["smiles"] = desalted_smiles

from collections import defaultdict

def smiles_to_molecular_formula(smiles: str) -> str:
    """
    从SMILES字符串生成分子式（Hill系统排序）。

    参数:
        smiles (str): SMILES字符串

    返回:
        str: 分子式字符串（如 "C6H12O6"），若解析失败返回空字符串
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""

        # 原子计数字典（不含氢）
        elem_count = defaultdict(int)
        hydrogen_count = 0

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'H':
                # 显式氢原子
                hydrogen_count += 1
            else:
                elem_count[symbol] += 1
                # 加上该原子连接的隐式氢原子数
                hydrogen_count += atom.GetTotalNumHs()

        # 按Hill系统排序：先C，再H，然后其他元素按字母顺序
        formula_parts = []

        # 碳
        if 'C' in elem_count:
            count = elem_count.pop('C')
            formula_parts.append(f"C{count if count > 1 else ''}")
        # 氢（始终放在碳之后，如果存在）
        if hydrogen_count > 0:
            formula_parts.append(f"H{hydrogen_count if hydrogen_count > 1 else ''}")

        # 其余元素按字母顺序
        for elem in sorted(elem_count.keys()):
            count = elem_count[elem]
            formula_parts.append(f"{elem}{count if count > 1 else ''}")

        return "".join(formula_parts)
    except Exception:
        # 捕获任何异常，返回空字符串
        return ""

formulas = []
for smi in desalted_smiles:
        formulas.append(smiles_to_molecular_formula(smi))

datas = {'TARGET': [], "hac":[], 'm/z':[], "CLS":[], "formulas":[], "smiles":[]}

#datas['TARGET'].extend(df1['CCS_AVG'].tolist())
datas['TARGET'].extend(df1['CCS'].tolist())
#datas['m/z'].extend(df1['m/z'].tolist())
#datas['formulas'].extend(df1['Formula'].tolist())
datas["smiles"].extend(df1['smiles'].tolist())

df1['Formula'] = formulas

smi_list = datas["smiles"]

def canonicalize_smiles(smi_list):
    def canonicalize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
    
    return [canonicalize(smiles) for smiles in smi_list]

smiles_list = canonicalize_smiles(smi_list)
"""
def remove_salts(mol):
     if mol is None:
        return None
    
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    
    if frags:
        largest_frag = max(frags, key=lambda x: x.GetNumAtoms())
        return largest_frag
    return mol

def clean_smiles_list(smi_list2):
     results = []
    
    for smiles in smi_list2:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            results.append("")  # 无法解析，返回空字符串
            continue
            
        desalted_mol = remove_salts(mol)
        
        if desalted_mol:
            results.append(Chem.MolToSmiles(desalted_mol))
        else:
            results.append("")  # 去盐失败，返回空字符串
            
    return results

smiles_list = clean_smiles_list(smi_list2)
"""
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

print(len(fingerprints))

elements = ["H", "Li", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Se", "Br", "I"]

def formula_to_vector(formula, elements, adjust=None):
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts = {elem: int(num) if num else 1 for elem, num in tokens}
    
    if adjust:
        for elem, delta in adjust.items():
            counts[elem] = counts.get(elem, 0) + delta
    
    return [counts.get(elem, 0) for elem in elements]

# 根据 Adduct 列确定调整规则
def get_adjust_from_adduct(adduct):
    if adduct == "[M+H]":
        return {"H": 1}
    elif adduct == "[M-H]":
        return {"H": -1}
    elif adduct == "[M+Na]":
        return {"Na": 1}
    else:
        # 如果没有匹配的加合物类型，不进行调整
        return {}

rows = []
for idx, row in df1.iterrows():
    formula = row['Formula']
    adduct = row['Adduct']
    
    # 根据 Adduct 列获取调整规则
    adjust = get_adjust_from_adduct(adduct)
    
    # 计算向量
    vector = formula_to_vector(formula, elements, adjust)
    
    rows.append(vector)


clf = UniMolRepr(data_type='molecule', model_name='unimolv1', remove_hs=False)

M_CLS_repr = clf.get_repr(smiles_list, return_atomic_reprs=False)

cls = []

for mol_idx, arr in enumerate(M_CLS_repr['cls_repr']):
    cls.append(arr.tolist())

mapping = {"[M+H]": 32, "[M+Na]": 155, "[M-H]": -32} 
Adduct_lst = df1["Adduct"].map(mapping).tolist()
print(len(Adduct_lst))

dof_repr = torch.FloatTensor(np.array(cls)).to(device)  
fp_features = torch.FloatTensor(np.array(fingerprints)).to(device)  
print(fp_features.shape)
vector_features = torch.FloatTensor(np.array(rows)).to(device)  

Adduct_lst = np.array(Adduct_lst).reshape(-1, 1)
adduct_features = torch.FloatTensor(Adduct_lst).to(device)  
 
y = torch.FloatTensor(np.array(datas["TARGET"])).to(device) 
 
all_indices = np.arange(len(y))

class MultiFeatureDataset:
    def __init__(self, indices, dof_repr, fp_features, vector_features, adduct_features, y):
        self.indices = indices
        self.dof_repr = dof_repr
        self.fp_features = fp_features
        self.vector_features = vector_features
        self.adduct_features = adduct_features
        self.y = y
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            self.dof_repr[i],
            self.fp_features[i],
            self.vector_features[i],
            self.adduct_features[i],
            self.y[i],
        )
 
test_dataset = MultiFeatureDataset(
    indices=all_indices,
    dof_repr=dof_repr,
    fp_features=fp_features,
    vector_features=vector_features,
    adduct_features=adduct_features,
    y=y.cpu().numpy()   
)
 
batch_size = 128
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
 

class MLPRegressor(nn.Module):
    def __init__(self, dof_dim=512, fp_dim=2048, vec_dim=16, adduct_dim=1, hidden_dim=512, output_dim=1):
        super(MLPRegressor, self).__init__()
 
        self.fp_projection = nn.Linear(fp_dim, 512)       
        self.vec_projection = nn.Linear(vec_dim, 512)     
        self.relu = nn.ReLU()

        self.regressor = nn.Sequential(
            nn.Linear(dof_dim + 1025, 512),   
            nn.ReLU(),
            nn.Dropout(0.003),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.001),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
       
    def forward(self, dof, fp, vec, addt):
 
        fp_proj = self.fp_projection(fp)       
        fp_proj = self.relu(fp_proj)
        vec_proj = self.vec_projection(vec)    
        vec_proj = self.relu(vec_proj)
    
        combined = torch.cat([addt, dof, vec_proj, fp_proj], dim=1)  
 
        output = self.regressor(combined)  

        return output
 
model = MLPRegressor().to(device)

checkpoint = torch.load("/home/yaping/projects/regre/shap/mlp2_15estp.pth")
model.load_state_dict(checkpoint['model_state_dict'])

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

mse = mean_squared_error(test_targets, test_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_targets, test_preds)
r2 = r2_score(test_targets, test_preds) 
median_ae = median_absolute_error(test_targets, test_preds)
    

print(f"Test Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"Median Absolute Error: {median_ae:.4f}")  

results = pd.DataFrame({
    "preds_CCS": test_preds,
    "target_CCS": test_targets
    })

results.to_csv("./UEFCCS_prediction_results.csv", index=False)
