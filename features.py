import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import re
from unimol_tools_edt014post1 import UniMolRepr
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

def remove_salts(mol):
    """去除盐离子，保留分子中原子数最多的片段"""
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
    """将 SMILES 列表转换为规范化的 SMILES（异构信息保留）"""
    def canonicalize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
    return [canonicalize(smiles) for smiles in smi_list]


def formula_to_vector(formula, elements, adjust=None):
    """将分子式转换为元素计数向量，可调整电荷对应的原子增减"""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts = {elem: int(num) if num else 1 for elem, num in tokens}
    if adjust:
        for elem, delta in adjust.items():
            counts[elem] = counts.get(elem, 0) + delta
    return [counts.get(elem, 0) for elem in elements]


def get_adjust_from_adduct(adduct):
    """根据加合离子类型返回需要调整的元素及其数量"""
    if adduct == "[M+H]":
        return {"H": 1}
    elif adduct == "[M-H]":
        return {"H": -1}
    elif adduct == "[M+Na]":
        return {"Na": 1}
    else:
        return {}

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


class MultiFeatureDataset:
    def __init__(self, indices, ion_repr, fp_features, fe_features, adduct_features, y):
        self.indices = indices
        self.ion_repr = ion_repr
        self.fp_features = fp_features
        self.fe_features = fe_features
        self.adduct_features = adduct_features
        self.y = y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            self.ion_repr[i],
            self.fp_features[i],
            self.fe_features[i],
            self.adduct_features[i],
            self.y[i],
        )

def prepare_features(csv_path, device):
    """读取CSV并生成所有特征张量，返回 (ion_repr, fp_features, fe_features, adduct_features, y)"""
    df = pd.read_csv(csv_path)
    df['Adduct'] = df['Adduct'].astype(str).str.strip()
    df = df[df['Adduct'].isin(['[M+H]', '[M+Na]', '[M-H]'])]

    smiles_list = canonicalize_smiles(df['smiles'].tolist())

    # 计算分子式（仅当 CSV 中没有 'Formula' 列时才执行）
    if 'Formula' not in df.columns:
        formulas = []
        for smi in smiles_list:
            formulas.append(smiles_2_formula(smi))
        df['Formula'] = formulas

    # 准备目标值
    # 生成 Morgan 指纹 (2048 bits, radius=2)
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

    # 生成元素计数向量（考虑加合离子）
    elements = ["H", "Li", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Se", "Br", "I"]
    rows = []
    for idx, row in df.iterrows():
        formula = row['Formula']
        adduct = row['Adduct']
        adjust = get_adjust_from_adduct(adduct)
        vector = formula_to_vector(formula, elements, adjust)
        rows.append(vector)

    # 获取 UniMol 表征 (CLS token)
    clf = UniMolRepr(data_type='molecule', model_name='unimolv1', remove_hs=False)
    M_CLS_repr = clf.get_repr(smiles_list, return_atomic_reprs=False)
    cls = [arr.tolist() for arr in M_CLS_repr['cls_repr']]

    # 加合离子编码映射
    mapping = {"[M+H]": 32, "[M+Na]": 155, "[M-H]": -32}
    Adduct_lst = df["Adduct"].map(mapping).tolist()
    print(f"Adduct list length: {len(Adduct_lst)}")

    # y值

    if 'CCS' not in df.columns:
        CCS = np.full(len(smiles_list), np.nan, dtype=np.float64)
        targets = CCS
    else:
        targets = df['CCS'].tolist()
        

    # 转换为 PyTorch 张量
    ion_repr = torch.FloatTensor(np.array(cls)).to(device)           # (N, 512)
    fp_features = torch.FloatTensor(np.array(fingerprints)).to(device)  # (N, 2048)
    fe_features = torch.FloatTensor(np.array(rows)).to(device)      # (N, len(elements))
    adduct_features = torch.FloatTensor(np.array(Adduct_lst).reshape(-1, 1)).to(device)  # (N, 1)
    y = torch.FloatTensor(np.array(targets)).to(device)        # (N,)

    return ion_repr, fp_features, fe_features, adduct_features, y

def predict_and_evaluate(model, test_loader, device, csv_path):
    """对测试集进行预测并计算评估指标，返回 (test_preds, test_targets, metrics_dict)
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器（仍然返回 batch_y，但若无CCS列则忽略）
        device: 计算设备
        csv_path: 测试数据的CSV文件路径（用于检查是否存在 'CCS' 列）
    """
    # 读取CSV，检查是否存在 'CCS' 列
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
                # 没有CCS列时，用None填充（长度与batch一致）
                test_targets.extend([None] * len(batch_y))

    test_preds = np.array(test_preds)

    if has_ccs:
        test_targets = np.array(test_targets)
        # 计算评估指标
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
        # 没有真实标签：test_targets 全为 None，所有指标值为 "not applicable"
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


# ==================== 新增函数：专用于训练的特征准备 ====================
def prepare_features4train(df, elements, device, unirepr_checkpoint=None):
    """
    从 DataFrame 中提取所有特征，返回张量。
    参数:
        df: pandas DataFrame，必须包含列 'smiles', 'Adduct', 'CCS_AVG', 'Molecular Formula'
        elements: 元素列表
        device: torch device
        unirepr_checkpoint: 可选，预计算的 UniMol 表征 checkpoint 路径。若为 None，则动态计算
    返回:
        ion_repr, fp_features, vec_features, adduct_features, y
    """
    # 1. 规范化 SMILES
    smiles_list = canonicalize_smiles(df['smiles'].tolist())

    # 2. Morgan 指纹
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

    # 3. 元素计数向量（考虑加合离子调整）
    rows = []
    for idx, row in df.iterrows():
        formula = row['Molecular Formula']
        adduct = row['Adduct']
        adjust = get_adjust_from_adduct(adduct)
        vector = formula_to_vector(formula, elements, adjust)
        rows.append(vector)
    vec_features = torch.FloatTensor(np.array(rows)).to(device)

    # 4. 加合离子编码（标量映射）
    mapping = {"[M+H]": 32, "[M+Na]": 155, "[M-H]": -32}
    adduct_vals = df["Adduct"].map(mapping).values.reshape(-1, 1)
    adduct_features = torch.FloatTensor(adduct_vals).to(device)

    # 5. UniMol 表征
    if unirepr_checkpoint is not None:
        checkpoint = torch.load(unirepr_checkpoint, map_location=device)
        ion_repr = checkpoint["cls_repr"].to(device)
        # 确保行数匹配
        assert ion_repr.shape[0] == len(df), "UniRepr checkpoint row count mismatch"
    else:
        clf = UniMolRepr(data_type='molecule', model_name='unimolv1', remove_hs=False)
        M_CLS_repr = clf.get_repr(smiles_list, return_atomic_reprs=False)
        cls = np.array([arr.tolist() for arr in M_CLS_repr['cls_repr']])
        ion_repr = torch.FloatTensor(cls).to(device)

    # 6. 目标值
    y = torch.FloatTensor(df['CCS_AVG'].values).to(device)

    return ion_repr, fp_features, vec_features, adduct_features, y

