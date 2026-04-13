import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unimol_tools_edt014post1 import UniMolRepr
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

df1=pd.read_csv('/home/yaping/projects/CCS/M_Dedu_H_Mono.csv')
df2=pd.read_csv('/home/yaping/projects/CCS/M_plus_Na_Mono.csv')
df3=pd.read_csv('/home/yaping/projects/CCS/M_plus_H_Mono.csv')

replace_map = {
    "O=C(NCCc1cnc2c1cccc2)CCCc1cnc2c1cccc2": "O=C(NCCC1=CNC2=CC=CC=C12)CCCC3=CNC4=CC=CC=C34",
    "Fc1ccc(cc1)Nc1ncnc2c1ncn2": "FC1=CC=C(NC2=C3N=CNC3=NC=N2)C=C1",
    "O=S(=O)(c1ccccc1)Nc1nc2c(n1)cccc2": "O=S(C1=CC=CC=C1)(NC2=NC3=CC=CC=C3N2)=O"
}

dfs = [df1, df2, df3]
for df in dfs:
    df['smiles'] = df['smiles'].astype(str).str.strip()
    df['smiles'] = df['smiles'].map(lambda x: replace_map.get(x, x))

all_smiles = []
for df in dfs:
    all_smiles.extend(df['smiles'].tolist())

clf = UniMolRepr(data_type='molecule', model_name='unimolv1', remove_hs=False)

M_CLS_repr = clf.get_repr(all_smiles, return_atomic_reprs=False)

#print(type(M_CLS_repr))
#print(M_CLS_repr)

datas = {"CLS_repr":[]}

for mol_idx, arr in enumerate(M_CLS_repr['cls_repr']):
    datas['CLS_repr'].append(str(arr.tolist())) 

dof = pd.DataFrame(datas['CLS_repr'], columns=['CLS_repr'])
dof.to_csv("/home/yaping/projects/results/CLS_7_Oct_V1_BS64_edt014post1_md1.csv", index=False)

