# -*- coding: utf-8 -*-
"""Build a world-average 50x50 technical-coefficient matrix (V1 scheme)
from the OECD ICIO 2021 edition table (reference year 2018).

Method:
  Z_world[i,j] = sum over all country pairs of intermediate flows from
                 industry i to industry j (45 ICIO industries).
  X_world[j]   = world gross output of industry j (OUTPUT row).
  A45[i,j]     = Z_world[i,j] / X_world[j].
  Bridge 45 -> 50 V1 sectors: split sectors inherit the parent's input
  structure (columns copied); parent rows are divided equally among the
  children to preserve column sums approximately. Documented approximation
  for network-adjacency use, not for full IO accounting.
Output: icio_A_50x50.csv (rows = supplying V1 sector, cols = using V1 sector).

Source data: OECD ICIO 2021 edition, 2018 reference year, downloaded from
https://figshare.com/articles/dataset/Inter_Country_Input_Output_table_from_OECD/21687470
(file ICIO2021_2018.csv). Place it in the scratchpad path below, or update
SRC to point at a local copy, before re-running.
"""
import io, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

SP = Path(r"C:\Users\wskim\AppData\Local\Temp\claude\G--My-Drive-Colab-Notebooks-Causal-Network-QQ\ec013144-18ac-4d8f-9566-784b71dae1b0\scratchpad")
SRC = SP / "ICIO2021_2018.csv"
BASE = Path(__file__).resolve().parent

IND45 = ['01T02','03','05T06','07T08','09','10T12','13T15','16','17T18','19',
         '20','21','22','23','24','25','26','27','28','29','30','31T33','35',
         '36T39','41T43','45T47','49','50','51','52','53','55T56','58T60','61',
         '62T63','64T66','68','69T75','77T82','84','85','86T88','90T93',
         '94T96','97T98']
assert len(IND45) == 45

# V1 (1..50) -> ICIO-45 industry code
V1_TO_45 = {1:'01T02',2:'01T02',3:'03',4:'05T06',5:'05T06',6:'07T08',7:'07T08',
            8:'09',9:'10T12',10:'13T15',11:'16',12:'17T18',13:'19',14:'20',
            15:'21',16:'22',17:'23',18:'24',19:'24',20:'25',21:'26',22:'27',
            23:'28',24:'29',25:'30',26:'30',27:'31T33',28:'35',29:'36T39',
            30:'41T43',31:'45T47',32:'49',33:'50',34:'51',35:'52',36:'53',
            37:'55T56',38:'58T60',39:'61',40:'62T63',41:'64T66',42:'68',
            43:'69T75',44:'77T82',45:'84',46:'85',47:'86T88',48:'90T93',
            49:'94T96',50:'97T98'}

print("loading ICIO ...")
df = pd.read_csv(SRC, index_col=0)
rows = df.index.astype(str)
cols = df.columns.astype(str)

def tail(s):
    m = re.match(r'^[A-Z0-9]{2,3}_(.+)$', s)
    return m.group(1) if m else s

row_tail = pd.Series([tail(r) for r in rows], index=df.index)
col_tail = pd.Series([tail(c) for c in cols], index=df.columns)

ind_rows = df.index[row_tail.isin(IND45)]
ind_cols = df.columns[col_tail.isin(IND45)]
print(f"industry rows={len(ind_rows)}, industry cols={len(ind_cols)}")

Z = df.loc[ind_rows, ind_cols]
rt = row_tail.loc[ind_rows].values
ct = col_tail.loc[ind_cols].values
# world aggregation: sum flows by industry tail
Z45 = pd.DataFrame(0.0, index=IND45, columns=IND45)
Zg = Z.groupby(rt).sum()          # rows -> 45
Zg = Zg.T.groupby(ct).sum().T     # cols -> 45
Z45.loc[Zg.index, Zg.columns] = Zg

out_row = df.loc['OUTPUT', ind_cols]
X45 = out_row.groupby(ct).sum().reindex(IND45)
A45 = Z45.div(X45.replace(0, np.nan), axis=1).fillna(0.0)
print("A45 built. col-sum range: %.3f - %.3f" % (A45.sum(0).min(), A45.sum(0).max()))

# ---- bridge to 50 V1 sectors ----
children = {}
for v1, c in V1_TO_45.items():
    children.setdefault(c, []).append(v1)

A50 = np.zeros((50, 50))
for j in range(1, 51):
    cj = V1_TO_45[j]
    for i in range(1, 51):
        ci = V1_TO_45[i]
        share = 1.0 / len(children[ci])       # split parent's row equally
        A50[i-1, j-1] = A45.loc[ci, cj] * share

out = pd.DataFrame(A50, index=[f"V1_{i}" for i in range(1, 51)],
                   columns=[f"V1_{j}" for j in range(1, 51)])
out.to_csv(BASE / "icio_A_50x50.csv")
print("saved icio_A_50x50.csv  col-sum range: %.3f - %.3f" % (A50.sum(0).min(), A50.sum(0).max()))
