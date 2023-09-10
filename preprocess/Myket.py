"""
MIT License

Copyright (c) [10/9/2023] [Erfan Loghmani, QiangHuang]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# TODO: Myket Dataset. This is the code of formatting the Myket data to align with the requirements of BenchTemp.

import pandas as pd
import numpy as np

# rename the column names
def rename_column():
    Myket = pd.read_csv("myket_with_features.csv")
    print(Myket.columns)

    Myket.rename(columns={'# user_id_map': 'user_id'}, inplace=True)
    Myket.rename(columns={'app_name_id': 'item_id'}, inplace=True)
    Myket.to_csv("myket_with_features_benchtemp.csv")

# preprocess the Myket dataset, file name: myket_with_features_benchtemp.csv
from pathlib import Path
import argparse


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      # Firstly, need to convert strings to floats, then convert to int.
      u = int(float(e[0]))
      i = int(float(e[1]))

      ts = float(e[2])
      label = float(e[3]) 

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    df.u, _ = pd.factorize(df.u)
    df.i, _ = pd.factorize(df.i)
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    new_df.u = df.u
    new_df.i = df.i  

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    print("No bipartite graph")
    interaction_num = len(new_df.u)
    print("edge_num: "+str(interaction_num))
    assert len(new_df.u) == len(new_df.i)
    all_index, nodes = pd.factorize(np.concatenate((df.u, df.i), axis=0))
    print("node_num: "+str(len(nodes)))
    new_df.u = all_index[0:interaction_num]
    new_df.i = all_index[interaction_num:interaction_num + interaction_num]

    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './{}.csv'.format(data_name)
  OUT_DF = './ml_{}.csv'.format(data_name)
  OUT_FEAT = './ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)
  print(feat.shape)
  print(rand_feat.shape)


parser = argparse.ArgumentParser('Interface for TGN data data')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='mooc')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

# rename_column()

run(args.data, bipartite=args.bipartite)