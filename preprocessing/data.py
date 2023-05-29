import json
import numpy as np
import pandas as pd
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
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

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
        print("edge_num: " + str(interaction_num))
        assert len(new_df.u) == len(new_df.i)
        all_index, nodes = pd.factorize(np.concatenate((df.u, df.i), axis=0))
        print("node_num: " + str(len(nodes)))
        new_df.u = all_index[0:interaction_num]
        new_df.i = all_index[interaction_num:interaction_num + interaction_num]

        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def data_preprocess(data_name, bipartite=True):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

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


def run_without_source(data_name, bipartite=True):
    Path("data/").mkdir(parents=True, exist_ok=True)
    # PATH = './{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    # OUT_FEAT = './ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    df = pd.read_csv('./ml_{}.csv'.format(data_name))
    edge_features = np.load('./ml_{}.npy'.format(data_name))
    node_features = np.load('./ml_{}_node.npy'.format(data_name))

    # df, feat = preprocess(PATH)
    print(edge_features.shape)
    print(node_features.shape)
    # new_df = reindex(graph_df, bipartite)

    print(str(max(df.u)))
    print(str(min(df.i)))

    if bipartite:
        print("bipartite graph")
        assert len(df.u) == len(df.i)
        interaction_num = len(df.u)
        print("edge_num: " + str(interaction_num))
        df.u, node1 = pd.factorize(df.u)
        df.i, node2 = pd.factorize(df.i)
        print("node_num: " + str(len(node1) + len(node2)))
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        df.i = new_i
        df.u += 1
        df.i += 1

        df.to_csv(OUT_DF)
        # print("node_dim: " + str(node_features.shape[1]))
        # if node_features.shape[1] != 172:
        max_idx = max(df.u.max(), df.i.max())
        rand_feat = np.zeros((max_idx + 1, 172))
        print(rand_feat.shape)
        np.save(OUT_NODE_FEAT, rand_feat)
        print("reconstruct node_dim: 172")
    else:
        print("No bipartite graph")
        interaction_num = len(df.u)
        print("edge_num: " + str(interaction_num))
        assert len(df.u) == len(df.i)
        all_index, nodes = pd.factorize(np.concatenate((df.u, df.i), axis=0))
        print("node_num: " + str(len(nodes)))
        df.u = all_index[0:interaction_num]
        df.i = all_index[interaction_num:interaction_num + interaction_num]

        df.u += 1
        df.i += 1

        df.to_csv(OUT_DF, columns=["u", "i", "ts", "label", "idx"])
        # print("node_dim: " + str(node_features.shape[1]))
        # if node_features.shape[1] != 172:
        max_idx = max(df.u.max(), df.i.max())
        rand_feat = np.zeros((max_idx + 1, 172))
        print(rand_feat.shape)
        np.save(OUT_NODE_FEAT, rand_feat)
        print("reconstruct node_dim: 172")
