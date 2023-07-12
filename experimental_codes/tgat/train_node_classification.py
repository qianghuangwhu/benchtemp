"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import sys
import random
import argparse

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import *

# import our benchmark library: benchtemp
import benchtemp as bt


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=50, help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true',
                    help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information',
                    default='time')

parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--n_runs', type=int, default=3, help='Number of runs')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

args.prefix = "TGAT"

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(args.prefix + "-NC-" + args.data + "-" + str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

dataloader = bt.nc.DataLoader(dataset_path="./data/", dataset_name=DATA)

# full_data, node_features, edge_features, train_data, val_data, test_data = \
#     get_data_node_classification(DATA, use_validation=args.use_validation)

full_data, node_features, edge_features, train_data, val_data, test_data = \
    dataloader.load()

### Load data and train val test split
e_feat = edge_features
n_feat = node_features

# src_l = g_df.u.values
# dst_l = g_df.i.values
# e_idx_l = g_df.idx.values
# label_l = g_df.label.values
# ts_l = g_df.ts.values

src_l = full_data.sources
dst_l = full_data.destinations
e_idx_l = full_data.edge_idxs
label_l = full_data.labels
ts_l = full_data.timestamps

train_src_l = train_data.sources
train_dst_l = train_data.destinations
train_ts_l = train_data.timestamps
train_e_idx_l = train_data.edge_idxs
train_label_l = train_data.labels

# use the validation as test dataset
val_src_l = val_data.sources
val_dst_l = val_data.destinations
val_ts_l = val_data.timestamps
val_e_idx_l = val_data.edge_idxs
val_label_l = val_data.labels

test_src_l = test_data.sources
test_dst_l = test_data.destinations
test_ts_l = test_data.timestamps
test_e_idx_l = test_data.edge_idxs
test_label_l = test_data.labels

max_idx = max(src_l.max(), dst_l.max())

### Initialize the data structure for graph and edge sampling
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)


def eval_epoch(src_l, dst_l, ts_l, label_l, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
    pred_prob = np.zeros(len(src_l))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            src_l_cut = src_l[s_idx:e_idx]
            dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]
            label_l_cut = label_l[s_idx:e_idx]
            size = len(src_l_cut)
            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            loss += lr_criterion_eval(lr_prob, src_label).item()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()

    auc_roc = roc_auc_score(label_l, pred_prob)
    return auc_roc, loss / num_instance


### Model initialize


device = torch.device('cuda:{}'.format(GPU))
test_auc_list = []
for i in range(args.n_runs):
    tgan = TGAN(train_ngh_finder, n_feat, e_feat,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
    # optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
    # criterion = torch.nn.BCELoss()
    tgan = tgan.to(device)

    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    logger.debug('num of training instances: {}'.format(num_instance))
    logger.debug('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list)

    logger.info('loading saved TGAN model')
    model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
    tgan.load_state_dict(torch.load(model_path))
    tgan.eval()
    logger.info('TGAN models loaded')
    logger.info('Start training node classification task')

    lr_model = LR(n_feat.shape[1])
    lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
    lr_model = lr_model.to(device)
    tgan.ngh_finder = full_ngh_finder
    idx_list = np.arange(len(train_src_l))
    lr_criterion = torch.nn.BCELoss()
    lr_criterion_eval = torch.nn.BCELoss()

    early_stopper = bt.EarlyStopMonitor()
    for epoch in range(args.n_epoch):
        start_epoch = time.time()
        lr_pred_prob = np.zeros(len(train_src_l))
        np.random.shuffle(idx_list)
        tgan = tgan.eval()
        lr_model = lr_model.train()
        # num_batch
        for k in tqdm(range(num_batch)):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = train_src_l[s_idx:e_idx]
            dst_l_cut = train_dst_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            label_l_cut = train_label_l[s_idx:e_idx]

            size = len(src_l_cut)

            lr_optimizer.zero_grad()
            with torch.no_grad():
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)

            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            lr_loss = lr_criterion(lr_prob, src_label)
            lr_loss.backward()
            lr_optimizer.step()

        # train_auc, train_loss = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l, BATCH_SIZE, lr_model,
        #                                    tgan)
        val_auc, val_loss = eval_epoch(val_src_l, val_dst_l, val_ts_l, val_label_l, BATCH_SIZE, lr_model, tgan)
        # torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
        total_epoch_time = time.time() - start_epoch
        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info(f'epcoh{epoch}---val auc: {val_auc}')

        if early_stopper.early_stop_check(val_auc):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgan.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgan.eval()
            break
        else:
            torch.save(tgan.state_dict(), get_checkpoint_path(epoch))

    test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l, BATCH_SIZE, lr_model, tgan)
    # torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
    logger.info(f'test auc: {test_auc}')
    test_auc_list.append(test_auc)

logger.info('NC task -- auc: {} \u00B1 {}'.format(np.average(test_auc_list), np.std(test_auc_list)))
logger.info("--------------Rounding to four decimal places--------------")
logger.info(
    'NC task -- auc: {} \u00B1 {}'.format(np.around(np.average(test_auc_list), 4), np.around(np.std(test_auc_list), 4)))
