"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
# import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import *

# import our benchmark library: benchtemp
import benchtemp as bt

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                    help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information',
                    default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')

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
# NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
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
fh = logging.FileHandler('log/{}.log'.format(args.prefix + "-" + args.data + "-" + str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)

dataloader = bt.lp.DataLoader(dataset_path="./data/", dataset_name=DATA)

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
    new_node_test_data, new_old_node_val_data, new_old_node_test_data, new_new_node_val_data, \
    new_new_node_test_data, unseen_nodes_num = dataloader.load()

e_feat = edge_features
n_feat = node_features

src_l = full_data.sources
dst_l = full_data.destinations
e_idx_l = full_data.edge_idxs
label_l = full_data.labels
ts_l = full_data.timestamps

max_idx = max(full_data.sources.max(), full_data.destinations.max())

train_src_l = train_data.sources
train_dst_l = train_data.destinations
train_ts_l = train_data.timestamps
train_e_idx_l = train_data.edge_idxs
train_label_l = train_data.labels

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
# validation and test with edges that at least has one new node (not in training set)
# new-
nn_val_src_l = new_node_val_data.sources
nn_val_dst_l = new_node_val_data.destinations
nn_val_ts_l = new_node_val_data.timestamps
nn_val_e_idx_l = new_node_val_data.edge_idxs
nn_val_label_l = new_node_val_data.labels

nn_test_src_l = new_node_test_data.sources
nn_test_dst_l = new_node_test_data.destinations
nn_test_ts_l = new_node_test_data.timestamps
nn_test_e_idx_l = new_node_test_data.edge_idxs
nn_test_label_l = new_node_test_data.labels

# new-old
nn_new_old_val_src_l = new_old_node_val_data.sources
nn_new_old_val_dst_l = new_old_node_val_data.destinations
nn_new_old_val_ts_l = new_old_node_val_data.timestamps
nn_new_old_val_e_idx_l = new_old_node_val_data.edge_idxs
nn_new_old_val_label_l = new_old_node_val_data.labels

nn_new_old_test_src_l = new_old_node_test_data.sources
nn_new_old_test_dst_l = new_old_node_test_data.destinations
nn_new_old_test_ts_l = new_old_node_test_data.timestamps
nn_new_old_test_e_idx_l = new_old_node_test_data.edge_idxs
nn_new_old_test_label_l = new_old_node_test_data.labels

# new-new
nn_new_new_val_src_l = new_new_node_val_data.sources
nn_new_new_val_dst_l = new_new_node_val_data.destinations
nn_new_new_val_ts_l = new_new_node_val_data.timestamps
nn_new_new_val_e_idx_l = new_new_node_val_data.edge_idxs
nn_new_new_val_label_l = new_new_node_val_data.labels

nn_new_new_test_src_l = new_new_node_test_data.sources
nn_new_new_test_dst_l = new_new_node_test_data.destinations
nn_new_new_test_ts_l = new_new_node_test_data.timestamps
nn_new_new_test_e_idx_l = new_new_node_test_data.edge_idxs
nn_new_new_test_label_l = new_new_node_test_data.labels

### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
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

train_rand_sampler = bt.lp.RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = bt.lp.RandEdgeSampler(src_l, dst_l, seed=0)
nn_val_rand_sampler = bt.lp.RandEdgeSampler(nn_val_src_l, nn_val_dst_l, seed=1)
test_rand_sampler = bt.lp.RandEdgeSampler(src_l, dst_l, seed=2)
nn_test_rand_sampler = bt.lp.RandEdgeSampler(nn_test_src_l, nn_test_dst_l, seed=3)
nn_new_old_test_rand_sampler = bt.lp.RandEdgeSampler(nn_new_old_test_src_l, nn_new_old_test_dst_l, seed=4)
nn_new_new_test_rand_sampler = bt.lp.RandEdgeSampler(nn_new_new_test_src_l, nn_new_new_test_dst_l, seed=5)

### Model initialize
device = torch.device('cuda:{}'.format(GPU))

test_auc_list = []
test_ap_list = []
nn_test_auc_list = []
nn_test_ap_list = []
new_old_test_auc_list = []
new_old_test_ap_list = []
new_new_test_auc_list = []
new_new_test_ap_list = []
for i in range(args.n_runs):
    tgan = TGAN(train_ngh_finder, n_feat, e_feat,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
    optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    tgan = tgan.to(device)

    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list)

    early_stopper = bt.EarlyStopMonitor()
    for epoch in range(NUM_EPOCH):
        # Training
        # training use only training graph
        start_epoch = time.time()
        tgan.ngh_finder = train_ngh_finder
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)
        logger.info('start {} epoch'.format(epoch))
        for k in tqdm(range(num_batch)):
            # percent = 100 * k / num_batch
            # if k % int(0.2 * num_batch) == 0:
            #     logger.info('progress: {0:10.4f}'.format(percent))

            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            label_l_cut = train_label_l[s_idx:e_idx]
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

            optimizer.zero_grad()
            tgan = tgan.train()
            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            loss = criterion(pos_prob, pos_label)
            loss += criterion(neg_prob, neg_label)

            loss.backward()
            optimizer.step()
            # get training results
            with torch.no_grad():
                tgan = tgan.eval()
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                # f1.append(f1_score(true_label, pred_label))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))

        # validation phase use all information
        tgan.ngh_finder = full_ngh_finder
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgan, val_rand_sampler, val_src_l,
                                                          val_dst_l, val_ts_l, val_label_l)

        nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for new nodes', tgan, val_rand_sampler,
                                                                      nn_val_src_l,
                                                                      nn_val_dst_l, nn_val_ts_l, nn_val_label_l)
        total_epoch_time = time.time() - start_epoch
        # logger.info('epoch: {}:'.format(epoch))
        total_epoch_time = time.time() - start_epoch
        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
        logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
        logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))
        # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgan.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgan.eval()
            break
        else:
            torch.save(tgan.state_dict(), get_checkpoint_path(epoch))

    # testing phase use all information
    tgan.ngh_finder = full_ngh_finder
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l,
                                                          test_dst_l, test_ts_l, test_label_l)

    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new- nodes', tgan, nn_test_rand_sampler,
                                                                      nn_test_src_l,
                                                                      nn_test_dst_l, nn_test_ts_l, nn_test_label_l)

    new_old_test_acc, new_old_test_ap, new_old_test_f1, new_old_test_auc = \
        eval_one_epoch('test for new-old nodes', tgan, nn_new_old_test_rand_sampler, nn_new_old_test_src_l,
                       nn_new_old_test_dst_l, nn_new_old_test_ts_l, nn_new_old_test_label_l)

    new_new_test_acc, new_new_test_ap, new_new_test_f1, new_new_test_auc = \
        eval_one_epoch('test for new-new nodes', tgan, nn_new_new_test_rand_sampler, nn_new_new_test_src_l,
                       nn_new_new_test_dst_l, nn_new_new_test_ts_l, nn_new_new_test_label_l)

    logger.info(
        'Test statistics: Transductive: Old  nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
    logger.info(
        'Test statistics: Inductive:    New- nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
    logger.info(
        'Test statistics: Inductive: New-Old nodes -- auc: {}, ap: {}'.format(new_old_test_auc, new_old_test_ap))
    logger.info(
        'Test statistics: Inductive: New-New nodes -- auc: {}, ap: {}'.format(new_new_test_auc, new_new_test_ap))
    # logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
    # logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

    test_auc_list.append(test_auc)
    test_ap_list.append(test_ap)
    nn_test_auc_list.append(nn_test_auc)
    nn_test_ap_list.append(nn_test_ap)
    new_old_test_auc_list.append(new_old_test_auc)
    new_old_test_ap_list.append(new_old_test_ap)
    new_new_test_auc_list.append(new_new_test_auc)
    new_new_test_ap_list.append(new_new_test_ap)

    logger.info('Saving TGAN model')
    torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGAN models saved')

logger.info(
    'AVG+STD Transductive: ---------------- Old  nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(test_auc_list), np.std(test_auc_list), np.average(test_ap_list), np.std(test_ap_list)))
logger.info(
    'AVG+STD Test statistics: Inductive: -- New- nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(nn_test_auc_list), np.std(nn_test_auc_list), np.average(nn_test_ap_list), np.std(nn_test_ap_list)))
logger.info(
    'AVG+STD Test statistics: Inductive: New-Old nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(new_old_test_auc_list), np.std(new_old_test_auc_list), np.average(new_old_test_ap_list),
        np.std(new_old_test_ap_list)))
logger.info(
    'AVG+STD Test statistics: Inductive: New-New nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(new_new_test_auc_list), np.std(new_new_test_auc_list), np.average(new_new_test_ap_list),
        np.std(new_new_test_ap_list)))

logger.info("--------------Rounding to four decimal places--------------")
logger.info(
    'AVG+STD Transductive: ---------------- Old  nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.around(np.average(test_auc_list), 4), np.around(np.std(test_auc_list), 4),
        np.around(np.average(test_ap_list), 4), np.around(np.std(test_ap_list), 4)))
logger.info(
    'AVG+STD Test statistics: Inductive: -- New- nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.around(np.average(nn_test_auc_list), 4), np.around(np.std(nn_test_auc_list), 4),
        np.around(np.average(nn_test_ap_list), 4), np.around(np.std(nn_test_ap_list), 4)))
logger.info(
    'AVG+STD Test statistics: Inductive: New-Old nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.around(np.average(new_old_test_auc_list), 4), np.around(np.std(new_old_test_auc_list), 4),
        np.around(np.average(new_old_test_ap_list), 4), np.around(np.std(new_old_test_ap_list), 4)))
logger.info(
    'AVG+STD Test statistics: Inductive: New-New nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.around(np.average(new_new_test_auc_list), 4), np.around(np.std(new_new_test_auc_list), 4),
        np.around(np.average(new_new_test_ap_list), 4), np.around(np.std(new_new_test_ap_list), 4)))






