import pandas as pd
from log import *
from utils import *
from train import *
from module import NeurTWs
from graph import NeighborFinder
import resource

# import our benchmark library: benchtemp
import benchtemp as bt

args, sys_argv = get_args()

assert (args.cpu_cores >= -1)
set_random_seed(args.seed)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

DATA = args.data

dataloader = bt.lp.DataLoader(dataset_path="./data/", dataset_name=DATA)

### Extract data for training, validation and testing
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

train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data.sources, train_data.destinations, train_data.timestamps, train_data.edge_idxs, train_data.labels
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data.sources, val_data.destinations, val_data.timestamps, val_data.edge_idxs, val_data.labels
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = test_data.sources, test_data.destinations, test_data.timestamps, test_data.edge_idxs, test_data.labels

test_src_new_l, test_dst_new_l, test_ts_new_l, test_e_idx_new_l, test_label_new_l = new_node_test_data.sources, new_node_test_data.destinations, new_node_test_data.timestamps, new_node_test_data.edge_idxs, new_node_test_data.labels
test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = new_old_node_test_data.sources, new_old_node_test_data.destinations, new_old_node_test_data.timestamps, new_old_node_test_data.edge_idxs, new_old_node_test_data.labels
test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = new_new_node_test_data.sources, new_new_node_test_data.destinations, new_new_node_test_data.timestamps, new_new_node_test_data.edge_idxs, new_new_node_test_data.labels

train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))

full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=args.temporal_bias, spatial_bias=args.spatial_bias,
                                 ee_bias=args.ee_bias, use_cache=args.ngh_cache, sample_method=args.pos_sample,
                                 limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span)

partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))

partial_ngh_finder = NeighborFinder(partial_adj_list, temporal_bias=args.temporal_bias, spatial_bias=args.spatial_bias,
                                    ee_bias=args.ee_bias, use_cache=args.ngh_cache, sample_method=args.pos_sample,
                                    limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span)

ngh_finders = partial_ngh_finder, full_ngh_finder
logger.info('Sampling module - temporal bias: {}, spatial bias: {}, E&E bias: {}'.format(args.temporal_bias,
                                                                                         args.spatial_bias,
                                                                                         args.ee_bias))

train_rand_sampler = bt.lp.RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = bt.lp.RandEdgeSampler(np.concatenate((train_src_l, val_src_l)), np.concatenate((train_dst_l, val_dst_l)), seed=0)
test_rand_sampler = bt.lp.RandEdgeSampler(np.concatenate((train_src_l, val_src_l, test_src_l)), np.concatenate((train_dst_l, val_dst_l, test_dst_l)), seed=1)
rand_samplers = train_rand_sampler, val_rand_sampler

test_auc_list = []
test_ap_list = []
nn_test_auc_list = []
nn_test_ap_list = []
new_old_test_auc_list = []
new_old_test_ap_list = []
new_new_test_auc_list = []
new_new_test_ap_list = []

device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')

for i in range(args.n_runs):
    model = NeurTWs(n_feat=n_feat, e_feat=e_feat, walk_mutual=args.walk_mutual, walk_linear_out=args.walk_linear_out,
                    pos_enc=args.pos_enc, pos_dim=args.pos_dim, num_layers=args.n_layer, num_neighbors=args.n_degree,
                    tau=args.tau, negs=args.negs, solver=args.solver, step_size=args.step_size, drop_out=args.drop_out,
                    cpu_cores=args.cpu_cores, verbosity=args.verbosity, get_checkpoint_path=get_checkpoint_path).to(
        device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stopper = bt.EarlyStopMonitor(tolerance=args.tolerance)

    # train_val(train_val_data, model, args.mode, args.bs, args.n_epoch, optimizer, early_stopper,
    #           ngh_finders, rand_samplers, logger, args.negs)

    # train_data, val_data = train_val_data
    # train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data
    # val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data
    # train_rand_sampler, val_rand_sampler = rand_samplers
    # partial_ngh_finder, full_ngh_finder = ngh_finders

    mode = args.mode
    bs = args.bs
    epochs = args.n_epoch
    negatives = args.negs


    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / bs)
    idx_list = np.arange(num_instance)

    for epoch in range(epochs):
        if mode == 't':
            model.update_ngh_finder(full_ngh_finder)
        elif mode == 'i':
            model.update_ngh_finder(partial_ngh_finder)
        else:
            raise ValueError('training mode {} not found.'.format(mode))

        start_epoch = time.time()
        ap, auc, m_loss = [], [], []
        np.random.shuffle(idx_list)
        logger.info('start {} epoch'.format(epoch))
        for k in tqdm(range(num_batch)):
            s_idx = k * bs
            e_idx = min(num_instance - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx]
            ts_l_cut = train_ts_l[batch_idx]
            e_l_cut = train_e_idx_l[batch_idx]
            label_l_cut = train_label_l[batch_idx]

            size = len(src_l_cut)
            _, dst_l_fake = train_rand_sampler.sample(negatives * size)
            optimizer.zero_grad()
            model.train()
            loss = model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                size = len(src_l_cut)
                _, dst_l_fake = train_rand_sampler.sample(size)
                pos_prob, neg_prob = model.inference(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                ap.append(average_precision_score(true_label, pred_score))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))

        val_ap, val_auc = eval_one_epoch(model, val_rand_sampler, val_src_l, val_dst_l,
                                         val_ts_l, val_label_l, val_e_idx_l)
        total_epoch_time = time.time() - start_epoch
        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('epoch: {}:'.format(epoch))
        logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
        logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))

        if epoch == 0:
            checkpoint_dir = '/'.join(model.get_checkpoint_path(0).split('/')[:-1])
            # model.ngh_finder.save_ngh_stats(checkpoint_dir)
            model.save_common_node_percentages(checkpoint_dir)

        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), model.get_checkpoint_path(epoch))

    model.update_ngh_finder(full_ngh_finder)
    test_ap, test_auc = eval_one_epoch(model, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l,
                                       test_e_idx_l)

    # test_new_new_ap, test_new_new_auc, test_new_old_ap, test_new_old_auc = [-1] * 4
    test_new_ap, test_new_auc = eval_one_epoch(model, test_rand_sampler, test_src_new_l,
                                               test_dst_new_l, test_ts_new_l,
                                               test_label_new_l, test_e_idx_new_l)

    test_new_new_ap, test_new_new_auc = eval_one_epoch(model, test_rand_sampler, test_src_new_new_l,
                                                       test_dst_new_new_l, test_ts_new_new_l,
                                                       test_label_new_new_l, test_e_idx_new_new_l)

    test_new_old_ap, test_new_old_auc = eval_one_epoch(model, test_rand_sampler, test_src_new_old_l,
                                                       test_dst_new_old_l, test_ts_new_old_l,
                                                       test_label_new_old_l, test_e_idx_new_old_l)

    logger.info(
        'Training --- Transductive: Test statistics: {} Old nodes -- auc: {}, ap: {}'.format(args.mode, test_auc, test_ap))
    logger.info('Training --- Inductive: Test statistics: {} New- nodes ---- auc: {}, ap: {}'.format(args.mode, test_new_auc,
                                                                                        test_new_ap))
    logger.info('Training --- Inductive: Test statistics: {} New-Old nodes - auc: {}, ap: {}'.format(args.mode, test_new_old_auc,
                                                                                        test_new_old_ap))
    logger.info('Training --- Inductive: Test statistics: {} New-New nodes - auc: {}, ap: {}'.format(args.mode, test_new_new_auc,
                                                                                        test_new_new_ap))
    nn_test_auc_list.append(test_new_auc)
    nn_test_ap_list.append(test_new_ap)
    new_old_test_auc_list.append(test_new_old_auc)
    new_old_test_ap_list.append(test_new_old_ap)
    new_new_test_auc_list.append(test_new_new_auc)
    new_new_test_ap_list.append(test_new_new_ap)
    test_auc_list.append(test_auc)
    test_ap_list.append(test_ap)

    logger.info('Saving model...')
    torch.save(model.state_dict(), best_model_path)
    logger.info('Saved model to {}'.format(best_model_path))
    logger.info('model saved')
logger.info(
    'AVG+STD Transductive: ---------------- Old  nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(test_auc_list), np.std(test_auc_list), np.average(test_ap_list), np.std(test_ap_list)))
logger.info(
    'AVG+STD Test statistics: Inductive: -- New- nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(nn_test_auc_list), np.std(nn_test_auc_list), np.average(nn_test_ap_list),
        np.std(nn_test_ap_list)))
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
