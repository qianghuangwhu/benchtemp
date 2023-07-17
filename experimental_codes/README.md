# Running the experiments

## Table of Contents
  - [Requirements](#requirements)
  - [Datasets](#datasets)
  - [Model Training](#model-training)
    - [Link Prediction Task](#link-prediction-task)
      - [TGN](#tgn)
      - [JODIE](#jodie)
      - [DyRep](#dyrep)
      - [TGAT](#tgat)
      - [CAWN](#cawn)
      - [NeurTW](#neurtw)
      - [NAT](#nat)
    - [Node Classification Task](#node-classification-task)
      - [TGN](#tgn-1)
      - [JODIE](#jodie-1)
      - [DyRep](#dyrep-1)
      - [TGAT](#tgat-1)
      - [CAWN](#cawn-1)
      - [NeurTW](#neurtw-1)
      - [NAT](#nat-1)

## Requirements
Please ensure that you have installed the following dependencies:
```bash
numpy >= 1.18.0
pandas>=1.1.0
torch>=1.6.0
scikit_learn>=0.23.1
# benchmark library: benchtemp 
benchtemp==1.1.1
```

## Datasets
Download the benchmark datasets from [Here](https://drive.google.com/drive/folders/1HKSFGEfxHDlHuQZ6nK4SLCEMFQIOtzpz?usp=sharing) and store those files in a folder named "data".

## Model Training

### Link Prediction Task
#### TGN
```bash
python train_link_prediction.py --use_memory --n_runs 3 --gpu 0  --data mooc
```

#### JODIE
```bash
python train_link_prediction.py --use_memory --memory_updater rnn --embedding_module time --n_runs 3 --gpu 0 --data mooc
```

#### DyRep
```bash
python train_link_prediction.py --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --n_runs 3 --gpu  0 --data mooc
```
#### TGAT
```bash
python -u train_link_prediction.py --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --n_head 2 --n_runs 3  -d mooc --gpu  0
```
**parameter: "--n_head" for different datasets**

<center>

|             | edge_dim | node_dim | --n_head  |
|-------------|----------|----------|----------|
| reddit      | 172      | 172      | 2        |
| wikipedia   | 172      | 172      | 2        |
| mooc        | 4        | 172      | 2        |
| lastfm      | 2        | 172      | 2        |
| enron       | 32       | 172      | 2        |
| SocialEvo   | 2        | 172      | 2        |
| uci         | 100      | 172      | 2        |
| CollegeMsg  | 172      | 172      | 2        |
| TaobaoSmall | 4        | 172      | 2        |
| CanParl     | 1        | 172      | 1        |
| Contacts    | 1        | 172      | 1        |
| Flights     | 1        | 172      | 1        |
| UNtrade     | 1        | 172      | 1        |
| USLegis     | 1        | 172      | 1        |
| UNvote      | 1        | 172      | 1        |

</center>


#### CAWN
```bash
python train_link_prediction.py   --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0  --n_runs 3 --pos_dim 100 -d mooc --gpu 0
```
**parameter: "--pos_dim" for different datasets**

<center>

|             | edge_dim | node_dim |  --pos_dim |
|-------------|-----|-----|------------|
| reddit      | 172 | 172 | 108        |
| wikipedia   | 172 | 172 | 108        |
| mooc        | 4   | 172 | 100        |
| lastfm      | 2   | 172 | 102        |
| enron       | 32  | 172 | 104        |
| SocialEvo   | 2   | 172 | 102        |
| uci         | 100 | 172 | 100        |
| CollegeMsg  | 172 | 172 | 108        |
| TaobaoSmall | 4   | 172 | 100        |
| CanParl     | 1   | 172 | 103        |
| Contacts    | 1   | 172 | 103        |
| Flights     | 1   | 172 | 103        |
| UNtrade     | 1   | 172 | 103        |
| USLegis     | 1   | 172 | 103        |
| UNvote      | 1   | 172 | 103        |
</center>

#### NeurTW
```bash
python train_link_prediction.py  --data_usage 1.0 --mode i --n_degree 64 1 --pos_dim 108 --pos_sample multinomial --pos_enc lp --temporal_bias 0.0001 --spatial_bias 0.01 --ee_bias 2 --tau 0.1 --negs 1 --solver rk4 --step_size 0.125 --bs 32  --seed 0 --limit_ngh_span --ngh_span 320 8 --n_runs 1 --gpu 0 -d mooc
```

#### NAT
```bash
python train_link_prediction.py  --pos_dim 16 --bs 100 --n_degree 16 --n_hop 1 --mode i --bias 1e-5 --seed 2 --verbosity 1 --drop_out 0.1 --attn_n_head 1 --ngh_dim 4 --self_dim 72 --n_run 3 --gpu 0 -d mooc 
```

### Node Classification Task

#### TGN
```bash
python train_node_classification.py --use_memory --n_runs 3 --gpu 0  --data mooc --use_validation
```
#### JODIE
```bash
python train_node_classification.py --use_memory --memory_updater rnn --embedding_module time --n_runs 3 --gpu 0 --data mooc --use_validation
```
#### DyRep
```bash
python train_node_classification.py --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --n_runs 3 --gpu 0 --data mooc --use_validation
```
#### TGAT
```bash
python -u train_node_classification.py -d mooc --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2
```
**parameter: "--n_head" for different datasets**

<center>

|             | edge_dim | node_dim | --n_head  |
|-------------|----------|----------|----------|
| reddit      | 172      | 172      | 2        |
| wikipedia   | 172      | 172      | 2        |
| mooc        | 4        | 172      | 2        |
| lastfm      | 2        | 172      | 2        |
| enron       | 32       | 172      | 2        |
| SocialEvo   | 2        | 172      | 2        |
| uci         | 100      | 172      | 2        |
| CollegeMsg  | 172      | 172      | 2        |
| TaobaoSmall | 4        | 172      | 2        |
| CanParl     | 1        | 172      | 1        |
| Contacts    | 1        | 172      | 1        |
| Flights     | 1        | 172      | 1        |
| UNtrade     | 1        | 172      | 1        |
| USLegis     | 1        | 172      | 1        |
| UNvote      | 1        | 172      | 1        |

</center>

#### CAWN
```bash
python train_node_classification.py   --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0  --n_runs 3 --pos_dim 100 -d mooc --gpu 0
```
**parameter: "--pos_dim" for different datasets**

<center>

|             | edge_dim | node_dim |  --pos_dim |
|-------------|-----|-----|------------|
| reddit      | 172 | 172 | 108        |
| wikipedia   | 172 | 172 | 108        |
| mooc        | 4   | 172 | 100        |
| lastfm      | 2   | 172 | 102        |
| enron       | 32  | 172 | 104        |
| SocialEvo   | 2   | 172 | 102        |
| uci         | 100 | 172 | 100        |
| CollegeMsg  | 172 | 172 | 108        |
| TaobaoSmall | 4   | 172 | 100        |
| CanParl     | 1   | 172 | 103        |
| Contacts    | 1   | 172 | 103        |
| Flights     | 1   | 172 | 103        |
| UNtrade     | 1   | 172 | 103        |
| USLegis     | 1   | 172 | 103        |
| UNvote      | 1   | 172 | 103        |
</center>

#### NeurTW
```bash
python train_node_classification.py  --data_usage 1.0 --mode i --n_degree 64 1 --pos_dim 108 --pos_sample multinomial --pos_enc lp --temporal_bias 0.0001 --spatial_bias 0.01 --ee_bias 2 --tau 0.1 --negs 1 --solver rk4 --step_size 0.125 --bs 32  --seed 0 --limit_ngh_span --ngh_span 320 8 --n_runs 1 --gpu 0 -d mooc
```
#### NAT
```bash
python train_node_classification.py  --pos_dim 16 --bs 100 --n_degree 16 --n_hop 1 --mode i --bias 1e-5 --seed 2 --verbosity 1 --drop_out 0.1 --attn_n_head 1 --ngh_dim 4 --self_dim 72 --n_run 3 --gpu 0 -d mooc 
```
