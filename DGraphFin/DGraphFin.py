import numpy as np
import pandas as pd

data = np.load('dgraphfin.npz')

print("Arrays in the npz file:", list(data.keys()))

node_features = data["x"]
print("shape of node features", node_features.shape)

node_labels = data["y"]
print("shape of node labels", node_labels.shape)

edge_indexs = data["edge_index"]
print(type(edge_indexs))
print("shape of node labels", edge_indexs.shape)

# number of edges
edges_num = len(edge_indexs)
print("Edges in the npz file:", edges_num)

def is_sorted_ascending(arr):
    return np.all(arr[:-1] <= arr[1:])

def is_sorted_descending(arr):
    return np.all(arr[:-1] >= arr[1:])

edge_timestamps = data["edge_timestamp"]
print("shape of edge timestamps", edge_timestamps.shape)

print(is_sorted_ascending(edge_timestamps)) 
print(is_sorted_descending(edge_timestamps)) 

edge_types = data["edge_type"]
print("shape of edge types", edge_types.shape)

# start nodes 
start_nodes = edge_indexs[:, 0]
start_nodes_num = len(np.unique(start_nodes))
print("start_nodes_num", start_nodes_num)
# end nodes
end_nodes = edge_indexs[:, 1]
end_nodes_num = len(np.unique(end_nodes))
print("end_nodes_num", end_nodes_num)

# concatenate
all_users = np.concatenate((start_nodes, end_nodes), axis=-1)
nodes_num = len(np.unique(all_users))
print("nodes_num", nodes_num)


DGraphFin = pd.DataFrame({
    'u': start_nodes,
    'i': end_nodes,
    'ts': edge_timestamps,
    'edge_label': edge_types
    })

start_nodes_labels_dynamic = node_labels[DGraphFin["u"].values]
print(start_nodes_labels_dynamic.shape)
DGraphFin['label'] =  start_nodes_labels_dynamic

# time sort
DGraphFin.sort_values("ts", inplace=True)
# sorted
print("sorted")
print(is_sorted_ascending(DGraphFin["ts"].values)) 
print(is_sorted_descending(DGraphFin["ts"].values)) 

# edge_indexs
edge_indexs = np.arange(1, edges_num+1)
DGraphFin['idx'] =  edge_indexs

DGraphFin_new = DGraphFin
interaction_num = len(DGraphFin_new.u)
print(interaction_num)

concatenate_index = np.concatenate((DGraphFin_new.u, DGraphFin_new.i), axis=0)
print(concatenate_index.shape)
all_index, nodes = pd.factorize(concatenate_index)
print("max_node_index: "+str(max(all_index)))
DGraphFin_new.u = all_index[0:interaction_num]
DGraphFin_new.i = all_index[interaction_num:interaction_num+interaction_num]

DGraphFin_new.u += 1
DGraphFin_new.i += 1

DGraphFin_new_path = "./ml_DGraphFin.csv"
selected_columns = ['u', 'i', 'ts', 'label', 'idx']
DGraphFin_new[selected_columns].to_csv(DGraphFin_new_path, index=False)

node_features_new_path = "./ml_DGraphFin_node.npy"
indexes_new = np.unique(concatenate_index, return_index=True)[1]
order_index = np.array([concatenate_index[index] for index in sorted(indexes_new)])
print("order_index.shape", order_index.shape)
node_features_new = node_features[order_index]
print("type(node_features_new)", type(node_features_new))
print("type(node_features_new.dtype)", node_features_new.dtype)
print("node_features_new.shape", node_features_new.shape)
zero_row = np.zeros(node_features_new.shape[1])[np.newaxis, :]
node_features_new = np.vstack((zero_row, node_features_new))
print("node_features_new.shape", node_features_new.shape)
print("type(node_features_new.dtype)", node_features_new.dtype)
np.save(node_features_new_path, node_features_new)

edge_types_sorted = DGraphFin_new["edge_label"].values
one_hot_encoded = np.eye(11)[edge_types_sorted - 1]
empty = np.zeros(one_hot_encoded.shape[1])[np.newaxis, :]
one_hot_encoded = np.vstack([empty, one_hot_encoded])
print(one_hot_encoded.shape)
edge_features_path = "./ml_DGraphFin.npy"
np.save(edge_features_path, one_hot_encoded)

