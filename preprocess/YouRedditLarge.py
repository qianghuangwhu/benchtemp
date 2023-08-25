import pandas as pd
import numpy as np

# read .pkl file
data = pd.read_pickle('urls_df.pkl')

print(type(data))
column_names = list(data.columns)
for column in column_names:
    values = data[column].values
    print(column)
    print(values.shape)
    print(len(set(values)))

YoutubeReddit = pd.DataFrame({
    'u': data["url"].values,
    'i': data["subreddit"].values,
    'ts': data["timestamp"].values,
    'author':data["author"].values,
    })

# sort
YoutubeReddit.sort_values("ts", inplace=True)

edges_num = len(YoutubeReddit)
print("edges_num", edges_num)
# edge_indexs
edge_indexs = np.arange(1, edges_num+1)
YoutubeReddit['idx'] =  edge_indexs
YoutubeReddit['label'] =  np.zeros(edges_num)

# factorize
YoutubeReddit['u'], _ =  pd.factorize(YoutubeReddit['u'].values)
YoutubeReddit['u'] += 1

YoutubeReddit['i'], _ =  pd.factorize(YoutubeReddit['i'].values)
upper_u = YoutubeReddit.u.max() + 1
YoutubeReddit['i'] = YoutubeReddit['i'] + upper_u
# YoutubeReddit['i'] += YoutubeReddit['u'].nunique() + 1

print("YoutubeReddit['ts'].max()", YoutubeReddit['ts'].max())
# YoutubeReddit['ts'] = np.log2(YoutubeReddit['ts']).round(0)

YoutubeReddit_path = "./ml_YoutubeRedditLarge.csv"
selected_columns = ['u', 'i', 'ts', 'label', 'idx']
YoutubeReddit[selected_columns].to_csv(YoutubeReddit_path, index=False)


#  test 
# import numpy as np
# import pandas as pd

graph_df = pd.read_csv("./ml_YoutubeRedditLarge.csv")
print(graph_df.head(10))

Youtube_nums = YoutubeReddit['u'].nunique()
print(Youtube_nums)
Reddit_nums =  YoutubeReddit['i'].nunique()
print(Reddit_nums)
total_nums = Youtube_nums + Reddit_nums
print(total_nums)

# node features 
max_idx = max(YoutubeReddit.u.max(), YoutubeReddit.i.max())
print("max_idx", max_idx)
node_features = np.zeros((max_idx + 1, 172))
node_features_path = "./ml_YoutubeRedditLarge_node.npy"
np.save(node_features_path, node_features)

# edge features

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


vectorizer = TfidfVectorizer(max_features=172)  
edge_features = vectorizer.fit_transform(YoutubeReddit["author"].values).toarray()
print("edge_features.shape")
print(edge_features.shape)
empty = np.zeros(edge_features.shape[1])[np.newaxis, :]
edge_features = np.vstack([empty, edge_features])
print("edge_features.shape")
print(edge_features.shape)
print(type(edge_features))
edge_features_path = "./ml_YoutubeRedditLarge.npy"
np.save(edge_features_path, edge_features)

