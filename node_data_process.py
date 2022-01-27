import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('node.csv')
df.head()
df_route = df.groupby('route_id')
df_route_length = df_route.size().sort_values()
df_length = df_route_length.reset_index().groupby(0).size()
df_length.plot(kind='bar')

route_count = df_route_length.shape[0]
route_length_split = 15
# 去除长度为后90%的路线
route_count, np.sum(df_route_length <= route_length_split) / route_count, np.sum(
    df_route_length > route_length_split) / route_count
# 长度小于等于3的路线占53%
np.sum(df_route_length <= 3) / route_count
# 保留长度再4和15之间的37%的数据
np.sum((df_route_length >= 4) & (df_route_length <= 15)) / route_count

# 路线筛选
route_id_include = df_route_length[(df_route_length >= 4) & (df_route_length <= 15)].index.values
df = df[df['route_id'].isin(route_id_include)]

cluster_unique = np.sort(df['cluster'].unique())
cluster_to_idx = {cluster: idx for idx, cluster in enumerate(cluster_unique)}

route_id_unique = df['route_id'].unique()

# ['ID', 'node', 'node_lenth', 'route_id', 'sentiment_score', 're_sentiment', 'cluster', 'latitude', 'longitude']
df.columns

sentiment_sequence = df[df['route_id'] == 6]['sentiment_score']
cluster_sequence = df[df['route_id'] == 6]['cluster']
cluster_idx_sequence = [cluster_to_idx[cluster] for cluster in cluster_sequence]

import torch
import torch.nn as nn
import torch.tensor as tensor

embedding_dimension = 5
lstm_hidden_dimension = 6

embedding_layer = nn.Embedding(len(cluster_unique), embedding_dimension)
cluster_embedding_sequence_tensor = embedding_layer(tensor(cluster_idx_sequence))
sentiment_sequence_tensor = tensor(sentiment_sequence.values, dtype=torch.float32).view(-1, 1)
cat_tensor = torch.cat((sentiment_sequence_tensor, cluster_embedding_sequence_tensor), 1)
lstm_input_tensor = cat_tensor[:-1, :]
linear_input_embedding = cluster_embedding_sequence_tensor[[-1], :].view(1, 1, -1)

lstm_layer = nn.LSTM(embedding_dimension + 1, lstm_hidden_dimension, batch_first=True)
lstm_out, (lstm_hidden, lstm_cell) = lstm_layer(lstm_input_tensor.view(1, *lstm_input_tensor.shape))

linear_layer = nn.Linear(embedding_dimension + lstm_hidden_dimension, 1)
linear_out = linear_layer(torch.cat([lstm_hidden, linear_input_embedding], dim=-1))

loss_func = nn.MSELoss()
loss_func(linear_out, sentiment_sequence_tensor[-1])


class Model(nn.Module):
    def __init__(self, embedding_dimension, lstm_hidden_dimension):
        super(Model, self).__init__()
        self.embedding_layer = nn.Embedding(len(cluster_unique), embedding_dimension)
        self.lstm_layer = nn.LSTM(embedding_dimension + 1, lstm_hidden_dimension, batch_first=True)
        self.linear_layer = nn.Linear(lstm_hidden_dimension + embedding_dimension, 1)

    def forward(self, sentiment_sequence_tensor, cluster_idx_sequence):
        cluster_embedding_sequence_tensor = self.embedding_layer(cluster_idx_sequence)
        cat_tensor = torch.cat((sentiment_sequence_tensor, cluster_embedding_sequence_tensor), dim=-1)

        lstm_input_tensor = cat_tensor[:, :-1, :]
        linear_input_embedding = cluster_embedding_sequence_tensor[:, [-1], :]

        lstm_out, (lstm_hidden, lstm_cell) = lstm_layer(lstm_input_tensor)
        linear_out = linear_layer(torch.cat([lstm_hidden, linear_input_embedding], dim=-1).flatten(1))

        return linear_out


model = Model(embedding_dimension, lstm_hidden_dimension)

sentiment_sequence_list = []
cluster_idx_sequence_list = []
for route_id in route_id_unique:
    df_route = df[df['route_id'] == route_id]

    sentiment_sequence = df_route['sentiment_score'].values
    cluster_sequence = df_route['cluster'].values
    cluster_idx_sequence = [cluster_to_idx[cluster] for cluster in cluster_sequence]

    sentiment_sequence_list.append(tensor(sentiment_sequence, dtype=torch.float32))
    cluster_idx_sequence_list.append(tensor(cluster_idx_sequence))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

import math

choice_index = np.random.choice(len(sentiment_sequence_list), math.floor(0.2 * len(sentiment_sequence_list)),
                                replace=False)


def model_evaluate():
    from sklearn.metrics import r2_score

    y_true_list = []
    y_predict_list = []
    with torch.no_grad():
        for index, (sentiment_sequence, cluster_idx_sequence) in enumerate(
                zip(sentiment_sequence_list, cluster_idx_sequence_list)):
            if index not in choice_index:
                continue
            output = model(sentiment_sequence.view(1, -1, 1), cluster_idx_sequence.view(1, -1))

            y_true_list.append(sentiment_sequence[-1])
            y_predict_list.append(output[0][0])

    r2 = r2_score(y_true_list, y_predict_list)
    print(r2)
    return r2


for epoch in range(500):
    running_loss = 0
    for index, (sentiment_sequence, cluster_idx_sequence) in enumerate(
            zip(sentiment_sequence_list, cluster_idx_sequence_list)):
        if index in choice_index:
            continue
        optimizer.zero_grad()
        output = model(sentiment_sequence.view(1, -1, 1), cluster_idx_sequence.view(1, -1))
        loss = loss_func(output, sentiment_sequence[-1].view(1, 1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if index % 500 == 499:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, index + 1, running_loss / 500))
            running_loss = 0.0

    print('evaluate:')
    model_evaluate()
    scheduler.step()

print()
