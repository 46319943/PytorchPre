import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence, pack_padded_sequence
import adabound

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def data_filter():
    df = pd.read_csv('node.csv')
    df.head()

    # 根据路线ID进行分组
    df_route = df.groupby('route_id')
    df_route_length = df_route.size().sort_values()
    # 计算各个路线长度下，有多少条路线
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
    return df


def preprocess(df):
    cluster_unique = np.sort(df['cluster'].unique())
    cluster_to_idx = {cluster: idx for idx, cluster in enumerate(cluster_unique)}

    route_id_unique = df['route_id'].unique()

    # ['ID', 'node', 'node_lenth', 'route_id', 'sentiment_score', 're_sentiment', 'cluster', 'latitude', 'longitude']
    df.columns

    sentiment_sequence_list = []
    cluster_idx_sequence_list = []
    for route_id in route_id_unique:
        df_route = df[df['route_id'] == route_id]

        sentiment_sequence = df_route['sentiment_score'].values
        cluster_sequence = df_route['cluster'].values
        cluster_idx_sequence = [cluster_to_idx[cluster] for cluster in cluster_sequence]

        sentiment_sequence_list.append(
            tensor(sentiment_sequence, dtype=torch.float32).to(device)
        )
        cluster_idx_sequence_list.append(
            tensor(cluster_idx_sequence).to(device)
        )

    choice_index = np.random.choice(
        len(sentiment_sequence_list),
        math.floor(0.2 * len(sentiment_sequence_list)),
        replace=False
    )

    sentiment_sequence_list_train = [sentiment_sequence for index, sentiment_sequence in
                                     enumerate(sentiment_sequence_list) if index not in choice_index]
    sentiment_sequence_list_valid = [sentiment_sequence for index, sentiment_sequence in
                                     enumerate(sentiment_sequence_list) if index in choice_index]
    cluster_idx_sequence_list_train = [cluster_idx_sequence for index, cluster_idx_sequence in
                                       enumerate(cluster_idx_sequence_list) if index not in choice_index]
    cluster_idx_sequence_list_valid = [cluster_idx_sequence for index, cluster_idx_sequence in
                                       enumerate(cluster_idx_sequence_list) if index in choice_index]

    sentiment_result_list_train = [sentiment_sequence[-1] for sentiment_sequence in sentiment_sequence_list_train]
    sentiment_result_list_valid = [sentiment_sequence[-1] for sentiment_sequence in sentiment_sequence_list_valid]

    from torch.nn.utils.rnn import pack_sequence
    packed = pack_sequence(cluster_idx_sequence_list_train, enforce_sorted=False)

    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)

    from torch.nn.utils.rnn import pad_sequence
    padded = pad_sequence(cluster_idx_sequence_list_train, batch_first=True)

    return (
        sentiment_sequence_list_train, cluster_idx_sequence_list_train, sentiment_result_list_train,
        sentiment_sequence_list_valid, cluster_idx_sequence_list_valid, sentiment_result_list_valid
    )


def model_pipeline(sentiment_sequence, cluster_idx_sequence):
    embedding_input = 9
    embedding_dimension = 5
    lstm_hidden_dimension = 6

    embedding_layer = nn.Embedding(embedding_input, embedding_dimension)
    cluster_embedding_sequence_tensor = embedding_layer(cluster_idx_sequence)
    sentiment_sequence_tensor = sentiment_sequence.view(-1, 1)
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
    def __init__(self, embedding_input, embedding_dimension, lstm_hidden_dimension, num_layers):
        super(Model, self).__init__()
        self.embedding_layer = nn.Embedding(embedding_input, embedding_dimension)
        self.lstm_layer = nn.LSTM(embedding_dimension + 1, lstm_hidden_dimension, num_layers, batch_first=True)
        self.linear_layer = nn.Linear(lstm_hidden_dimension * num_layers + embedding_dimension, 1)

    def forward(self, sentiment_sequence_list, cluster_idx_sequence_list):
        # 分离输入。
        # 去除情感序列最后一列
        sentiment_sequence_list = [sentiment_sequence_tensor[:-1] for sentiment_sequence_tensor in
                                   sentiment_sequence_list]
        # 去除cluster序列最后一列，
        cluster_idx_sequence_list = [cluster_idx_sequence_tensor[:-1] for cluster_idx_sequence_tensor in
                                     cluster_idx_sequence_list]
        # 提取cluster序列最后一列，转为张量，进行嵌入
        cluster_idx_last_list = [cluster_idx_sequence_tensor[-1] for cluster_idx_sequence_tensor in
                                 cluster_idx_sequence_list]
        cluster_idx_last = tensor(cluster_idx_last_list, device=device)
        linear_input_embedding = self.embedding_layer(cluster_idx_last)

        # 获取序列长度
        seq_len = [len(sentiment_sequence_tensor) for sentiment_sequence_tensor in sentiment_sequence_list]

        # padding + embedding
        sentiment_padded_sequence = pad_sequence(sentiment_sequence_list, batch_first=True).unsqueeze(-1)
        cluster_idx_padded_sequence = pad_sequence(cluster_idx_sequence_list, batch_first=True)
        cluster_embedding_padded_sequence = self.embedding_layer(cluster_idx_padded_sequence)

        # concatenate + pack
        cat_padded_tensor = torch.cat(
            (sentiment_padded_sequence, cluster_embedding_padded_sequence),
            dim=-1
        )
        cat_packed_tensor = pack_padded_sequence(cat_padded_tensor, seq_len, enforce_sorted=False, batch_first=True)

        lstm_out, (lstm_hidden, lstm_cell) = self.lstm_layer(cat_packed_tensor)

        linear_input = torch.cat([lstm_hidden.permute(1, 0, 2).flatten(-2), linear_input_embedding], dim=-1).flatten(1)
        linear_out = self.linear_layer(linear_input)

        return linear_out


def train(
        sentiment_sequence_list_train, cluster_idx_sequence_list_train, sentiment_result_list_train,
        sentiment_sequence_list_valid, cluster_idx_sequence_list_valid, sentiment_result_list_valid
):
    embedding_input = 9
    embedding_dimension = 5
    lstm_hidden_dimension = 6
    num_layers = 1

    model = Model(embedding_input, embedding_dimension, lstm_hidden_dimension, num_layers)
    model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 400)
    loss_func = nn.MSELoss()

    y_true_valid = tensor(sentiment_result_list_valid)

    def model_evaluate():
        from sklearn.metrics import r2_score
        with torch.no_grad():
            y_predict_list = model(sentiment_sequence_list_valid, cluster_idx_sequence_list_valid)
        r2 = r2_score(y_true_valid, y_predict_list.cpu())
        print(r2)
        return r2

    y_true = tensor(sentiment_result_list_train, device=device).unsqueeze(-1)
    for epoch in range(1600):
        optimizer.zero_grad()
        output = model(sentiment_sequence_list_train, cluster_idx_sequence_list_train)
        loss = loss_func(output, y_true)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 49:
            print(f'loss in epoch {epoch}: {loss.item()}')
            print('evaluate:')
            model_evaluate()
            scheduler.step()

    print()


def main():
    df = data_filter()
    (
        sentiment_sequence_list_train, cluster_idx_sequence_list_train, sentiment_result_list_train,
        sentiment_sequence_list_valid, cluster_idx_sequence_list_valid, sentiment_result_list_valid
    ) = preprocess(df)
    train(
        sentiment_sequence_list_train, cluster_idx_sequence_list_train, sentiment_result_list_train,
        sentiment_sequence_list_valid, cluster_idx_sequence_list_valid, sentiment_result_list_valid
    )


if __name__ == '__main__':
    main()
