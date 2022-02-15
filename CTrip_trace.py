import numpy as np

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


def load_data():
    import pickle
    with open('location_sequence_list.pkl', 'rb') as f:
        location_sequence_list = pickle.load(f)

    with open('sentiment_sequence_list.pkl', 'rb') as f:
        sentiment_sequence_list = pickle.load(f)

    # location_sequence_dict = {}
    # for location_sequence in location_sequence_list:
    #     location_sequence = tuple(location_sequence.squeeze().tolist())
    #     location_sequence_count = location_sequence_dict.get(location_sequence, 0)
    #     location_sequence_dict[location_sequence] = location_sequence_count + 1
    #
    # location_sequence_list = [location_sequence.squeeze() for location_sequence in location_sequence_list if
    #                           2 <= len(location_sequence) <= 6 and location_sequence_dict[location_sequence] > 5]

    location_sequence_list = [location_sequence.squeeze() for location_sequence in location_sequence_list if
                              3 <= len(location_sequence) <= 10]
    sentiment_sequence_list = [sentiment_sequence.squeeze() for sentiment_sequence in sentiment_sequence_list if
                               3 <= len(sentiment_sequence) <= 10]

    location_whole = np.concatenate(location_sequence_list)
    location_unique = np.unique(location_whole)
    location_to_idx = {location: idx for idx, location in enumerate(location_unique)}

    location_idx_sequence_list = [
        tensor([
            location_to_idx[location] for location in location_sequence
        ]).to(device) for location_sequence in location_sequence_list
    ]
    sentiment_sequence_list = [
        tensor(sentiment_sequence, dtype=torch.float32).to(device) for sentiment_sequence in sentiment_sequence_list
    ]

    import math
    choice_index = np.random.choice(
        len(sentiment_sequence_list),
        math.floor(0.2 * len(sentiment_sequence_list)),
        replace=False
    )

    sentiment_sequence_list_train = [sentiment_sequence for index, sentiment_sequence in
                                     enumerate(sentiment_sequence_list) if index not in choice_index]
    sentiment_sequence_list_valid = [sentiment_sequence for index, sentiment_sequence in
                                     enumerate(sentiment_sequence_list) if index in choice_index]
    location_idx_sequence_list_train = [location_idx_sequence for index, location_idx_sequence in
                                        enumerate(location_idx_sequence_list) if index not in choice_index]
    location_idx_sequence_list_valid = [location_idx_sequence for index, location_idx_sequence in
                                        enumerate(location_idx_sequence_list) if index in choice_index]

    sentiment_result_list_train = [sentiment_sequence[-1] for sentiment_sequence in sentiment_sequence_list_train]
    sentiment_result_list_valid = [sentiment_sequence[-1] for sentiment_sequence in sentiment_sequence_list_valid]

    return (
        sentiment_sequence_list_train, location_idx_sequence_list_train, sentiment_result_list_train,
        sentiment_sequence_list_valid, location_idx_sequence_list_valid, sentiment_result_list_valid
    )


if __name__ == '__main__':
    load_data()
    print()
