import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

def transform_token2index(sequences):
    token2index = {'A': 1, 'G': 2, 'U': 3, 'C': 4, 'X': 0}
    max_len = 41
    token_list = []
    padded_seqs = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + 'X' * (max_len - len(seq))  
        else:
            padded_seq = seq[:max_len] 
        padded_seqs.append(padded_seq)
    for seq in padded_seqs:
        seq_id = [token2index[aa] for aa in seq]
        token_list.append(torch.tensor(seq_id))
    return token_list
def construct_dataset(seqs, labels, train=True, batch_size=64):
    seqs, labels = list(seqs), list(labels)
    token_list = transform_token2index(seqs)
    seqs_data = rnn_utils.pad_sequence(token_list, batch_first=True)  # Fill the sequence to the same length
    data_loader = Data.DataLoader(Data.TensorDataset(seqs_data, torch.LongTensor(labels)),
                                  batch_size=batch_size,
                                  shuffle=train,
                                  drop_last=False)
    return data_loader

def load_bench_data(file):
    tmp = pd.read_csv(file)
    seqs, labels = tmp["seq"].values.tolist(), tmp["label"].values.tolist() 
    train_seqs, test_seqs, train_labels, test_labels = train_test_split(seqs, labels, test_size=0.2, random_state=42)
    train_iter = construct_dataset(train_seqs,train_labels, train =True)
    valid_iter = construct_dataset(test_seqs,test_labels, train = False)
    return train_iter, valid_iter

def load_ind_data(file):
    tmp = pd.read_csv(file)
    seqs, labels = tmp["seq"].values.tolist(), tmp["label"].values.tolist()
    data_iter = construct_dataset(seqs, labels, train=False)
    return data_iter
