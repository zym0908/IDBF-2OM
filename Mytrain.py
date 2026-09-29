import os
import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt 
from termcolor import colored
from sklearn.model_selection import StratifiedKFold
from MyDataset import construct_dataset, load_ind_data
from Mymodel import abcmodel
from until import reg_loss, evaluate
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=20, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_acc = None

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        self.best_acc = val_acc
        path = 'best_network.pt'
        torch.save(model.state_dict(), path)


def train_test(train_iter, test_iter, iter_k):
    global best_ROC
    net = abcmodel().to(device)
    lr = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15, delta=0.0001)

    best_acc = 0
    EPOCH = 100
    best_ROC = 0
    best_PRC = 0
    best_results = []
    best_performance = []
    best_ind_performance = []

    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()

        net.train()
        for x, label in train_iter:
            if device:
                x, label = x.to(device), label.to(device)

            # output, attn = net(x)
            # print(f'attn: {attn}')
            output = net(x)
            loss = reg_loss(net, output, label).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())

        net.eval()
        with torch.no_grad():
            train_performance, _, _, _ = evaluate(train_iter, net)
            test_performance, test_roc_data, test_prc_data, label_real = evaluate(test_iter, net)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 13 + ' validation Performance. Epoch[{}] '.format(epoch + 1) + '=' * 13 \
                   + '\n[ACC,\tREC-SN,\t\tPRE,\t\tMCC,\tAUROC,\tAUPRC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[1], test_performance[2], test_performance[3],
            test_performance[4], test_performance[5]) + '\n' + '=' * 60
        print(results)

        test_acc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]

        if test_acc > 0.89:
            filename = '{}, {}[{:.4f}].pt'.format(
                'mRNA_Model' + ', {}fold'.format(iter_k + 1) + ', epoch[{}]'.format(epoch + 1), 'ACC', test_acc)
            save_path_pt = os.path.join('./Result', filename)
            torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)

        scheduler.step(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_performance = test_performance

            best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tREC-SN,\t\tPRE,\t\tMCC,\tAUROC,\tAUPRC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}' \
                               .format(best_performance[0], best_performance[1], best_performance[2],
                                       best_performance[3], best_performance[4], best_performance[5]) \
                           + '\n' + '=' * 60
            best_ROC = test_roc_data
            best_PRC = test_prc_data

        early_stopping(test_acc, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    best_ROC = np.array(best_ROC, dtype=object)
    best_PRC = np.array(best_PRC, dtype=object)

    roc_filename = '{}, {}折, {}[{:.4f}], {}[{:.4f}].npy'.format('ROC_', iter_k + 1, 'AUROC', best_performance[4],
                                                                 'ACC', best_acc)
    full_path_roc = os.path.join('./Result', roc_filename)
    np.save(full_path_roc, best_ROC)
    prc_filename = '{}, {}折, {}[{:.4f}], {}[{:.4f}].npy'.format('PRC_', iter_k + 1, 'AUPRC', best_performance[5],
                                                                 'ACC', best_acc)
    full_path_prc = os.path.join('./Result', prc_filename)
    np.save(full_path_prc, best_PRC)

    print(best_results)
    return best_performance, best_results, best_ROC, best_PRC


def K_CV(file, k):
    tmp = pd.read_csv(file)
    seqs = tmp["seq"]
    labels = tmp["label"]
    seqs, labels = np.array(seqs), np.array(labels)
    CV_perform = []

    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for iter_k, (train_index, test_index) in enumerate(kfold.split(seqs, labels)):
        print("\n" + "=" * 16 + "k = " + str(iter_k + 1) + "=" * 16)

        train_seqs, test_seqs = seqs[train_index], seqs[test_index]
        train_lables, test_labels = labels[train_index], labels[test_index]
        train_iter = construct_dataset(train_seqs, train_lables, train=True)
        test_iter = construct_dataset(test_seqs, test_labels, train=False)
        performance, _, ROC, PRC = train_test(train_iter, test_iter, iter_k)
        CV_perform.append(performance)

    print('\n' + '=' * 16 + colored(' Cross-Validation Performance ',
                                    'red') + '=' * 16 + '\n[ACC,\tREC-SN,\t\tPRE,\t\tMCC,\tAUROC,\tAUPRC]\n')
    for out in np.array(CV_perform):
        print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(out[0], out[1], out[2], out[3], out[4],
                                                                           out[5]))
    mean_out = np.array(CV_perform).mean(axis=0)
    print('\n' + '=' * 16 + "Mean out" + '=' * 16)
    print(
        '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(mean_out[0], mean_out[1], mean_out[2], mean_out[3],
                                                                     mean_out[4], mean_out[5]))
    print('\n' + '=' * 60)


if __name__ == '__main__':
    # k-fold cross-validation
    K_CV('data/A_train.csv', k=10)
