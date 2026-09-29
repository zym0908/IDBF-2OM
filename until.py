from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    for x, label in data_iter:
        x, label = x.to(device), label.to(device)
        #output_tran, outputs, rep = net(x)  
        outputs = net(x).to(device)
        pred_prob_positive = outputs[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + label.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    #rep_plt(np.array(X), label_real) 
    return performance, roc_data, prc_data, label_real

def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    ACC = float(tp + tn) / test_num
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)
    # F1分数
    if Precision + Recall == 0:
        F1 = 0
    else:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)
    AUC = auc(FPR, TPR)
    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)
    performance = [ACC, Recall, Precision, MCC, AUC, AP]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    l2_lambda = 0.001  
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.norm(param, p=2)
    total_loss = criterion(output, label) + l2_lambda * regularization_loss
    return total_loss
