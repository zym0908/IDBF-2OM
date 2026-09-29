import torch
from termcolor import colored

from MyDataset import load_ind_data
from Mymodel import abcmodel
from until import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model


if __name__ == '__main__':
    file = "data/G_test.csv"
    test_iter = load_ind_data(file)
    path_pretrain_model = "Result/G2OM_BiFAM-2OM.pt"
    model = abcmodel().to(device)
    model = load_model(model, path_pretrain_model)
    model.eval()
    with torch.no_grad():
        for x, label in test_iter:
            if device:
                x, label = x.to(device), label.to(device)

        ind_performance, ind_roc_data, ind_prc_data, _ = evaluate(test_iter, model)
    ind_results = '\n' + '=' * 16 + colored(' Independent Test Set Performance', 'red') + '=' * 16 \
                  + '\n[ACC,\tREC-SN,\t\tPRE,\t\tMCC,\tAUROC,\tAUPRC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        ind_performance[0],
        ind_performance[1], ind_performance[2], ind_performance[3], ind_performance[4], ind_performance[5]) + '\n' + '=' * 60
    print(ind_results)

