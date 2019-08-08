import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from Clusters_Classification.myconfig import cfg
from Clusters_Classification.feature_dataset import feature_dataset
from Clusters_Classification.feature_cls_model import feature_cls_model
import copy
import matplotlib.pyplot as plt
import numpy as np


def test(features, feature_labels, model):
    """
    """
    num_cluster = max(feature_labels)+1
    test_set = feature_dataset(features, feature_labels, state="test")
    data_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=cfg.NUM_WORKERS, pin_memory=False)

    model.to(cfg.GPU_ID)
    model.eval()
    # loss setting
    criterion = nn.CrossEntropyLoss()
    corrects = 0
    test_loss = 0.0
    with torch.no_grad():
        for feature, label in data_loader:

            feature = feature.to(cfg.GPU_ID)
            label = label.to(cfg.GPU_ID)
            out = model(feature)
            loss = criterion(out, label)
            test_loss += loss.item()*feature.size(0)
            _, pred = torch.max(out, 1)
            corrects += torch.sum(pred == label)

            # print(pred)
            # print(label)
            # print(corrects)
            # input('')

        test_loss = test_loss / len(test_set)
        acc = corrects.double()/len(test_set)

        print(f'{num_cluster}-Cluster-Classifiction')
        print(f'test loss :{test_loss:.4f}')
        print(f'accuracy : {acc:.4f}')


if __name__ == '__main__':
    pass
