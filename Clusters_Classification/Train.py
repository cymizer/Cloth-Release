import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from Clusters_Classification.feature_dataset import feature_dataset
from Clusters_Classification.feature_cls_model import feature_cls_model
from Clusters_Classification.myconfig import cfg


import copy
import os
import matplotlib.pyplot as plt
import numpy as np

import joblib


def train(features, feature_labels, epoch=50, save_model=True, draw=True):

    train_set = feature_dataset(features, feature_labels, state="train")
    train_loader = torch.utils.data.DataLoader(
        # ,sampler=ImbalancedDatasetSampler(train_set)
        train_set, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=False)

    # load weight and model
    # model = torch.load(cfg.PATH_TO_WEIGHTS)
    # model.to(cfg.GPU_ID)

    feature_size = features.shape[1]
    num_cluster = max(feature_labels)+1
    model = feature_cls_model(
        num_cluster=num_cluster, feature_size=feature_size).to(cfg.GPU_ID)
    #optimizer = optim.SGD(model.parameters(), lr=cfg.LR, momentum=cfg.MOMENTUM)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.CrossEntropyLoss()
    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0
    loss_list = []
    acc_list = []
    print("begin train")
    model.train()

    for i in range(epoch):
        print(f'Epoch: {i+1}/{epoch}')
        print('-' * len(f'Epoch: {i+1}/{epoch}'))

        training_loss = 0.0
        training_corrects = 0

        for batch_idx, (feature, label) in enumerate(train_loader):

            feature = feature.to(cfg.GPU_ID)
            label = label.to(cfg.GPU_ID)

            #print("get data")
            optimizer.zero_grad()
            output = model(feature)  # label prediction

            _, pred = torch.max(output, 1)
            training_corrects += torch.sum(pred.data == label.data)

            loss = criterion(output, label)
            training_loss += loss.item() * feature.size(0)
            loss.backward()
            optimizer.step()

            # if batch_idx % cfg.LOG_INTERVAL == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\t corrects:{:.4f}\tLoss: {:.4f}'.format(
            #         i+1, batch_idx * len(feature), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), training_corrects, loss.item()))

        training_loss = training_loss / len(train_set)
        training_acc = training_corrects.double() / len(train_set)

        loss_list.append(training_loss)
        acc_list.append(training_acc)

        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())
        if (i+1) % 5 == 0 and save_model:
            model.load_state_dict(best_model_params)
            torch.save(
                model, f'{cfg.MODEL_PATH}/features-{best_acc:.02f}-train_acc_epoch{i+1}.pth')

    # DRAW
    _, ax1 = plt.subplots()
    ax1.plot(np.arange(epoch), loss_list)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('train loss')
    ax1.set_title('feature_cls_model')
    plt.savefig(f'{cfg.Result_save}/Loss_Curve_{epoch}.png')
    if draw:
        plt.show()
    
    return model


if __name__ == "__main__":
    m = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Cloth-Release/test/material_fv.npy')
    t = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Cloth-Release/test/texture_fv.npy')

    c = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Cloth-Release/test/color.npy')

    features = np.concatenate((m, t, c), axis=1)
    kms = joblib.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Cloth-Release/test/60-means.pkl')
    train(features, kms.labels_, epoch=25,draw=False)
