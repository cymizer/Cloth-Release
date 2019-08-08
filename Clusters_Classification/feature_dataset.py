import os
import torch.utils.data as data
import torch
import numpy as np
from sklearn.model_selection import train_test_split


class feature_dataset(data.Dataset):
    """
    """

    def __init__(self, features, feature_labels, state="train"):

        self.features = features.astype(np.float32)
        self.labels = feature_labels.astype(np.int64)

        num = [i for i in range(len(features))]
        train, test = train_test_split(
            num, random_state=2, train_size=0.8)
        train.sort()
        test.sort()

        self.data_use = train if state == "train" else test

    def __len__(self):
        return len(self.data_use)

    def __getitem__(self, index):
        return self.features[self.data_use[index]], self.labels[self.data_use[index]]


if __name__ == "__main__":
    m = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Hipster/feature/all_pca_m.npy')
    t = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Hipster/feature/all_pca_t.npy')
    c = np.load(
        'Z:/Users/cymb103u/Desktop/WorkSpace/Hipster/feature/all_pca_c.npy')
    features = np.concatenate((m, t, c), axis=1)
    labels = None
    FTD = feature_dataset(features, labels, "train")
    print()
