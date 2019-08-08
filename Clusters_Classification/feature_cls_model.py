import torchvision
import torch
import torch.nn as nn
import sys


class feature_cls_model(nn.Module):
    """
    input: N * (feature_size)
    output: N * num_cluster_label
    """

    def __init__(self, num_cluster, feature_size):
        super(feature_cls_model, self).__init__()
        print("set backbone")
        self.backbone = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU()
        )

        # classifier
        self.classifier = nn.Linear(256, num_cluster)

        print("end of model building")

    def forward(self, x):
        # print("forward")
        x = self.backbone(x)
        # flatten
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


if __name__ == "__main__":
    # conv1 = nn.Conv1d(in_channels=835, out_channels=64, kernel_size=1)
    # inputs = torch.randn(32, 835, 1)
    # out = conv1(inputs)
    # print(out.size())
    pass