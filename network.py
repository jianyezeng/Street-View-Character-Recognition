import torch.nn as nn
import torchvision.models as models
class svhn(nn.Module):
    def __init__(self):
        super(svhn, self).__init__()

        # resnet18
        self.model_conv = models.resnet50(pretrained=True)
        self.model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model_conv = nn.Sequential(*list(self.model_conv.children())[:-1])  # 去除最后一个fc layer
        self.cnn = self.model_conv

        self.fc1 = nn.Linear(2048, 11)
        self.fc2 = nn.Linear(2048, 11)
        self.fc3 = nn.Linear(2048, 11)
        self.fc4 = nn.Linear(2048, 11)
        self.fc5 = nn.Linear(2048, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)

        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)

        return c1,c2,c3,c4,c5