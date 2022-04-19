import torch
import torch.nn as nn


class Yolov1(nn.Module):
    def __init__(self):
        super(Yolov1, self).__init__()

        # 根据卷积的原理，当kernel的步长碰到了边界外的，应该向下取整
        # Block1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block3
        self.conv3_a = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        self.conv3_b = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_c = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv3_d = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block4
        self.conv4_a = self.Layer4Component(repeat=4)
        self.conv4_b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.conv4_c = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block5
        self.conv5_a = self.Layer5Component(repeat=2)
        self.conv5_b = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_c = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)

        # Block6
        self.conv6_a = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv6_b = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        # Classifier
        # self.ln7 = nn.Linear(in_features=7 * 7 * 1024, out_features=4096)
        # self.ln8 = nn.Linear(in_features=4096, out_features=630)
        self.fc = self.Classifier()


    def Classifier(self):
        classify = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 1024, out_features=4096),
            nn.LeakyReLU(),
            nn.Linear(in_features=4096, out_features=1470)  # S * S * (B * 5 + C)
        )
        return classify

    def Layer4Component(self, repeat=4):
        layers = []

        for i in range(repeat):
            layers.append(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1))
            layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    def Layer5Component(self, repeat=2):
        layers = []

        for i in range(repeat):
            layers.append(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1))
            layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.conv1(x)
        h = self.max1(h)

        h = self.conv2(h)
        h = self.max2(h)

        h = self.conv3_a(h)
        h = self.conv3_b(h)
        h = self.conv3_c(h)
        h = self.conv3_d(h)
        h = self.max3(h)

        h = self.conv4_a(h)
        h = self.conv4_b(h)
        h = self.conv4_c(h)
        h = self.max4(h)

        h = self.conv5_a(h)
        h = self.conv5_b(h)
        h = self.conv5_c(h)

        h = self.conv6_a(h)
        h = self.conv6_b(h)

        h = h.flatten(start_dim=1)  # start_dim - end_dim乘起来，其他维度不变
        # h = self.ln7(h)
        # h = self.ln8(h)
        h = self.fc(h)
        h = torch.sigmoid(h)

        return torch.reshape(h, (-1, 7, 7, 30))


# aa = torch.randn((1, 3, 448, 448))
# model = Yolov1()
# print(model)
# result = model(aa)
#
# print("sdd")
