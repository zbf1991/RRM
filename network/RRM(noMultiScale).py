import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self):
        super().__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, 20, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)

        self.fc8_seg_conv2 = nn.Conv2d(512, 21, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8, self.fc8_seg_conv1, self.fc8_seg_conv2]


    def forward(self, x):
        x = super().forward(x)

        x_cam = x

        x_seg = self.fc8_seg_conv1(x)
        x_seg = F.relu(x_seg)
        x_seg = self.fc8_seg_conv2(x_seg)

        x = self.dropout7(x)

        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x = self.fc8(x)
        x = x.view(x.size(0), -1)

        cam = F.conv2d(x_cam, self.fc8.weight)
        cam = F.relu(cam)

        return x, cam, x_seg

    def forward_cam(self, x):
        x = super().forward(x)

        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups