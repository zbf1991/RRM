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

        self.GAU_w1 = nn.Conv2d(4096, 256, (1, 1), bias=False)
        self.GAU_w2 = nn.Conv2d(4096, 256, (1, 1), bias=False)
        self.GAU_w3 = nn.Conv2d(4096, 256, (1, 1), bias=False)
        self.fc8_seg_conv3 = nn.Conv2d(512, 256, (3, 3), stride=1, padding=1, dilation=1, bias=False)
        self.fc8_seg_conv4 = nn.Conv2d(256, 21, (3, 3), stride=1, padding=1, dilation=1, bias=False)

        # self.GAU_leakyrelu = nn.LeakyReLU(0.2)

        self.not_training = []
        self.from_scratch_layers = [self.fc8]
        # self.from_scratch_layers = [self.fc8, self.fc8_seg_conv1, self.fc8_seg_conv2,self.fc8_seg_conv3,self.fc8_seg_conv4]


    def forward(self, x,require_cam=True,require_seg=True):
        x = super().forward(x)

        if require_cam==True and require_seg==True:
            xAt = x.clone()

            x_cam = x.clone()

            x_seg = x.clone()

            x = self.dropout7(x)

            x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)

            x = self.fc8(x)
            x = x.view(x.size(0), -1)

            cam = F.conv2d(x_cam, self.fc8.weight)
            cam = F.relu(cam)

            GSeg1 = self.GAU_w1(xAt)
            GSeg1 = GSeg1.view([GSeg1.size(0), GSeg1.size(1), -1]).permute(0,2,1)
            GSeg2 = self.GAU_w2(xAt)
            GSeg2 = GSeg2.view([GSeg2.size(0), GSeg2.size(1), -1])
            attention = F.softmax(torch.bmm(GSeg1,GSeg2),dim=1)
            GSeg3 = self.GAU_w3(xAt)
            v_m = GSeg3.view([GSeg3.size(0), GSeg3.size(1), -1])
            v_a = torch.bmm(v_m,attention)
            v_a = v_a.view([GSeg3.size(0), GSeg3.size(1), GSeg3.size(2), GSeg3.size(3)])

            new_feature = torch.cat([v_a,GSeg3],dim=1)
            x_seg2 = self.fc8_seg_conv3(new_feature)
            x_seg2 = F.relu(x_seg2)
            x_seg2 = self.fc8_seg_conv4(x_seg2)

            x_seg = self.fc8_seg_conv1(x_seg)
            old_feature = F.relu(x_seg)
            x_seg = self.fc8_seg_conv2(old_feature)

            x_seg = (x_seg+x_seg2)/2

            seg_feature = torch.cat([old_feature,new_feature],dim=1)

            return x, cam, x_seg, seg_feature

        elif require_cam==True and require_seg==False:
            x_cam = x.clone()
            cam = F.conv2d(x_cam, self.fc8.weight)
            cam = F.relu(cam)
            return cam
        else:
            x = self.dropout7(x)

            x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)

            x = self.fc8(x)
            x = x.view(x.size(0), -1)
            return x

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


class SegNet(Net):
    def __init__(self):
        super().__init__()

    def forward(self, x, require_cam=True,require_seg=True):
        if require_cam == True and require_seg == True:
            input_size_h = x.size()[2]
            input_size_w = x.size()[3]

            x2 = F.interpolate(x, size=(int(input_size_h * 0.5), int(input_size_w * 0.5)), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x, size=(int(input_size_h * 1.5), int(input_size_w * 1.5)), mode='bilinear', align_corners=False)
            x4 = F.interpolate(x, size=(int(input_size_h * 2), int(input_size_w * 2)), mode='bilinear', align_corners=False)

            # seg = []
            with torch.enable_grad():
                xf_temp, cam1, seg1, seg1_feature = super().forward(x,require_cam=True,require_seg=True)
            with torch.no_grad():
                cam2 = super().forward(x2,require_cam=True,require_seg=False)
                cam3 = super().forward(x3,require_cam=True,require_seg=False)
                cam4 = super().forward(x4,require_cam=True,require_seg=False)

            cam2 = F.interpolate(cam2, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear', align_corners=False)
            cam3 = F.interpolate(cam3, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear', align_corners=False)
            cam4 = F.interpolate(cam4, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear', align_corners=False)

            cam = (cam1+cam2+cam3+cam4)/4

            # seg.append(seg1)

            return xf_temp, cam, seg1, seg1_feature
        else:
            xf = super().forward(x,require_cam=False,require_seg=False)
            self.not_training = [self.conv1a]
            return xf

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