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

        # self.not_training = [self.conv1a]
        self.not_training = []
        # self.from_scratch_layers = []
        self.from_scratch_layers = [self.fc8, self.fc8_seg_conv1, self.fc8_seg_conv2]


    def forward(self, x, require_seg = True, require_mcam = True):
        x = super().forward(x)
        if require_seg == True and require_mcam == True:
            x_cam = x.clone()
            x_seg = x.clone()

            x = self.dropout7(x)

            x = F.avg_pool2d(
                x, kernel_size=(x.size(2), x.size(3)), padding=0)

            x = self.fc8(x)
            x = x.view(x.size(0), -1)

            cam = F.conv2d(x_cam, self.fc8.weight)
            cam = F.relu(cam)

            x_seg = self.fc8_seg_conv1(x_seg)
            x_seg = F.relu(x_seg)
            x_seg = self.fc8_seg_conv2(x_seg)

            return x, cam, x_seg
        elif require_mcam == True and require_seg== False:
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

    def forward(self, x, require_seg=True, require_mcam= True):
        if require_seg == True and require_mcam == True:
            input_size_h = x.size()[2]
            input_size_w = x.size()[3]
            # self.interp1 = nn.UpsamplingBilinear2d(size=(int(input_size_h * 0.5), int(input_size_w * 0.5)))
            # self.interp2 = nn.UpsamplingBilinear2d(size=(int(input_size_h * 1.5), int(input_size_w * 1.5)))
            # self.interp3 = nn.UpsamplingBilinear2d(size=(int(input_size_h * 2), int(input_size_w * 2)))
            # x2 = self.interp1(x)
            # x3 = self.interp2(x)
            # x4 = self.interp3(x)
            x2 = F.interpolate(x, size=(int(input_size_h * 0.5), int(input_size_w * 0.5)), mode='bilinear',align_corners=False)
            x3 = F.interpolate(x, size=(int(input_size_h * 1.5), int(input_size_w * 1.5)), mode='bilinear',align_corners=False)
            x4 = F.interpolate(x, size=(int(input_size_h * 2), int(input_size_w * 2)), mode='bilinear',align_corners=False)

            seg = []
            with torch.enable_grad():
                xf1, cam1, seg1 = super().forward(x,require_seg=True, require_mcam=True)
            with torch.no_grad():
                cam2 = super().forward(x2,require_seg=False, require_mcam=True)
                cam3 = super().forward(x3,require_seg=False, require_mcam=True)
                cam4 = super().forward(x4,require_seg=False, require_mcam=True)

            xf_temp = xf1

            cam2 = F.interpolate(cam2, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear',align_corners=False)
            cam3 = F.interpolate(cam3, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear',align_corners=False)
            cam4 = F.interpolate(cam4, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear',align_corners=False)
            # self.interp_final = nn.UpsamplingBilinear2d(size=(int(seg1.shape[2]), int(seg1.shape[3])))
            # cam2 = self.interp_final(cam2)
            # cam3 = self.interp_final(cam3)
            # cam4 = self.interp_final(cam4)

            cam = (cam1+cam2+cam3+cam4)/4

            seg.append(seg1)  # for original scale

            return xf_temp, cam, seg

        if require_mcam == False and require_seg == False:
            xf = super().forward(x,require_seg=False,require_mcam=False)
            self.not_training = [self.conv1a]
            return xf
        if require_mcam == False and require_seg == True:
            xf, cam, seg = super().forward(x, require_seg=True, require_mcam=True)
            return seg



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