import torch
from torch import nn
from monai.losses.ssim_loss import SSIMLoss
import torch.nn.functional as F


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

def L1_loss(tensor1, tensor2):
    loss = nn.L1Loss()
    l = loss(tensor1, tensor2)
    return l

def r_loss(flow):
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d / 3.0
    return grad

def ssim_loss(img1, img2):
    # img1 = normalize(img1)
    # img2 = normalize(img2)
    # print(img1.shape)
    # print(img2.shape)
    device = img1.device
    data_range = torch.tensor(1.0).to(device)
    return SSIMLoss(spatial_dims=2)(img1, img2)

def gradient_loss(fusion_img, img1, img2):
    grad_filter = Sobelxy().requires_grad_(False)
    grad_filter.to(fusion_img.device)
    fusion_img_g = grad_filter(fusion_img)
    max_g_img1_2 = torch.maximum(grad_filter(img1), grad_filter(img2))
    return L1_loss(fusion_img_g, max_g_img1_2)

def regFusion_loss(label1, label2,
                   pre1, pre2,
                   feature_pred1, feature_pred2,
                   flow, flows, warped_img2, flow_GT,
                   img1, img1_2, fusion_img,
                   parameter):

    cls_loss = nn.CrossEntropyLoss()(pre1, label1) +  nn.CrossEntropyLoss()(pre2, label2)
    grad_filter = Sobelxy().requires_grad_(False)
    grad_filter.to(fusion_img.device)
    trans_label = torch.tensor([0.5, 0.5]).expand(feature_pred1.shape[0], -1).to(feature_pred1.device)
    transfer_loss = nn.CrossEntropyLoss()(feature_pred1, trans_label)  + nn.CrossEntropyLoss()(feature_pred2, trans_label)
    flow_loss = torch.tensor(0.0).to(feature_pred1.device)
    alpha = 0.0001
    for i in range(len(flows) // 2):
        flow_loss += (r_loss(flows[i]) + r_loss(flows[i + 1])) * alpha
        alpha *= 10
    flow_loss += r_loss(flow)

    ssim1 = ssim_loss(fusion_img, img1)
    ssim2 = ssim_loss(fusion_img, warped_img2)
    fu_loss = ssim1 + parameter*ssim2 + 0.5*L1_loss(fusion_img, torch.maximum(img1, warped_img2)) + gradient_loss(fusion_img, img1, warped_img2)
    reg_loss = ssim_loss(img1_2, warped_img2) + L1_loss(img1_2, warped_img2)

    return cls_loss, transfer_loss, flow_loss, fu_loss, reg_loss, ssim1, ssim2
