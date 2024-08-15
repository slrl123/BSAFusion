import os

import cv2
from scipy.ndimage.filters import gaussian_filter
import math
from PIL import Image
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch import nn


def min_max(data):
    min = torch.min(data)
    max = torch.max(data)
    data = (data - min) / (max - min)
    return data

class Transformer2D(nn.Module):
    def __init__(self):
        super(Transformer2D, self).__init__()

    def forward(self, src, flow, padding_mode="border"):
        b = flow.shape[0]
        size = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1).to(flow.device)
        new_locs = grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
        warped = F.grid_sample(src, new_locs, align_corners=True, padding_mode=padding_mode)
        return warped

def create_affine_transformation_matrix(n_dims, scaling=None, rotation=None, shearing=None, translation=None):
    """
        create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy matrix
    """

    trans_scaling = np.eye(n_dims + 1)
    trans_shearing = np.eye(n_dims + 1)
    trans_translation = np.eye(n_dims + 1)

    if scaling is not None:
        trans_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)] = np.append(scaling, 1)

    if shearing is not None:
        shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype='bool')
        shearing_index[np.eye(n_dims + 1, dtype='bool')] = False
        shearing_index[-1, :] = np.zeros((n_dims + 1))
        shearing_index[:, -1] = np.zeros((n_dims + 1))
        trans_shearing[shearing_index] = shearing

    if translation is not None:
        trans_translation[np.arange(n_dims), n_dims *
                          np.ones(n_dims, dtype='int')] = translation

    if n_dims == 2:
        if rotation is None:
            rotation = np.zeros(1)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        trans_rot = np.eye(n_dims + 1)
        trans_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation), np.sin(rotation),
                                                                     np.sin(rotation) * -1, np.cos(rotation)]
        return trans_translation @ trans_rot @ trans_shearing @ trans_scaling

    else:
        if rotation is None:
            rotation = np.zeros(n_dims)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        trans_rot1 = np.eye(n_dims + 1)
        trans_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [np.cos(rotation[0]),
                                                                      np.sin(
                                                                          rotation[0]),
                                                                      np.sin(
                                                                          rotation[0]) * -1,
                                                                      np.cos(rotation[0])]
        trans_rot2 = np.eye(n_dims + 1)
        trans_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [np.cos(rotation[1]),
                                                                      np.sin(
                                                                          rotation[1]) * -1,
                                                                      np.sin(
                                                                          rotation[1]),
                                                                      np.cos(rotation[1])]
        trans_rot3 = np.eye(n_dims + 1)
        trans_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[2]),
                                                                      np.sin(
                                                                          rotation[2]),
                                                                      np.sin(
                                                                          rotation[2]) * -1,
                                                                      np.cos(rotation[2])]
        return trans_translation @ trans_rot3 @ trans_rot2 @ trans_rot1 @ trans_shearing @ trans_scaling


def non_affine(imgs, padding_modes, opt, elastic_random=None):
    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]
    if not isinstance(padding_modes, list) and not isinstance(padding_modes, tuple):
        padding_modes = [padding_modes]

    w, h = imgs[0].shape[-2:]
    if elastic_random is None:
        elastic_random = torch.rand([2, w, h]).numpy() * 2 - 1  # .numpy()

    sigma = opt['gaussian_smoothing']   # 需要根据图像大小调整
    alpha = opt['non_affine_alpha']  # 需要根据图像大小调整

    dx = gaussian_filter(elastic_random[0], sigma) * alpha
    dy = gaussian_filter(elastic_random[1], sigma) * alpha
    dx = np.expand_dims(dx, 0)
    dy = np.expand_dims(dy, 0)
    flow = np.concatenate((dx, dy), 0)
    flow = np.expand_dims(flow, 0)
    flow = torch.from_numpy(flow).to(torch.float32)

    results = []
    for img, mode in zip(imgs, padding_modes):
        img = Transformer2D()(img.unsqueeze(0), flow, padding_mode=mode)
        results.append(img.squeeze(0))

    return results[0] if len(results) == 1 else results


def affine(random_numbers, imgs, padding_modes, opt):
    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]
    if not isinstance(padding_modes, list) and not isinstance(padding_modes, tuple):
        padding_modes = [padding_modes]

    if opt['dim'] == 3:
        tmp = np.ones(3)
        tmp[0:3] = random_numbers[0:3]
        scaling = tmp * opt['scaling'] + 1
        tmp[0:3] = random_numbers[3:6]
        rotation = tmp * opt['rotation']
        tmp[0:2] = random_numbers[6:8]
        tmp[2] = 0
        translation = tmp * opt['translation']
    else:
        scaling = random_numbers[0:2] * opt['scaling'] + 1
        rotation = random_numbers[2] * opt['rotation']
        translation = random_numbers[3] * opt['translation']

    theta = create_affine_transformation_matrix(
        n_dims=opt['dim'], scaling=scaling, rotation=rotation, shearing=None, translation=translation)
    theta = theta[:-1, :]
    theta = torch.from_numpy(theta).to(torch.float32)
    size = imgs[0].size()
    grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=True)

    res_img = []
    for img, mode in zip(imgs, padding_modes):
        res_img.append(F.grid_sample(img, grid, align_corners=True, padding_mode=mode).squeeze(0))

    return res_img[0] if len(res_img) == 1 else res_img


def random_warp_data_list(data_list, padding_batch=False):
    """
    image only
    """
    opt = {
        #####
        'dim': 2,
        'size': [256, 256],

        'rotation': 3,        # range of rotation if use affine
        'translation': 0.08,    # range of translation if use affine
        'scaling': 0.08,      # range of scaling if use affine

        # non affine
        'non_affine_alpha': 120,
        'gaussian_smoothing': 12,
    }

    random_numbers = torch.rand(8).numpy() * 2 - 1
    A_1, A_2 = affine(random_numbers=random_numbers, imgs=[data_list[0], data_list[1]], padding_modes=['border', 'border'], opt=opt)
    A_1, A_2 = non_affine(imgs=[A_1, A_2], padding_modes=['border', 'border'], opt=opt)
    return A_1, A_2  # data: C H W D

def get_visualize_image_pair(index1, index2, modal):
    root = './data'
    csv_name = modal + '_MRI.csv'
    data = pd.read_csv(os.path.join(root, csv_name))
    data_list = data['Val'].dropna().tolist()
    model_path1 = os.path.join(root, f'{modal}-MRI/{modal}/{data_list[index1]}')
    model_path2 = os.path.join(root, f'{modal}-MRI/MRI/{data_list[index2]}')
    model_img1 = torch.from_numpy(np.array(Image.open(model_path1)))
    model_img2 = torch.from_numpy(np.array(Image.open(model_path2)))
    if modal == 'CT':
        model_img1 = model_img1.unsqueeze(0)
        model_img2 = model_img2.unsqueeze(0)
    else:
        model_img1 = model_img1.unsqueeze(0)
        model_img2 = model_img2.permute(2, 0, 1)
    model_img1, model_img2 = model_img1.to(dtype=torch.float32), model_img2.to(dtype=torch.float32)
    data_list = [model_img1, model_img2]
    for i in [0, 1]:
        data_list[i] = min_max(data_list[i]).unsqueeze(0)
    return data_list

def project(x, image_size):
    """将 torch.Size([1, 512, 768]) 转换成图像形状[channel, W/p, H/p]"""
    W, H = image_size[0], image_size[1]
    x = rearrange(x, 'b (w h) hidden -> b w h hidden', w=W // 16, h=H // 16)
    x = x.permute(0, 3, 1, 2)
    return x

def value_and_plt_transfer(model, index1, index2, modal, iter_num):
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        img1, img2 = get_visualize_image_pair(index1, index2, modal)
        transed_img1, transed_img2, cls1, cls2, new_cls1, new_cls2 = model(img1.cuda(), img2.cuda())
        plt.figure("check", (18, 18))
        plt.subplot(2, 2, 1)
        plt.title("image1")
        plt.imshow(img1.cpu().numpy()[0, 0, :, :], cmap="gray")
        plt.subplot(2, 2, 2)
        plt.title("image2")
        plt.imshow(img2.cpu().numpy()[0, 0, :, :], cmap="gray")
        plt.subplot(2, 2, 3)
        plt.title("trans img1")
        plt.imshow(transed_img1.cpu().numpy()[0, 0, :, :], cmap="gray")
        plt.subplot(2, 2, 4)
        plt.title("trans img2")
        plt.imshow(transed_img2.cpu().numpy()[0, 0, :, :], cmap="gray")
        figure_save_path = "./val_figs_3_6"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path, f'exam_1_29_{iter_num}.png'))  # 第一个是指存储路径，第二个是图片名字

def get_img_pair(index, modal):
    root = './data'
    csv_name = modal + '_MRI.csv'
    data = pd.read_csv(os.path.join(root, csv_name))
    data_list = data['Val'].dropna().tolist()
    model_path1 = os.path.join(root, f'{modal}-MRI/{modal}/{data_list[index]}')
    model_path2 = os.path.join(root, f'{modal}-MRI/MRI/{data_list[index]}')
    model_img1 = torch.from_numpy(np.array(Image.open(model_path1)))
    model_img2 = torch.from_numpy(np.array(Image.open(model_path2)))
    if modal == 'CT':
        model_img1 = model_img1.unsqueeze(0)
        model_img2 = model_img2.unsqueeze(0)
    else:
        model_img1 = model_img1.unsqueeze(0)
        model_img2 = model_img2.permute(2, 0, 1)
    model_img1, model_img2 = model_img1.unsqueeze(0), model_img2.unsqueeze(0)
    model_img1, model_img2 = model_img1.to(dtype=torch.float32), model_img2.to(dtype=torch.float32)

    img_list = [model_img2, model_img1]

    opt = {
        #####
        'dim': 2,
        'size': [256, 256],

        'rotation': 3,  # range of rotation if use affine
        'translation': 0.08,  # range of translation if use affine
        'scaling': 0.08,  # range of scaling if use affine

        # non affine
        'non_affine_alpha': 120,
        'gaussian_smoothing': 12,
    }
    # A_1, B_1 = random_warp_data_list([img_list[0], img_list[1]], padding_batch=True)
    A_1, B_1 = img_list[0].squeeze(0), img_list[1].squeeze(0)
    random_numbers = torch.rand(8).numpy() * 2 - 1
    B_2 = affine(random_numbers=random_numbers, imgs=[img_list[1]], padding_modes=['border'], opt=opt)
    B_2 = non_affine(imgs=[B_2], padding_modes=['border'], opt=opt)

    data_list = [A_1, B_1, B_2]

    for i in [0, 1, 2]:
        data_list[i] = min_max(data_list[i]).unsqueeze(0)
    return data_list


def np_rgb2ycrcb(rgb_image):
    rgb_image = np.transpose(rgb_image, (2, 3, 1, 0))  # 调整维度顺序
    rgb_image = rgb_image.squeeze()  # 移除维度为1的维度

    # 将RGB图像转换为YCrCb颜色空间
    ycrcb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    return ycrcb_image

def rgb2ycbcr(rgb_image):
    H, W = rgb_image.shape[2], rgb_image.shape[3]
    device = rgb_image.device
    transform_matrix = torch.tensor([[0.257, 0.564, 0.098],
                                     [-0.148, -0.291, 0.439],
                                     [0.439, -0.368, -0.071]]).to(device)

    rgb_image = rgb_image.permute(0, 2, 3, 1).reshape(-1, 3)
    bias = torch.tensor([0.0625, 0.5, 0.5]).to(device)
    ycbcr_image = torch.matmul(rgb_image, transform_matrix.T) + bias

    ycbcr_image = ycbcr_image.reshape(1, H, W, 3).permute(0, 3, 1, 2)
    return ycbcr_image


def ycbcr2rgb(ycrcb_tensor):

    device = ycrcb_tensor.device
    H, W = ycrcb_tensor.size(2), ycrcb_tensor.size(3)

    transform_matrix = torch.tensor([[1.164, 0.000, 1.596],
                                     [1.164, -0.392, -0.813],
                                     [1.164, 2.017, 0.000]]).to(device)

    bias = torch.tensor([0.0625, 0.5, 0.5]).to(device)
    # 将YCRCB图像的通道维度调整为适合矩阵乘法的形状
    ycrcb_tensor = ycrcb_tensor.permute(0, 2, 3, 1).reshape(-1, 3)
    # 执行矩阵乘法
    rgb_tensor = torch.matmul(ycrcb_tensor-bias, transform_matrix.T)
    # 将结果重新调整为图像张量的形状
    rgb_tensor = rgb_tensor.reshape(-1, H, W, 3).permute(0, 3, 1, 2)

    return rgb_tensor


def padding_img(img, num):
    c, h, w = img.size()

    # 计算需要填充的行数和列数
    pad_h = ((h + num - 1) // num) * num - h
    pad_w = ((w + num - 1) // num) * num - w

    # 计算左右和上下填充的量
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # 使用0填充边缘
    padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)

    return padded_img

def padding_img_(img, num):
    c, h, w = img.size()
    pad_h = max(num - h, 0)
    pad_w = max(num - w, 0)
    padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return padded_img





