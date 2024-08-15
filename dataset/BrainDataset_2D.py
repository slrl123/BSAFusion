import os
import cv2
import kornia
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from utils_2d.utils import rgb2ycbcr, padding_img_, padding_img
from utils_2d.warp import ImageTransform_1, warp2D, Warper2d
import pathlib

def min_max(data):
    min = torch.min(data)
    max = torch.max(data)
    data = (data - min) / (max - min)
    return data

class RegDataset_F(Dataset):

    def __init__(self, root, mode, model, is_normalize=True):
        super(RegDataset_F).__init__()
        self.root = root
        self.mode = mode
        self.model = model
        self.is_normalize = is_normalize
        csv_name = model + '_MRI.csv'
        data = pd.read_csv(os.path.join(root, csv_name))
        if mode == 'train':
            self.data_list = data['Train'].dropna().tolist()
            self.data_list = self.data_list
        elif mode == 'test':
            self.data_list = data['Test'].dropna().tolist()

        self.transform=transforms.Compose([
        transforms.RandomHorizontalFlip,
        transforms.RandomVerticalFlip,
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
            ])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        model_path1 = os.path.join(self.root, f'{self.model}-MRI/{self.model}/{self.data_list[item]}')
        model_path2 = os.path.join(self.root, f'{self.model}-MRI/MRI/{self.data_list[item]}')
        model_img1 = torch.from_numpy(np.array(Image.open(model_path1))).to(dtype=torch.float32)
        model_img2 = torch.from_numpy(np.array(Image.open(model_path2))).to(dtype=torch.float32)



        if self.model == 'CT':
            model_img1 = model_img1.unsqueeze(0)
            model_img2 = model_img2.unsqueeze(0)
        else:
            model_img1 = model_img1.permute(2, 0, 1)
            model_img2 = model_img2.unsqueeze(0)

        if self.mode == 'train':
            # make sure transform in same paradigm
            seed = torch.seed()
            torch.manual_seed(seed)
            model_img1 = self.transform(model_img1)
            torch.manual_seed(seed)
            model_img2 = self.transform(model_img2)

        model_img1, model_img2 = model_img1.unsqueeze(0), model_img2.unsqueeze(0)
        model_img1, model_img2 = model_img1.to(dtype=torch.float32), model_img2.to(dtype=torch.float32)
        img_list = [model_img2, model_img1]
        A_1, B_1 = img_list[0].squeeze(0), img_list[1].squeeze(0)
        _, flow = ImageTransform_1()(img_list[1])
        flow = flow.permute(0, 3, 1, 2) * 256
        B_2 = warp2D()(img_list[1], flow).squeeze(0)
        flow_GT = -flow.squeeze(0)


        data_list = [A_1, B_1, B_2]

        if self.is_normalize:
            for i in [0, 1, 2]:
                data_list[i] = min_max(data_list[i])

        if self.model == 'CT':
            pass
        else:
            data_list[1], data_list[2] = rgb2ycbcr(data_list[1].unsqueeze(0)).squeeze(0), rgb2ycbcr(data_list[2].unsqueeze(0)).squeeze(0)
        data_list.append(flow_GT)
        data_list.append(torch.tensor([0.0, 1.0]))
        data_list.append(torch.tensor([1.0, 0.0]))
        return data_list # T1, t2, warpedT2, label1, label2


class TestData(Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, img1_folder, img2_folder, modal):
        super(TestData, self).__init__()
        img1_folder = pathlib.Path(img1_folder)
        img2_folder = pathlib.Path(img2_folder)
        # gain infrared and visible images list

        self.img1_list = [x for x in sorted(img1_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.img2_list = [x for x in sorted(img2_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.modal = modal

    def __len__(self):
        return len(self.img1_list)

    def __getitem__(self, index):
        img1_path = self.img1_list[index]
        img2_path = self.img2_list[index]
        file_name_with_extension = os.path.basename(img1_path)
        if self.modal=='CT':
            # read image as type Tensor
            img1 = self.imread(path=img1_path)
            img2 = self.imread(path=img2_path)
        else:

            img1 = self.imread(path=img1_path)
            img2 =(torch.from_numpy(np.array(Image.open(img2_path).convert('RGB')))/255.).permute(2, 0, 1)
        return (img1, img2, file_name_with_extension)  # fake
    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts
