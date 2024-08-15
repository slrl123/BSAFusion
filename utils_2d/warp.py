import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from kornia.filters.kernels import get_gaussian_kernel2d
import kornia
import kornia.filters as KF
import kornia.geometry.transform as KGT
import kornia.utils as KU

class Warper2d(nn.Module):
    def __init__(self):
        super(Warper2d, self).__init__()

        """
        warp an image/tensor (im2) back to im1, according to the optical flow
#        img_src: [B, 1, H1, W1] (source image used for prediction, size 32)
        img_smp: [B, 1, H2, W2] (image for sampling, size 44)
        flow: [B, 2, H1, W1] flow predicted from source image pair
        """

    def forward(self, flow, img):
        H, W = flow.size()[2], flow.size()[3]

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W)
        yy = yy.view(1, H, W)
        grid = torch.cat((xx, yy), 0).float()  # [2, H, W]

        device = img.device

        if img.is_cuda:
            grid = grid.to(device)

        vgrid = Variable(grid, requires_grad=False) + flow
        vgrid[:, 0] = 2.0 * vgrid[:, 0] / (H - 1) - 1.0  # max(W-1,1)
        vgrid[:, 1] = 2.0 * vgrid[:, 1] / (W - 1) - 1.0  # max(H-1,1)

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(img, vgrid)

        return output  # *mask


class warp2D:
    def __init__(self, padding=False):
        self.padding = padding

    def __call__(self, I, flow):
        return self._transform(I, flow[:, 0, :, :], flow[:, 1, :, :])

    def _meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones(height, 1),
                           torch.linspace(0.0, float(width) - 1.0, width).unsqueeze(0))

        y_t = torch.matmul(torch.linspace(0.0, float(height) - 1.0, height).unsqueeze(1),
                           torch.ones(1, width))

        return x_t, y_t

    def _transform(self, I, dx, dy):
        device = I.device
        batch_size = dx.shape[0]
        height = dx.shape[1]
        width = dx.shape[2]

        # Convert dx and dy to absolute locations
        x_mesh, y_mesh = self._meshgrid(height, width)
        x_mesh = x_mesh.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        y_mesh = y_mesh.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self._interpolate(I, x_new, y_new)

    def _repeat(self, x, n_repeats):
        rep = torch.ones(n_repeats, dtype=torch.int)
        x = torch.matmul(x.view([-1, 1]).int(), rep.unsqueeze(0))
        return x.view([-1])

    def _interpolate(self, im, x, y):
        device = im.device
        if self.padding:
            im = F.pad(im, (1, 1, 1, 1))

        num_batch = im.shape[0]
        channels = im.shape[1]
        height = im.shape[2]
        width = im.shape[3]

        out_height = x.shape[1]
        out_width = x.shape[2]

        x = x.view([-1])
        y = y.view([-1])

        padding_constant = 1 if self.padding else 0
        x = x.float() + padding_constant
        y = y.float() + padding_constant

        max_x = int(width - 1)
        max_y = int(height - 1)

        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim1 = width

        base = self._repeat(torch.arange(num_batch) * dim1 * height,
                            out_height * out_width).to(device)

        idx_a = (base + x0 + y0 * dim1)[:, np.newaxis].repeat(1, channels)
        idx_b = (base + x0 + y1 * dim1)[:, np.newaxis].repeat(1, channels)
        idx_c = (base + x1 + y0 * dim1)[:, np.newaxis].repeat(1, channels)
        idx_d = (base + x1 + y1 * dim1)[:, np.newaxis].repeat(1, channels)

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.permute(0, 2, 3, 1).contiguous().view([-1, channels]).float()

        Ia = torch.gather(im_flat, 0, idx_a.long())
        Ib = torch.gather(im_flat, 0, idx_b.long())
        Ic = torch.gather(im_flat, 0, idx_c.long())
        Id = torch.gather(im_flat, 0, idx_d.long())

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy)[:, np.newaxis]
        wb = (dx * (1 - dy))[:, np.newaxis]
        wc = ((1 - dx) * dy)[:, np.newaxis]
        wd = ((1 - dx) * (1 - dy))[:, np.newaxis]

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        output = output.view([-1, out_height, out_width, channels])

        return output.permute(0, 3, 1, 2)


class ImageTransform_1(nn.Module):
    def __init__(self, ET_kernel_size=101, ET_kernel_sigma=16, AT_translate=0.01, AT_degrees=0):
        super(ImageTransform_1, self).__init__()
        self.ET_kernel_size = ET_kernel_size
        self.ET_kernel_sigma = ET_kernel_sigma
        self.AT_translate = AT_translate
        self.AT_degrees = AT_degrees

    def elastic_transform(self, x, kernel_size, sigma):

        batch_size, _, height, weight = x.shape
        device = x.device
        noise = torch.rand(batch_size, 2, height, weight, device=device) * 2 - 1

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(x)}")

        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

        if not len(x.shape) == 4:
            raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {x.shape}")

        if not len(noise.shape) == 4 or noise.shape[1] != 2:
            raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

        disp = KF.gaussian_blur2d(noise, kernel_size=(kernel_size, kernel_size),
                                  sigma=(sigma, sigma)).permute(0, 2, 3, 1)
        return disp

    def affine_transform(self, x):
        h, w = x.shape[2], x.shape[3]
        rand_angle = (torch.rand(1) * 2 - 1) * self.AT_degrees
        rand_trans = (torch.rand(1, 2) * 2 - 1) * self.AT_translate
        M = KGT.get_affine_matrix2d(translations=rand_trans, center=torch.zeros(1, 2), scale=torch.ones(1, 2),
                                    angle=rand_angle)
        M = M.inverse()
        grid = KU.create_meshgrid(h, w).to(x.device)
        warp_grid = kornia.geometry.linalg.transform_points(M, grid)
        return warp_grid - grid

    def generate_grid(self, input):
        device = input.device
        batch_size, _, height, weight = input.size()
        # affine transform
        affine_disp = self.affine_transform(input)  # warped, warped_grid_sample, disp
        # elastic transform
        elastic_disp = self.elastic_transform(input, self.ET_kernel_size, self.ET_kernel_sigma)
        # make grid
        base = kornia.utils.create_meshgrid(height, weight).to(dtype=input.dtype).repeat(batch_size, 1, 1, 1).to(device)
        disp = affine_disp + elastic_disp
        grid = base + disp
        # grid : original到warp的变换
        # 使用F.grid_sample单步采样实现”仿射变换+弹性变换“的效果
        # F.grid_sample(image, grid, align_corners=False, mode='bilinear')
        return grid, disp

    def forward(self, image):
        # generate grid that affine and elastic
        grid, disp = self.generate_grid(image)
        image_warp = F.grid_sample(image, grid, align_corners=False, mode='bilinear')

        return image_warp, disp
