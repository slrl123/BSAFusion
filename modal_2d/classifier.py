import monai
import torch
from einops import rearrange
from monai.networks.blocks import SABlock, MLPBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
import torch.nn as nn
import torch.nn.functional as F



def project(x, image_size):
    """将 torch.Size([1, 512, 768]) 转换成图像形状[channel, W/p, H/p]"""
    W, H = image_size[0], image_size[1]
    x = rearrange(x, 'b (w h) hidden -> b w h hidden', w=W // 16, h=H // 16)
    x = x.permute(0, 3, 1, 2)
    return x


class PatchEmbedding2D(nn.Module):
    def __init__(self, in_c, embedding_dim, patch_size):
        super(PatchEmbedding2D, self).__init__()
        self.patch_embedding = nn.Conv2d(in_c, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.position_embedding = nn.Parameter(torch.randn([1, 1, 256, 256]))
    def forward(self, x):
        x_H, x_W = x.shape[2], x.shape[3]
        position_embedding = F.interpolate(self.position_embedding, size=[x_H, x_W], mode='bilinear', align_corners=True)
        x = x + position_embedding[:, :]
        x = self.patch_embedding(x)  # b, embedding_dim, H/16, W/16
        x = rearrange(x, 'b embedding_dim h w -> b (h w) embedding_dim')
        return x


class VitBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, vit_drop, qkv_bias, mlp_dim, mlp_drop):
        super(VitBlock, self).__init__()
        self.attention = SABlock(hidden_size=hidden_size, num_heads=num_heads, dropout_rate=vit_drop, qkv_bias=qkv_bias)
        self.mlp = MLPBlock(hidden_size=hidden_size, mlp_dim=mlp_dim, dropout_rate=mlp_drop)
        self.norm_layer1 = nn.LayerNorm(hidden_size)
        self.norm_layer2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attention(self.norm_layer2(x))  # batch patch emb_dim
        x = x + self.mlp(self.norm_layer2(x))
        return x


class VIT(nn.Module):
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(VIT, self).__init__()
        self.hidden_size = 768
        self.embedding = PatchEmbeddingBlock(in_channels=in_c,
                                             img_size=img_size,
                                             patch_size=patch_size,
                                             hidden_size=768,
                                             num_heads=num_heads,
                                             pos_embed='perceptron',
                                             dropout_rate=0.0,
                                             spatial_dims=2)


        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=768,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=3072, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        x = self.embedding(x)  # image_embedding
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)

        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]

class model_classifer(nn.Module):
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(model_classifer, self).__init__()
        self.hidden_size = 768
        self.embedding = PatchEmbeddingBlock(in_channels=in_c,
                                             img_size=img_size,
                                             patch_size=patch_size,
                                             hidden_size=768,
                                             num_heads=num_heads,
                                             pos_embed='perceptron',
                                             dropout_rate=0.0,
                                             spatial_dims=2)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=768,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=3072, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)

        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]


class VIT_V2(nn.Module):
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(VIT_V2, self).__init__()
        self.hidden_size = 768
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=768, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=768,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=3072, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        x = self.embedding(x)  # image_embedding
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]


class model_classifer_V2(nn.Module):
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(model_classifer_V2, self).__init__()
        self.hidden_size = 768
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=768, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=768,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=3072, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]
