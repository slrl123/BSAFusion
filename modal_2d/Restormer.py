import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange


class MDTA(nn.Module):
    def __init__(self, out_c):
        super(MDTA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        self.conv4 = nn.Conv2d(out_c, out_c, 1, 1, 0)
    def forward(self, x):
        x_o = x
        x = F.layer_norm(x, x.shape[-2:])
        C , W, H = x.size()[1], x.size()[2], x.size()[3]
        q = self.conv1(x)
        q = rearrange(q, 'b c w h -> b (w h) c')
        k = self.conv2(x)
        k = rearrange(k, 'b c w h -> b c (w h)')
        v = self.conv3(x)
        v = rearrange(v, 'b c w h -> b (w h) c')

        A = torch.matmul(k, q)
        A = rearrange(A, 'b c1 c2 -> b (c1 c2)', c1=C, c2=C)
        A = torch.softmax(A, dim=1)
        A = rearrange(A, 'b (c1 c2) -> b c1 c2', c1=C, c2=C)

        v = torch.matmul(v, A)
        v = rearrange(v, 'b (h w) c -> b c h w', c = C, h=H, w=W)
        return self.conv4(v) + x_o

class GDFN(nn.Module):
    def __init__(self, out_c):
        super(GDFN, self).__init__()

        self.Dconv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c*4, 1, 1, 0),
            nn.Conv2d(out_c*4, out_c*4, 3, 1, 1)
        )
        self.Dconv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c * 4, 1, 1, 0),
            nn.Conv2d(out_c * 4, out_c * 4, 3, 1, 1)
        )
        self.conv = nn.Conv2d(out_c * 4, out_c, 1, 1, 0)
    def forward(self, x):
        x_o = x
        x = F.layer_norm(x, x.shape[-2:])
        x = F.gelu(self.Dconv1(x)) * self.Dconv2(x)
        x = x_o + self.conv(x)
        return x


class Restormer(nn.Module):
    def __init__(self, in_c, out_c):
        super(Restormer, self).__init__()
        self.mlp = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.mdta = MDTA(out_c)
        self.gdfn = GDFN(out_c)
    def forward(self, feature):
        feature = self.mlp(feature)
        feature = self.mdta(feature)
        return self.gdfn(feature)