import torch
import torch.nn as nn

nc = 3  # 通道数
nef = 64  # encoder的第一层层数层数
nBottleneck = 4000
ngf = nef
ndf = 64


class CoderGan(nn.Module):
    def __init__(self):
        super(CoderGan, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nef, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nef * 8, nBottleneck, 4, bias=False),
            nn.BatchNorm2d(nBottleneck),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(nBottleneck, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, d_in):
        d_out = self.layer(d_in)
        return d_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, d_in):
        d_out = self.layer(d_in)
        return d_out.view(-1, 1)


class ContentNet(nn.Module):
    def __init__(self):
        super(ContentNet, self).__init__()
        self.layer = nn.Sequential(
            # 第一层
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.ELU(alpha=0.2, inplace=True),
            # 第二层
            nn.Conv2d(nef, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            nn.ELU(alpha=0.2, inplace=True),
            # 第三层
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.ELU(alpha=0.2, inplace=True),
            # 第四层
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.ELU(alpha=0.2, inplace=True),
            # 第五层
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.ELU(alpha=0.2, inplace=True),
            # encoder瓶颈
            nn.Conv2d(nef * 8, nBottleneck, 4, bias=False),
            nn.BatchNorm2d(nBottleneck),
            nn.ELU(alpha=0.2, inplace=True),
            # 第一层
            nn.ConvTranspose2d(nBottleneck, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ELU(alpha=0.2, inplace=True),
            # 第二层
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(alpha=0.2, inplace=True),
            # 第三层
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(alpha=0.2, inplace=True),
            # 第四层
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ELU(alpha=0.2, inplace=True),
            # 输出层
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, d_in):
        d_out = self.layer(d_in)
        return d_out


