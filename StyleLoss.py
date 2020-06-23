import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLoss(nn.Module):
    def __init__(self, target, patch_size, mrf_style_stride, mrf_synthesis_stride, gpu_chunk_size, device):
        super(StyleLoss, self).__init__()
        self.patch_size = patch_size
        self.mrf_style_stride = mrf_style_stride
        self.mrf_synthesis_stride = mrf_synthesis_stride
        self.gpu_chunk_size = gpu_chunk_size
        self.device = device
        self.loss = None

        self.style_patches = self.patchesSampling(target.detach(), patch_size=self.patch_size,
                                                  stride=self.mrf_style_stride)
        self.style_patches_norm = self.calPatchesNorm()
        self.style_patches_norm = self.style_patches_norm.view(-1, 1, 1)

    def update(self, target):
        self.style_patches = self.patchesSampling(target.detach(), patch_size=self.patch_size,
                                                  stride=self.mrf_style_stride)
        self.style_patches_norm = self.calPatchesNorm()
        self.style_patches_norm = self.style_patches_norm.view(-1, 1, 1)

    def forward(self, d_in):
        sysnthesis_patches = self.contentPatchesSampling(d_in, patch_size=self.patch_size,
                                                         stride=self.mrf_synthesis_stride)
        max_response = []
        for i in range(0, self.style_patches.shape[0], self.gpu_chunk_size):
            start = i
            end = min(i + self.gpu_chunk_size, self.style_patches.shape[0])
            weight = self.style_patches[start:end, :, :, :]
            response = F.conv2d(d_in, weight, stride=self.mrf_synthesis_stride)
            max_response.append(response.squeeze(dim=0))
        max_response = torch.cat(max_response, dim=0)
        max_response = max_response.div(self.style_patches_norm)
        # 发挥最大的response,得到nn(i)
        max_response = torch.argmax(max_response, dim=0)
        max_response = torch.reshape(max_response, (1, -1)).squeeze()
        loss = 0

        for i in range(0, len(max_response), self.gpu_chunk_size):
            start = i
            end = min(i + self.gpu_chunk_size, len(max_response))
            tp_ind = tuple(range(start, end))
            sp_ind = max_response[start:end]
            loss += torch.sum(
                torch.mean(torch.pow(sysnthesis_patches[tp_ind, :, :, :] - self.style_patches[sp_ind, :, :, :], 2),
                           dim=[1, 2, 3]))
        self.loss = loss / len(max_response)
        return d_in

    def patchesSampling(self, image, patch_size, stride):
        h, w = image.shape[2:4]
        patches = []
        # 取样非mark区的patches
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                centerX = i + self.patch_size / 2
                centerY = j + self.patch_size / 2
                # 判断是否在Mark区
                bool = (centerX > h / 4) and (centerX < (h * 3 / 4)) and (centerY > w / 4) and (centerY < (w * 3 / 4))
                if not bool:
                    patches.append(image[:, :, i:i + patch_size, j:j + patch_size])
        patches = torch.cat(patches, dim=0).to(self.device)
        return patches

    def contentPatchesSampling(self, image, patch_size, stride):
        h, w = image.shape[2:4]
        patches = []
        # 取样patches
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patches.append(image[:, :, i:i + patch_size, j:j + patch_size])
        patches = torch.cat(patches, dim=0).to(self.device)
        return patches

    # 计算每张图片的范数
    def calPatchesNorm(self):
        norm_array = torch.zeros(self.style_patches.shape[0])
        for i in range(self.style_patches.shape[0]):
            norm_array[i] = torch.pow(torch.sum(torch.pow(self.style_patches[i], 2)), 0.5)
        return norm_array.to(self.device)
