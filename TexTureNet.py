import torch.nn as nn
import torch
import torchvision.models as models
import TvLoss, StyleLoss, ContentLoss


class MrfNet(nn.Module):
    def __init__(self, style_image, content_image, device, content_weight, style_weight, tv_weight, gpu_chunk_size=256,
                 mrf_style_stride=2, mrf_synthesis_stride=2):
        super(MrfNet, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.patch_size = 3
        self.device = device
        self.gpu_chunk_size = gpu_chunk_size
        self.mrf_style_stride = mrf_style_stride

        self.mrf_synthesis_stride = mrf_synthesis_stride
        self.style_layers = [12, 21]
        self.content_layers = [22]
        self.model, self.content_losses, self.style_losses, self.tv_loss = self.getModelAndLosses(
            style_image=style_image, content_image=content_image)

    def forward(self, d_in):
        self.model(d_in)
        style_score = 0
        content_score = 0
        tv_score = self.tv_loss.loss
        for stl in self.style_losses:
            style_score += stl.loss
        for col in self.content_losses:
            content_score += col.loss
        # 联合损失
        loss = self.style_weight * style_score + self.content_weight * content_score + self.tv_weight * tv_score
        # print(loss)
        return loss

    # 构建网络以及获取损失
    def getModelAndLosses(self, style_image, content_image):
        style_image.to(self.device)
        content_image.to(self.device)
        vgg = models.vgg19(pretrained=True).to(self.device)
        model = nn.Sequential().to(self.device)
        content_losses = []
        style_losses = []
        tv_loss = TvLoss.TvLoss().to(self.device)
        # 添加为子Module
        model.add_module("tv_loss", tv_loss)
        next_content_idx = 0
        next_style_idx = 0
        for i in range(len(vgg.features)):
            if next_content_idx >= len(self.content_layers) and next_style_idx >= len(self.style_layers):
                break
            layer = vgg.features[i]
            layer = layer.to(self.device)
            model.add_module(str(i), layer)
            content_image.to(self.device)
            if i in self.content_layers:
                target = model(content_image).detach()
                content_loss = ContentLoss.ContentLoss(target)
                model.add_module("content_loss_{}".format(next_content_idx), content_loss)
                content_losses.append(content_loss)
                next_content_idx += 1
            if i in self.style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss.StyleLoss(target_feature, patch_size=self.patch_size,
                                                 mrf_style_stride=self.mrf_style_stride,
                                                 mrf_synthesis_stride=self.mrf_synthesis_stride,
                                                 gpu_chunk_size=self.gpu_chunk_size, device=self.device)
                model.add_module("style_loss_{}".format(next_style_idx), style_loss)
                style_losses.append(style_loss)
                next_style_idx += 1
        return model, content_losses, style_losses, tv_loss

    def updateStyleAndContentImage(self, style_image, content_image):
        style_image.to(self.device)
        content_image.to(self.device)
        x = style_image.clone().to(self.device)
        next_style_idx = 0
        i = 0
        for layer in self.model:
            if isinstance(layer, TvLoss.TvLoss) or isinstance(layer, StyleLoss.StyleLoss) or isinstance(layer,
                                                                                                        ContentLoss.ContentLoss):
                continue
            if next_style_idx >= len(self.style_losses):
                break
            x = layer(x)
            if i in self.style_layers:
                self.style_losses[next_style_idx].update(x)
                next_style_idx += 1
            i += 1
        x = content_image.clone().to(self.device)
        next_content_idx = 0
        i = 0
        for layer in self.model:
            if isinstance(layer, TvLoss.TvLoss) or isinstance(layer, ContentLoss.ContentLoss) or isinstance(layer,
                                                                                                            StyleLoss.StyleLoss):
                continue
            if next_content_idx >= len(self.content_losses):
                break
            x = layer(x)
            if i in self.content_layers:
                self.content_losses[next_content_idx].update(x)
                next_content_idx += 1
            i += 1
