import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import cv2 as cv
from torchvision import transforms
import torch.nn.functional as F
from TexTureNet import MrfNet
from math import e

content_path = "data/content.jpg"
style_path = "data/style.jpg"

content_weight = 1
style_weight = 0.6
tv_weight = 0.35
gpu_chunk_size = 256
mrf_synthesis_stride = 2
mrf_style_stride = 2
max_iter = 200
sample_step = 50
num_res = 3


def getSynthesisImage(synthesis, denorm, device):
    cpu_device = torch.device('cpu')
    image = synthesis.clone().squeeze().to(cpu_device)
    image = denorm(image)
    return image.to(device).clamp_(0, 1)


def unsampleSynthesis(height, width, synthesis, device):
    synthesis = F.interpolate(synthesis, size=[height, width], mode='bilinear')
    synthesis = synthesis.clone().detach().requires_grad_(True).to(device)
    return synthesis


def train(cropped, synthesis_in, dir):
    device = torch.device("cuda:0")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    denorm_transform = transforms.Normalize(mean=(-2.12, -2.04, -1.80), std=(4.37, 4.46, 4.44))
    size = 256
    synthesis_in = F.interpolate(synthesis_in, [size, size], mode="bilinear")
    pyramid_content_image = []
    pyramid_style_image = []
    cropped.to(device)
    synthesis_in.to(device)
    for i in range(num_res):
        cropped_sub = F.interpolate(cropped, scale_factor=1 / pow(2, num_res - 1 - i), mode='bilinear')
        synthesis_in_sub = F.interpolate(synthesis_in, scale_factor=1 / pow(2, num_res - 1 - i), mode='bilinear')
        pyramid_style_image.append(cropped_sub)
        pyramid_content_image.append(synthesis_in_sub)
    global iterator
    iterator = 0
    synthesis = None
    mrf = MrfNet(style_image=pyramid_style_image[0], content_image=pyramid_content_image[0], device=device,
                 content_weight=content_weight, style_weight=style_weight, tv_weight=tv_weight,
                 gpu_chunk_size=gpu_chunk_size, mrf_synthesis_stride=mrf_synthesis_stride,
                 mrf_style_stride=mrf_style_stride).to(device)
    mrf.train()
    for i in range(0, num_res):
        if i == 0:
            synthesis = pyramid_content_image[0].clone().requires_grad_(True).to(device)
        else:
            synthesis = unsampleSynthesis(pyramid_content_image[i].shape[2], pyramid_content_image[i].shape[3],
                                          synthesis, device)
            mrf.updateStyleAndContentImage(style_image=pyramid_style_image[i], content_image=pyramid_content_image[i])
        optimizer = optim.LBFGS([synthesis], lr=1, max_iter=max_iter)

        def closure():
            global iterator
            optimizer.zero_grad()
            loss = mrf(synthesis)
            loss.backward(retain_graph=True)
            if (iterator + 1) % 10 == 0:
                print('res-%d-iteration-%d: %f' % (i + 1, iterator + 1, loss.item()))
            if (iterator + 1) % sample_step == 0 or iterator + 1 == max_iter:
                image = getSynthesisImage(synthesis, denorm_transform, device)
                image = F.interpolate(image.unsqueeze(0), size=pyramid_content_image[i].shape[2:4], mode='bilinear')
                torchvision.utils.save_image(image.squeeze(), dir + 'res-%d-result-%d.jpg' % (i + 1, iterator + 1))
                print('save image: res-%d-result-%d.jpg' % (i + 1, iterator + 1))
            iterator += 1
            if iterator == max_iter:
                iterator = 0
            return loss

        optimizer.step(closure)
    image = getSynthesisImage(synthesis, denorm_transform, device)
    image = F.interpolate(image.unsqueeze(0), size=pyramid_content_image[2].shape[2:4], mode='bilinear')
    return image


def texture(cropped, synthesis, dir):
    return train(cropped, synthesis, dir)
