import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils
import os
from PIL import Image
from torch.autograd import Variable
from TextureTrain import texture
from Net import ContentNet
from Net import CoderGan

overlap_pred = 4
device = torch.device("cuda:0")
n_epoches = 25
image_size = 128
image_size_raw = 512
batch_size = 1
beta_1 = 0.5
lr = 0.0002
wtl2 = 0.998
wtlD = 0.001
niter = 25
overlapL2Weight = 10
dir = 'result/'
content_path = 'test/3.jpg'
transform = transforms.Compose([transforms.Scale(image_size),

                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform1 = transforms.Compose([transforms.Scale(image_size_raw),
                                 transforms.CenterCrop(image_size_raw),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
denorm_transform = transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2))
denorm_transform1 = transforms.Normalize(mean=(-2.12, -2.04, -1.80), std=(4.37, 4.46, 4.44))
transform3 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
pic = content_path
if not os.path.exists(dir):
    os.mkdir(dir)

content_image_ori = cv.imread(pic)
print(pic)
# print(content_image_ori)
content_image_ori = cv.cvtColor(content_image_ori, cv.COLOR_BGR2RGB)
content_image_ori_PIL = Image.fromarray(content_image_ori)
content_image = transform(content_image_ori_PIL).unsqueeze(0)
content_images = content_image
device = torch.device("cuda:0")
content_raw = transform1(content_image_ori_PIL).unsqueeze(0).to(device)

torchvision.utils.save_image(denorm_transform1(content_raw[0]), dir + '/real.jpg')

content_raw[:, :, int(image_size_raw / 4):int(image_size_raw / 4 + image_size_raw / 2),
int(image_size_raw / 4):int(image_size_raw / 4 + image_size_raw / 2)] = 0.0
content_raw[:, :, int(image_size_raw / 4):int(image_size_raw / 4 + image_size_raw / 2),
int(image_size_raw / 4):int(image_size_raw / 4 + image_size_raw / 2)] = torch.mean(content_raw)

torchvision.utils.save_image(denorm_transform1(content_raw[0]), dir + '/cropped.jpg')

photos = ["test/2.jpg", "test/3.jpg"]
for path in photos:
    content_image_ori1 = cv.imread(path)
    content_image_ori1 = cv.cvtColor(content_image_ori1, cv.COLOR_BGR2RGB)
    content_image_ori_PIL1 = Image.fromarray(content_image_ori1)
    content_image1 = transform(content_image_ori_PIL1).unsqueeze(0)
    content_images = torch.cat((content_images, content_image1), 0)

input_cropped = torch.FloatTensor(batch_size, 3, image_size, image_size)

input_cropped = Variable(input_cropped).to(device)

input_cropped.resize_(content_images.size()).copy_(content_images)

input_cropped.data[:, 0,

int(image_size / 4 + overlap_pred):int(image_size / 4 + image_size / 2 - overlap_pred),
int(image_size / 4 + overlap_pred):int(
    image_size / 4 + image_size / 2 - overlap_pred)] = 2 * 117.0 / 255.0 - 1.0
input_cropped.data[:, 1,
int(image_size / 4 + overlap_pred):int(image_size / 4 + image_size / 2 - overlap_pred),
int(image_size / 4 + overlap_pred):int(
    image_size / 4 + image_size / 2 - overlap_pred)] = 2 * 104.0 / 255.0 - 1.0
input_cropped.data[:, 2,
int(image_size / 4 + overlap_pred):int(image_size / 4 + image_size / 2 - overlap_pred),
int(image_size / 4 + overlap_pred):int(
    image_size / 4 + image_size / 2 - overlap_pred)] = 2 * 123.0 / 255.0 - 1.0

content_net = CoderGan().to(device)
content_net.load_state_dict(torch.load("GAN_cifar10.pth", map_location=lambda storage, location: storage)['state_dict'])
resume_epoch = torch.load("GAN_cifar10.pth")['epoch']
print(content_net)
synthesis = content_net(input_cropped)
recon_image = input_cropped.clone()
recon_image.data[:, :, int(image_size / 4):int(image_size / 4 + image_size / 2),
int(image_size / 4):int(image_size / 4 + image_size / 2)] = synthesis.data
torchvision.utils.save_image(denorm_transform(recon_image.data[0]), dir + '/input.jpg')
torchvision.utils.save_image(denorm_transform(synthesis.data[0]), dir + '/output.jpg')
content_result = denorm_transform(synthesis.data[0])
content_result = transform3(content_result)
content_result = content_result.unsqueeze(0)
content_result = F.interpolate(content_result, [int(image_size_raw / 2), int(image_size_raw / 2)],
                               mode="bilinear")
content_result.cuda()
result = texture(content_raw, content_result, dir)
content_raw = denorm_transform1(content_raw.data[0])
for i in range(3):
    content_raw.data[i, int(image_size_raw / 4):int(image_size_raw / 4 + image_size_raw / 2), \
    int(image_size_raw / 4):int(image_size_raw / 4 + image_size_raw / 2)] = result.data[0][i]
torchvision.utils.save_image(content_raw, dir + '/result.jpg')
