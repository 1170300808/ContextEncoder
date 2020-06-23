import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import random
import os
from Net import CoderGan
from Net import Discriminator
import torchvision.utils

import argparse
import myCifar

# overlap_pred = 4
# device = torch.device("cuda:0")
# n_epoches = 25
# size = 128
# batch_size = 64
# beta_1 = 0.5
# lr = 0.001
# wtl2 = 0.998
overlapL2Weight = 10

# argparse
parser = argparse.ArgumentParser(description="set args")
parser.add_argument('-o', '--overlap', default=4, type=int)
parser.add_argument('-d', '--device', default='GPU', type=str, choices=['CPU', 'GPU'])
parser.add_argument('-e', '--epoches', default=25, type=int)
parser.add_argument('-s', '--size', default=128, type=int)
parser.add_argument('-bs', '--batchsize', default=64, type=int)
parser.add_argument('-b', '--betas', default=0.5, type=float)
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-w', '--wtl2', default=0.998, type=float)
parser.add_argument('-GAN', default='model/GAN_cifar10.pth')
parser.add_argument('-Dis', default='model/Discriminator_cifar10.pth')

args = parser.parse_args()

overlap_pred = args.overlap
if args.device == 'GPU':
    device = torch.device("cuda:0")
elif args.device == 'CPU':
    device = torch.device("cpu")
n_epoches = args.epoches
size = args.size
batch_size = args.batchsize
beta_1 = args.betas
lr = args.lr
wtl2 = args.wtl2
GAN_path = args.GAN
Dis_path = args.Dis

if os.path.exists("result/train/cropped"):
    print("文件路径已存在")
    pass
else:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")

seed = random.randint(1, 10000)
print(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# dataset = torchvision.datasets.CIFAR10(root="data//", download=True,
#                                        transform=transforms.Compose([
#                                            transforms.Scale(size),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                        ])
#                                        )

dataset = myCifar.MYCIFAR10(root="data//", transform=transforms.Compose([
    transforms.Scale(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch = 0

gan = CoderGan().to(device)
gan.apply(weights_init)
if GAN_path != '' and os.path.exists(GAN_path):
    gan.load_state_dict(torch.load(GAN_path, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(GAN_path)['epoch']

dis = Discriminator().to(device)
dis.apply(weights_init)
if Dis_path != '' and os.path.exists(Dis_path):
    dis.load_state_dict(torch.load(Dis_path, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(Dis_path)['epoch']

criterion = nn.BCELoss().to(device)
criterionMSE = nn.MSELoss().to(device)

input_real = torch.FloatTensor(batch_size, 3, size, size).to(device)
input_cropped = torch.FloatTensor(batch_size, 3, size, size).to(device)
label = torch.FloatTensor(batch_size).to(device)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(batch_size, 3, size // 2, size // 2).to(device)

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

optimizer_d = optim.Adam(dis.parameters(), lr=lr, betas=(beta_1, 0.999))
optimizer_g = optim.Adam(gan.parameters(), lr=lr, betas=(beta_1, 0.999))

for epoch in range(resume_epoch, n_epoches):
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        # 截取区域
        # real
        real_center_cpu = real_cpu[:, :, int(size / 4):int(size / 4) + int(size / 2),
                          int(size / 4):int(size / 4) + int(size / 2)]
        batch_size = real_cpu.size(0)
        input_real.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        # 添加mask并且标准化
        input_cropped.data[:, 0,
        int(size / 4 + overlap_pred):int(size / 4 + size / 2 - overlap_pred),
        int(size / 4 + overlap_pred):int(
            size / 4 + size / 2 - overlap_pred)] = 2 * 117.0 / 255.0 - 1.0
        input_cropped.data[:, 1,
        int(size / 4 + overlap_pred):int(size / 4 + size / 2 - overlap_pred),
        int(size / 4 + overlap_pred):int(
            size / 4 + size / 2 - overlap_pred)] = 2 * 104.0 / 255.0 - 1.0
        input_cropped.data[:, 2,
        int(size / 4 + overlap_pred):int(size / 4 + size / 2 - overlap_pred),
        int(size / 4 + overlap_pred):int(
            size / 4 + size / 2 - overlap_pred)] = 2 * 123.0 / 255.0 - 1.0

        # 用real数据训练
        dis.zero_grad()
        label.resize_(batch_size).fill_(real_label)

        output = dis(real_center)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # 训练fake
        fake = gan(input_cropped)
        label.fill_(fake_label)
        output = dis(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizer_d.step()

        gan.zero_grad()
        label.fill_(real_label)
        output = dis(fake)
        errG_D = criterion(output, label)

        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(wtl2 * overlapL2Weight)
        wtl2Matrix.data[:, :, int(overlap_pred):int(size / 2 - overlap_pred),
        int(overlap_pred):int(size / 2 - overlap_pred)] = wtl2

        errG_l2 = (fake - real_center).pow(2)
        errG_l2 *= wtl2Matrix
        errG_l2 = errG_l2.mean()

        errG = (1 - wtl2) * errG_D + wtl2 * errG_l2

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizer_g.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch + 1, n_epoches, i, len(dataloader),
                 errD.item(), errG_D.item(), errG_l2.item(), D_x, D_G_z1,))
        if i % 100 == 0:
            torchvision.utils.save_image(real_cpu,
                                         'result/train/real/real_samples_epoch_%03d.png' % epoch)
            torchvision.utils.save_image(input_cropped.data,
                                         'result/train/cropped/cropped_samples_epoch_%03d.png' % epoch)
            recon_image = input_cropped.clone()
            recon_image.data[:, :, int(size / 4):int(size / 4 + size / 2),
            int(size / 4):int(size / 4 + size / 2)] = fake.data
            torchvision.utils.save_image(recon_image.data,
                                         'result/train/recon/recon_center_samples_epoch_%03d.png' % epoch)

        if i % 10 == 0:
            # checkpoint
            torch.save({'epoch': epoch,
                        'state_dict': gan.state_dict()},
                       'model/GAN_cifar10.pth')
            torch.save({'epoch': epoch,
                        'state_dict': dis.state_dict()},
                       'model/Discriminator_cifar10.pth')
            print("第{}已保存".format(i))
