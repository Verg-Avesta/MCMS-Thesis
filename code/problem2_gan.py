import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import cv2

torch.manual_seed(1)

# Hyper Parameters
EPOCH = 1000
BATCH_SIZE = 32
LR_D = 0.0001
LR_G = 0.001
IDEA_NUM = 96
DOWNLOAD_MNIST = False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(IDEA_NUM, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 64),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 64)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 4, 2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 64, 7, 7)  # reshape 通道是 128，大小是 7x7
        x = self.conv(x)
        return x


class Discrim(nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


D = Discrim().cuda()
G = Generator().cuda()

# Load the pass way images.
img_data = np.load(open('./data/problem2_shot_image.npy', 'rb'))
img_value = np.load(open('./data/problem2_shot_value.npy', 'rb'))

ok_data = []

for i in range(img_data.shape[0]):
    if img_value[i] > 1.5:
        ok_data.append(img_data[i])
print(len(ok_data))
img_torch = torch.from_numpy(np.array(ok_data)).unsqueeze(
    1).type(torch.FloatTensor)
img_torch = img_torch / 255.0


train_loader = Data.DataLoader(
    Data.dataset.TensorDataset(img_torch), BATCH_SIZE, True)

try:
    D.load_state_dict(torch.load('./data/GAND.pkl'))
    G.load_state_dict(torch.load('./data/GANG.pkl'))
    print('Previous model loaded.')
except:
    print('Previous model not found.')

optimizer_G = torch.optim.Adam(G.parameters(), LR_G)
optimizer_D = torch.optim.Adam(D.parameters(), LR_D)

loss_func = nn.BCEWithLogitsLoss()

for epoch in range(EPOCH):
    G.train()
    for step, (batchx,) in enumerate(train_loader):
        # 创造灵感
        IDEA = (torch.rand(batchx.size(0), IDEA_NUM).cuda()-0.5)/0.5

        # 开始作画
        shit_paintings = G(IDEA)

        # 开始评测
        score_good = D(batchx.cuda())
        score_shit = D(shit_paintings)

        # 计算损失函数
        good_label = torch.ones(batchx.size(0), 1).cuda()
        shit_label = torch.zeros(batchx.size(0), 1).cuda()
        loss_D = loss_func(score_good, good_label) + \
            loss_func(score_shit, shit_label)
        loss_G = loss_func(score_shit, good_label)

        # 反向传播
        optimizer_D.zero_grad()
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if step % 20 == 0:
            print('EPOCH:', epoch, "Step:", step, 'LossD:',
                  loss_D.item(), 'LossG:', loss_G.item())

    if epoch % 500 == 0:
        G.eval()
        newIDEA = (torch.rand(BATCH_SIZE, IDEA_NUM).cuda()-0.5)/0.5
        shit_paintings = (G(newIDEA).squeeze()
                          ).data.cpu().view(-1, 28, 28).numpy()
        # 绘制多个图像
        for i in range(1, 17):
            plt.subplot(4, 4, i)
            ret, cur_plt = cv2.threshold(
                shit_paintings[i], 0.5, 255, cv2.THRESH_BINARY)
            plt.imshow(cur_plt, cmap='gray')
        plt.show()
    torch.save(D.state_dict(), './data/GAND.pkl')
    torch.save(G.state_dict(), './data/GANG.pkl')
    print('Model Saved.')
