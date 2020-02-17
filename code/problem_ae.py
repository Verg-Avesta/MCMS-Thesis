import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        decoded = self.decoder(x)
        return decoded


# 超参数
EPOCH = 1000
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

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

autoencoder = AutoEncoder().cuda()
optimizer = torch.optim.Adam(autoencoder.parameters(), LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step, (batchx,) in enumerate(train_loader):
        b_x = batchx.view(-1, 28*28).cuda()
        b_x.requires_grad = True
        b_y = batchx.view(-1, 28*28).cuda()

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if step % 10 == 0:
            print(f'EPOCH:{epoch},STEP:{step},LOSS:{loss.item()}')
            if epoch > 100:
                plt.imshow(decoded[0].cpu().view(28, 28).detach().numpy())
                plt.show()
