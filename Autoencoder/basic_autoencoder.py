# # 오토인코더로 이미지의 특징을 추출하기

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# Axes3D : matplotlib 에서 3차원의 plot 을 그리는 용도.
# cm     : 데이터 포인트에 색상을 입히는 데 사용.


# 하이퍼파라미터
EPOCH = 10
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device:", DEVICE)


# Fashion MNIST 데이터셋
trainset = datasets.FashionMNIST(
    root      = './.data/', 
    train     = True,
    download  = True,
    transform = transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    # num_workers = 2 # GPU 가 없는 내 PC 에서는 이 line 을 없애야 잘 작동한다. Colab 에서는 GPU를 사용하므로 잘 작동.
)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),      # input feature 는 28*28 (784) 차원
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),           # 입력의 특징을 3차원으로 압축합니다 (시각화 하기 위해 3차원으로 압축.)
        )                               # encoder 의 ouput 이 latent variable 이다.
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),      # encoder 의 input feature 수와 동일한 feature 수로 복원한다.
            nn.Sigmoid(),               # encoder 를 뒤집은 것과 같고 마지막에 Sigmoid 로 픽셀당 0~1 사이 값 출력.
        )

    def forward(self, x):
        encoded = self.encoder(x)       # latent variable
        decoded = self.decoder(encoded) # 복원된 이미지.
        return encoded, decoded


autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)    # DL1 에서 공부한 Adam. README 참고.
criterion = nn.MSELoss()                                            # Mean Squared Error Loss


# 원본 이미지를 시각화 하기 (첫번째 열)
view_data = trainset.data[:5].view(-1, 28*28)
view_data = view_data.type(torch.FloatTensor)/255.                  # (5 * 784) 크기의 0 ~ 1 사이값 tensor
                                                                    # 현재 모델은 0 ~ 1 을 인식하므로 RGB값인 255 로 나누어준다.


def train(autoencoder, train_loader):
    autoencoder.train()
    for step, (x, label) in enumerate(train_loader):
        x = x.view(-1, 28*28).to(DEVICE)
        y = x.view(-1, 28*28).to(DEVICE)
        label = label.to(DEVICE)

        encoded, decoded = autoencoder(x)               # forward

        loss = criterion(decoded, y)                    # cacluate loss (복원과 원본 사이의 오차제곱합)
        optimizer.zero_grad()
        loss.backward()                                 # backward
        optimizer.step()                                # update


for epoch in range(1, EPOCH+1):
    train(autoencoder, train_loader)

    # 디코더에서 나온 이미지를 시각화 하기 (두번째 열)
    test_x = view_data.to(DEVICE)                       # sample image 를 한 번 train 한 autoencoder에 넣기 (forward 만 수행)
    _, decoded_data = autoencoder(test_x)               # sample 복원 image 를 get

    # 원본과 디코딩 결과 비교해보기 (시각화)
    f, a = plt.subplots(2, 5, figsize=(5, 2))                               # 2행 5열 액자 구성
    print("[Epoch {}]".format(epoch))
    for i in range(5):
        img = np.reshape(view_data.data.numpy()[i],(28, 28))                # 원본 이미지
        a[0][i].imshow(img, cmap='gray')                                    # 1행에 원본 이미지 넣기
        a[0][i].set_xticks(()); a[0][i].set_yticks(())                      # set_xticks, set_yticks 에 빈 튜플을 넣어 눈금자 표시를 없앤다.

    for i in range(5):  
        img = np.reshape(decoded_data.to("cpu").data.numpy()[i], (28, 28))  # 복원 이미지
        a[1][i].imshow(img, cmap='gray')                                    # 2행에 원본 이미지 넣기
        a[1][i].set_xticks(()); a[1][i].set_yticks(())
    plt.show()


# # 잠재변수 들여다보기

# 잠재변수를 3D 플롯으로 시각화
view_data = trainset.data[:200].view(-1, 28*28)
view_data = view_data.type(torch.FloatTensor)/255.
test_x = view_data.to(DEVICE)
encoded_data, _ = autoencoder(test_x)
encoded_data = encoded_data.to("cpu")


CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig)

X = encoded_data.data[:, 0].numpy()
Y = encoded_data.data[:, 1].numpy()
Z = encoded_data.data[:, 2].numpy()

labels = trainset.targets[:200].numpy()

for x, y, z, s in zip(X, Y, Z, labels):
    name = CLASSES[s]
    color = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, name, backgroundcolor=color)

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()




