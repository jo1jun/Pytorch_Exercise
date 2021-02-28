# # GAN으로 새로운 패션아이템 생성하기
# *GAN을 이용하여 새로운 패션 아이템을 만들어봅니다*
# 이 프로젝트는 최윤제님의 파이토치 튜토리얼 사용 허락을 받아 참고했습니다.
# * [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) - MIT License

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np


# EPOCHS 과 BATCH_SIZE 등 학습에 필요한 하이퍼 파라미터 들을 설정해 줍니다.

# 하이퍼파라미터
EPOCHS = 500
BATCH_SIZE = 100
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device:", DEVICE)


# 학습에 필요한 데이터셋을 로딩합니다. 

# Fashion MNIST 데이터셋
trainset = datasets.FashionMNIST(
    './.data',
    train=True,
    download=True,
    transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
    ])
)
train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True
)


# 생성자는 64차원의 랜덤한 텐서를 입력받아 이에 행렬곱(Linear)과 활성화 함수(ReLU, Tanh) 연산을 실행합니다. 
# 생성자의 결과값은 784차원, 즉 Fashion MNIST 속의 이미지와 같은 차원의 텐서입니다.

# 생성자 (Generator)
G = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Tanh())              # 값을 -1 ~ 1 사이로 압축


# 판별자는 784차원의 텐서를 입력받습니다. 판별자 역시 입력된 데이터에 행렬곱과 활성화 함수를 실행시키지만, 
# 생성자와 달리 판별자의 결과값은 입력받은 텐서가 진짜인지 구분하는 예측값입니다.

# 판별자 (Discriminator)
D = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),      # ReLU 와 다르게 약간의 음의 기울기도 생성자에 전달하여 생성자를 효과적으로 학습시킨다.
        nn.Linear(256, 1),
        nn.Sigmoid())


# 생성자와 판별자 학습에 쓰일 오차 함수와 최적화 알고리즘도 정의해 줍니다.

# 모델의 가중치를 지정한 장치로 보내기
D = D.to(DEVICE)
G = G.to(DEVICE)

# 이진 크로스 엔트로피 (Binary cross entropy) 오차 함수와
# 생성자와 판별자를 최적화할 Adam 모듈
criterion = nn.BCELoss()                                # 레이블이 진짜, 가짜 두가지므로 BCELoss() 사용.
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)


# 모델 학습에 필요한 준비는 끝났습니다. 그럼 본격적으로 GAN을 학습시키는 loop을 만들어 보겠습니다. 

total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(train_loader):
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE)
        
        # '진짜'와 '가짜' 레이블 생성
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)      # real label = 1
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)     # fake label = 0
        
        # 판별자가 진짜 이미지를 진짜로 인식하는 오차를 계산 (판별자는 진짜를 진짜로 판단해야하므로 이 오차를 줄여야한다.)
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # 무작위 텐서로 가짜 이미지 생성
        z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
        fake_images = G(z)
        
        # 판별자가 가짜 이미지를 가짜로 인식하는 오차를 계산 (판별자는 가짜를 가짜로 판단해야하므로 이 오차를 줄여아한다.)
        outputs = D(fake_images)                                # 생성자가 만든 가짜이미지를 봐야 오차를 계산함. (G 거친다.)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # 진짜와 가짜 이미지를 갖고 낸 오차를 더해서 판별자의 오차 계산 
        d_loss = d_loss_real + d_loss_fake                      # 판별자는 두 오차를 더한 전체 오차를 줄여야 한다.

        # 역전파 알고리즘으로 판별자 모델의 학습을 진행
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()                                 # G를 거친 loss 도 있으므로 G의 gradient도 초기화 해주어야 한다.
        d_loss.backward()                                       
        d_optimizer.step()
        
        # 생성자가 판별자를 속였는지에 대한 오차를 계산
        fake_images = G(z)
        outputs = D(fake_images)                                # 판별자가 판별한 결과를 봐야 오차를 계산함. (D 거친다.)
        g_loss = criterion(outputs, real_labels)                # 생성자는 가짜를 진짜로 판단하게 해야하므로 이 오차를 줄여야 한다.
        
        # 역전파 알고리즘으로 생성자 모델의 학습을 진행
        d_optimizer.zero_grad()                                 # D도 거쳤으므로 D의 gradient 도 초기화 해주어야 한다.
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
    # 학습 진행 알아보기
    print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
          .format(epoch, EPOCHS, d_loss.item(), g_loss.item(), 
                  real_score.mean().item(), fake_score.mean().item()))


# 학습이 끝난 생성자의 결과물을 한번 확인해 보겠습니다.

z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
fake_images = G(z)  # 이미지를 무작위로 생성.
for i in range(10):
    fake_images_img = np.reshape(fake_images.data.cpu().numpy()[i],(28, 28))
    plt.imshow(fake_images_img, cmap = 'gray')
    plt.show()

