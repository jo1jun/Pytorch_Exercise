# # cGAN으로 생성 제어하기

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np


# 하이퍼파라미터
EPOCHS = 300
BATCH_SIZE = 100
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device:", DEVICE)


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


# 생성자 (Generator)
class Generator(nn.Module):                     # label정보를 붙여서 생성물과 label 관계성을 학습하여 생성하고싶은 대로 생성.
    def __init__(self):
        super().__init__()
        
        self.embed = nn.Embedding(10, 10)       # [batch, 1] 크기의 레이블 텐서를 [batch, 10] 의 연속적인 텐서로 전환
                                                # RNN 때 보았던 layer로, 10 개의 label이 있고, 각 label 을 10차원 벡터로 임베딩.
                                                # 정수값을 10차원 배열로 매핑하는 이유는 연속적인 값이 학습에 더 유용하기 때문.
        self.model = nn.Sequential(
            nn.Linear(110, 256),                # 무작위 텐서의 크기가 100 이고, 나머지 10이 레이블에 관한 정보이다.
            nn.LeakyReLU(0.2, inplace=True),    # 활성화 함수 살짝 바꿈. / inplace=True : 입력을 복사하지 않고 바로 조작한다.
            nn.Linear(256, 512),                
            nn.LeakyReLU(0.2, inplace=True),    # 앞선 모델보다 더 복잡한 모델이므로 한층을 더 늘리고 neuron 수도 늘림.
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        c = self.embed(labels)                  # embedding 으로 [batch size, 1] -> [batch size, 10]
        x = torch.cat([z, c], 1)                # concat 으로 [batch size, 10] -> [batch size, 110]
        return self.model(x)                    # 나머지 layer 통과.


# 판별자 (Discriminator)
class Discriminator(nn.Module):                 # 판별자에도 label 정보를 똑같이 이어붙여준다.
    def __init__(self):
        super().__init__()
        
        self.embed = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        c = self.embed(labels)                  # embedding 으로 [batch size, 1] -> [batch size, 10]
        x = torch.cat([x, c], 1)                # concat 으로 [batch size, 10] -> [batch size, 794]
        return self.model(x)                    # 나머지 layer 통과.


# 모델 인스턴스를 만들고 모델의 가중치를 지정한 장치로 보내기
D = Discriminator().to(DEVICE)
G = Generator().to(DEVICE)

# 이진 교차 엔트로피 함수와
# 생성자와 판별자를 최적화할 Adam 모듈
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr =0.0002)
g_optimizer = optim.Adam(G.parameters(), lr =0.0002)


total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE)
        
        # '진짜'와 '가짜' 레이블 생성
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

        # 판별자가 진짜 이미지를 진짜로 인식하는 오차 계산 (데이터셋 레이블 입력)
        labels = labels.to(DEVICE)
        outputs = D(images, labels)
        d_loss_real = criterion(outputs, real_labels)   # 진짜(outputs)를 진짜(real_labels)로 보았나?
        real_score = outputs
    
        # 무작위 텐서와 무작위 레이블을 생성자에 입력해 가짜 이미지 생성
        z = torch.randn(BATCH_SIZE, 100).to(DEVICE)
        g_label = torch.randint(0, 10, (BATCH_SIZE,)).to(DEVICE)
        fake_images = G(z, g_label) 
        
        # 판별자가 가짜 이미지를 가짜로 인식하는 오차 계산 (생성자가 이미지 생성할 때 쓴 레이블 입력)
        outputs = D(fake_images, g_label)
        d_loss_fake = criterion(outputs, fake_labels)   # 가짜(outputs)를 가짜(fake_labels)로 보았나?
        fake_score = outputs
        
        # 진짜와 가짜 이미지를 갖고 낸 오차를 더해서 판별자의 오차 계산
        d_loss = d_loss_real + d_loss_fake
        
        # 역전파 알고리즘으로 판별자 모델의 학습을 진행
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # 생성자가 판별자를 속였는지에 대한 오차 계산 (앞에서 만들었던 fake_image가 판별자를 속였는 지에 대한 오차 계산.)
        fake_images = G(z, g_label)                 # 앞에서 생성했던 이미지 다시 생성.
        outputs = D(fake_images, g_label)
        g_loss = criterion(outputs, real_labels)    # 가짜(outputs)를 진짜(fake_labels)로 보았나?
        
        # 역전파 알고리즘으로 생성자 모델의 학습을 진행
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
    print('이폭 [{}/{}] d_loss:{:.4f} g_loss: {:.4f} D(x):{:.2f} D(G(z)):{:.2f}'
          .format(epoch,
                  EPOCHS,
                  d_loss.item(),
                  g_loss.item(),
                  real_score.mean().item(),
                  fake_score.mean().item()))


# 만들고 싶은 아이템 생성하고 시각화하기
item_number = 9 # 만들고 싶은 아이템 번호
z = torch.randn(1, 100).to(DEVICE) # 하나만 출력하기 위해 배치 크기를 1로 하는 정규분포 무작위 텐서 생성. 
g_label = torch.full((1,), item_number, dtype=torch.long).to(DEVICE)    # full(텐서 크기, 원소들 초기값) -> tensor 생성.
sample_images = G(z, g_label)      # 무작위 텐서와 레이블 텐서를 생성자에 입력하여 만들고 싶은 아이템을 생성.
# 무작위 텐서로 인해 레이블은 같지만 약간씩 다른 스타일의 아이템들이 생성된다. -> 새로운 패션 아이템 생성.
sample_images_img = np.reshape(sample_images.data.cpu().numpy()
                               [0],(28, 28))
plt.imshow(sample_images_img, cmap = 'gray')
plt.show()

