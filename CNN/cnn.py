# # CNN으로 패션 아이템 구분하기
# Convolutional Neural Network (CNN) 을 이용하여 패션아이템 구분 성능을 높여보겠습니다.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


EPOCHS     = 40
BATCH_SIZE = 64


# ## 데이터셋 불러오기

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./.data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./.data',
                   train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)


# ## 뉴럴넷으로 Fashion MNIST 학습하기

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)                # Conv layer (input 채널 수가 1, output(feature map) 채널 수가 10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)               # nn.Conv2d(입력 채널 수, 출력 채널 수, filter 크기)
        self.conv2_drop = nn.Dropout2d()                            # dropout
        self.fc1 = nn.Linear(320, 50)                               # FC layer
        self.fc2 = nn.Linear(50, 10)
                                                                    # filter 크기가 정수면 정사각형, 배열로 직사각형 지정 가능.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                  # max polling(input data, filter 크기)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # output size : [64, 20, 4, 4] (batch 크기, channel 크기, image shape)
        x = x.view(-1, 320)                                         # FC layer 준비. input size : 320 (20 * 4 * 4)
        x = F.relu(self.fc1(x))                                     # FD layer 는 Conv layer 와 달리 1차원 data 를 batch 처리한다.
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# ## 하이퍼파라미터 

model     = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 아래는 이전에 보았던 것과 동일하다.
# ## 학습하기

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# ## 테스트하기

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# ## 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 학습이 되는지 확인해봅시다!

for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)
    
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
          epoch, test_loss, test_accuracy))

