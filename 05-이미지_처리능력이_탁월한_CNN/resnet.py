# # 신경망 깊게 쌓아 컬러 데이터셋에 적용하기
# Convolutional Neural Network (CNN) 을 쌓아올려 딥한 러닝을 해봅시다.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


# ## 하이퍼파라미터

EPOCHS     = 300
BATCH_SIZE = 128


# ## 데이터셋 불러오기

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),            # data augmentation function
                       transforms.RandomHorizontalFlip(),               # data augmentation function
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),            # 3 channel : ((평균1, 2, 3), (표편1, 2, 3))
                                            (0.5, 0.5, 0.5))])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data',
                   train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])),
    batch_size=BATCH_SIZE, shuffle=True)


# ## ResNet 모델 만들기

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):                # input channel, output channel, stride
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)                           # batch normalization
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()                             # 입력값과 중간값의 크기가 일치하는 경우 입력값을 그대로 더해준다.
        if stride != 1 or in_planes != planes:                      # stride 가 1이 아니라 크기가 작아지거나 / 채널 수가 달라진 경우
            self.shortcut = nn.Sequential(                          # 입력값과 중간값이 크기가 다르므로 입력값 크기를 중간값과 맞춘다.
                nn.Conv2d(in_planes, planes,                        # Conv&batch 를 거쳐 channel(plane에 의해 변동)수를 바꾸고
                          kernel_size=1, stride=stride, bias=False),# size(stride에 의해 변동)를 바꾸어 중간값에 더해준다.
                nn.BatchNorm2d(planes)                              # padding 사이즈 또한 크기를 변동시키지만 1로 통일했으므로 배제.
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)                 # 이전 입력값 (혹은 channel or size 가 맞지않아 맞춰준 값)을 더해주고
        out = F.relu(out)                       # 활성화 함수를 거친다.
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)     # feature map 크기 변화 x / channel 수 증가
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)             # feature map 크기 변화 x / channel 수 변화 x
        self.layer2 = self._make_layer(32, 2, stride=2)             # feature map 크기 절반   / channel 수 두배
        self.layer3 = self._make_layer(64, 2, stride=2)             # feature map 크기 절반   / channel 수 두배
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)                     # Basic block 마다의 stride list
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))   
            self.in_planes = planes                                     # input plane 수 갱신
        return nn.Sequential(*layers)                                   # layers list 를 unpacking 한 후 가변인자 전달.

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))       # out.shape : [128, 16, 32, 32] (batch, channel, size1, size2)
        out = self.layer1(out)                      # out.shape : [128, 16, 32, 32] (batch, channel, size1, size2)
        out = self.layer2(out)                      # out.shape : [128, 32, 16, 16] (batch, channel, size1, size2)
        out = self.layer3(out)                      # out.shape : [128, 64, 8, 8] (batch, channel, size1, size2)
        out = F.avg_pool2d(out, 8)                  # out.shape : [128, 64, 1, 1] (batch, channel, size1, size2)
        out = out.view(out.size(0), -1)             # out.shape : [128, 64] (batch, input_neuron)
        out = self.linear(out)                      # out.shape : [128, 10] (batch, class_number)
        return out


# ## 준비

model = ResNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)   # 학습 효율을 높이기 위해 학습률 감소 기법 사용
                                                                            # 학습률이 점차 감소하여 갱신값이 0으로 수렴한다.
                                                                            # 따라서 optima 주변을 배회하지 않고 수렴.
                                                                            # scheduler 는 epoch 마다 호출되고 step_size 마다
                                                                            # gamma 가 learning rate 에 곱해진다.

print(model)    # 처음부터 끝까지 모든 계층의 구성을 볼 수 있다.

# 아래는 이전에 보았던 것과 동일하다. (단, 학습을 실행할 때 scheduler.step() 함수로 learning rate 를 조금 낮추는 단계가 추가)
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
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 훈련이 되는지 확인해봅시다!

for epoch in range(1, EPOCHS + 1):
    scheduler.step() # 매 epoch 마다 호출된다. learning rate 를 조금 낮추는 단계 (50번 호출 마다)
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)
    
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
          epoch, test_loss, test_accuracy))




