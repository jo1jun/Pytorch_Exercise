# # 뉴럴넷으로 패션 아이템 구분하기
# Fashion MNIST 데이터셋과 앞서 배운 인공신경망을 이용하여 패션아이템을 구분해봅니다.
import torch
import torch.nn as nn
# torch.nn : neural network model 의 재료들을 담고있다.
import torch.optim as optim
# torch.optim : 최적화를 위한 module
import torch.nn.functional as F
# torch.nn.functional : nn module 의 함수 버전
from torchvision import transforms, datasets


USE_CUDA = torch.cuda.is_available()                    # 현재 컴퓨터에서 CUDA 이용 가능 여부
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")    # true : "cuda" / false : "cpu" 를 device 에 설정
                                                        # -> 연산을 하는 곳을 결정 (코드 공유시 유용)

EPOCHS = 30
BATCH_SIZE = 64     # batch size 는 2의 n 승이 편---안

# 데이터셋 불러오기

transform = transforms.Compose([
    transforms.ToTensor()
])


trainset = datasets.FashionMNIST(
    root      = './.data/', 
    train     = True,
    download  = True,
    transform = transform
)

testset = datasets.FashionMNIST(
    root      = './.data/', 
    train     = False,
    download  = True,
    transform = transform
)

train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
)

test_loader = torch.utils.data.DataLoader(
    dataset     = testset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
)


# ## 뉴럴넷으로 Fashion MNIST 학습하기
# 입력 `x` 는 `[배치크기, 색, 높이, 넓이]`로 이루어져 있습니다.
# `x.size()`를 해보면 `[64, 1, 28, 28]`이라고 표시되는 것을 보실 수 있습니다.
# Fashion MNIST에서 이미지의 크기는 28 x 28, 색은 흑백으로 1 가지 입니다.
# 그러므로 입력 x의 총 특성값 갯수는 28 x 28 x 1, 즉 784개 입니다.
# 우리가 사용할 모델은 3개의 레이어를 가진 인공신경망 입니다. 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)      # fully connedted = Linear = affine = 행렬곱
        self.fc2 = nn.Linear(256, 128)      # 가중치가 있는 연산이므로 nn.Layer 로 이용하고
        self.fc3 = nn.Linear(128, 10)       # 생성자에서 선언하는 것이 좋다.

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))             # activation function 은 가중치를 지니지 않는다.
        x = F.relu(self.fc2(x))             # 따라서 functional 에 있는 함수를 직접 사용해도 됨.
        x = self.fc3(x)
        return x


# ## 모델 준비하기
# `to()` 함수는 모델의 파라미터들을 지정한 곳으로 보내는 역할을 합니다.
# 일반적으로 CPU 1개만 사용할 경우 필요는 없지만,
# GPU를 사용하고자 하는 경우 `to("cuda")`로 지정하여 GPU로 보내야 합니다.
# 지정하지 않을 경우 계속 CPU에 남아 있게 되며 빠른 훈련의 이점을 누리실 수 없습니다.
# 최적화 알고리즘으로 파이토치에 내장되어 있는 `optim.SGD`를 사용하겠습니다.

model        = Net().to(DEVICE)
optimizer    = optim.SGD(model.parameters(), lr=0.01)


# 학습하기

def train(model, train_loader, optimizer): 
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # 전체 train set 을 batch 로 나눠서 모든 batch 마다 forward,backward,갱신
        # model 를 DEVICE의 메모리로 보냈으므로 학습 데이터 또한 DEVICE의 메모리로 보냄
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)                        # forward 자동 호출
        loss = F.cross_entropy(output, target)      # output layer 가 10개(class 가 10개)이므로 BCELoss 말고 cross_entropy 사용.
        loss.backward()                             # backward
        optimizer.step()                            # update


# 테스트하기

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():                           # 평가할 때는 gradient 계산 필요X. with 문으로 기울기 계산 꺼주고 평가 후 초기화.
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)                    # forward
            
            # 모든 오차 더하기
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()    # reduction='sum' 으로 미니배치 평균 대신 합으로 
                                                                    # 전부 합산 후 dataset 의 전체 크기로 나눠줄 것.
            # 가장 큰 값을 가진 클래스가 모델의 예측입니다.
            # 예측과 정답을 비교하여 일치할 경우 correct에 1을 더합니다.
            pred = output.max(1, keepdim=True)[1]                   # numpy 와 사용법 유사. 행단위, keepdim 으로  유지
                                                                    # 두개의 원소 반환. [0] : max 원소 [1], : 해당 index
                                                                    
            correct += pred.eq(target.view_as(pred)).sum().item()   # A.eq(B) : A 와 B 가 일치하면 1 아니면 0
                                                                    # A.view_as(B) : A 를 B shape 대로 정렬

    test_loss /= len(test_loader.dataset)                           # len(test_loader.dataset) = 10000
    test_accuracy = 100. * correct / len(test_loader.dataset)       # 숫자. : float type
    return test_loss, test_accuracy


# 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 훈련이 되는지 확인해봅시다!

for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)
    
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
          epoch, test_loss, test_accuracy))

