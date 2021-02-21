from torchvision import datasets, transforms, utils
# torchvision.datasets : torch.utils.data.Dataset 을 상속하는 data set 의 모음. 데이터를 내려받고 이미지 변환설정 적용.
# torchivision.transforms : image data set edit tool 들이 들어있다.
# torchivision.utils : image data 를 save, visualize tool 들이 들어있다.
from torch.utils import data
# torch.utils.data : 데이터셋의 표준을 정의, 데이터셋 load, 편집 도구들이 들어있는 module.
# dataset 의 표준을 torch.utils.data.Dataset 에 정의.
# torch.utils.data.DataLoader 는 학습시 데이터 공급.
# torch.utils.Dataset 파생 클래스들이 입력으로 들어간다.
import matplotlib.pyplot as plt
import numpy as np


# Fashion MNIST 데이터셋

# transforms.Compose 에 입력으로 transforms.Resize() 등의 편집 함수들을 list 로 넣어 순차적으로 데이터를 가공할 수 있다.
transform = transforms.Compose([    
    transforms.ToTensor()           # image 를 tensor 로 바꿔준다.
])

# 여러가지 data 가공 합수들
# ToTensor : 이미지를 파이토치 텐서로 변환
# Resize : 이미지 크기 조정
# Normalize : 평균, 표준편차로 정규화
# RandomHorizontalFlip : 무작위로 이미지를 왼쪽 오른쪽 뒤집기
# RandomCrop : 이미지를 무작위로 자르기


# torchivision.datasets 로 생성된 객체는 torch.utils.data.Dataset 를 상속한다.
# data 를 내려받고 transforms 로 데이터 가공 설정들을 수행한다.
trainset = datasets.FashionMNIST(
    root      = './.data/', 
    train     = True,   # 학습용 data set
    download  = True,   # 현재 root 로 지정한 폴더에 data set 존재 유무 확인 후 없다면 자동 저장됨.
    transform = transform
)

testset = datasets.FashionMNIST(
    root      = './.data/', 
    train     = False,  # 평가용 data set
    download  = True,
    transform = transform
)


batch_size = 16

# torchivision.datasets 가 torchi.utils.data.Dataset 를 상속하므로 torch.utils.data.DataLoader 의 입력으로 들어갈 수 있다.
# torch.utils.data.DataLoader 의 입력으로 torchi.utils.data.Dataset 의 파생클래스들이 들어가기 때문.
# DataLoader 는 데이터를 배치로 나누어 학습시 데이터를 공급한다.
train_loader = data.DataLoader(
    dataset     = trainset,
    batch_size  = batch_size
)

test_loader = data.DataLoader(
    dataset     = testset,
    batch_size  = batch_size
)


dataiter       = iter(train_loader)     # 반복문 안에서 이용할 수 있도록 해준다.
images, labels = next(dataiter)         # 배치 1개를 가져온다.


# 멀리서 살펴보기
img   = utils.make_grid(images, padding=0)  # 여러 이미지를 모아 하나의 이미지로 만든다.
npimg = img.numpy()                         # matplotlib 과 호환이 되도록 numpy 로 변환
plt.figure(figsize=(10, 7))
plt.imshow(np.transpose(npimg, (1,2,0)))    # matplotlib 이 인식하는 차원의 순서로 바꿔주고 image print
plt.show()


print(labels)


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

for label in labels:                        # 가져온 1개의 배치에서 image class 종류들 출
    index = label.item()
    print(CLASSES[index])

# 가까이서 살펴보기
idx = 1

plt.figure()
item_img = images[idx]                      # 가져온 1개의 배치에서 이미지 한개를 indexing
item_npimg = item_img.squeeze().numpy()     #  (28*28*1) -> (28*28) -> numpy 로 변환 for matplotlib
plt.title(CLASSES[labels[idx].item()])
print(item_npimg.shape)
plt.imshow(item_npimg, cmap='gray')
plt.show()