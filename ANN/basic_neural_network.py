import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

n_dim = 2
# make_blobs(샘플수, feature 수, center들의 array, 랜덤 데이터셋, 표본들의 표준편차) -> return data set, label(data가 속한 cluster label)
# reference : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
x_train, y_train = make_blobs(n_samples=80, n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)
x_test, y_test = make_blobs(n_samples=20, n_features=n_dim, centers=[[1,1],[-1,-1],[1,-1],[-1,1]], shuffle=True, cluster_std=0.3)


# 4개의 center 를 2개의 center 로 줄이기.
def label_map(y_, from_, to_):  
    y = numpy.copy(y_)
    for f in from_:
        y[y_ == f] = to_
    return y

y_train = label_map(y_train, [0, 1], 0)
y_train = label_map(y_train, [2, 3], 1)
y_test = label_map(y_test, [0, 1], 0)
y_test = label_map(y_test, [2, 3], 1)


def vis_data(x,y = None, c = 'r'):
    if y is None:                       # y 가 None 인 경우, label 없다는 뜻 -> * 로 mark
        y = [None] * len(x)
    for x_, y_ in zip(x,y):
        if y_ is None:
            plt.plot(x_[0], x_[1], '*',markerfacecolor='none', markeredgecolor=c)
        else:
            plt.plot(x_[0], x_[1], c+'o' if y_ == 0 else c+'+')

plt.figure()
vis_data(x_train, y_train, c='r')   # c 는 plot 할 때 color 지정.
plt.show()

# numpy array 를 tensor 로.
x_train = torch.FloatTensor(x_train)
print(x_train.shape)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)


class NeuralNet(torch.nn.Module):                           # torch.nn.Moudle 상
        def __init__(self, input_size, hidden_size):
            super(NeuralNet, self).__init__()               # 부모 class 생성자 호출 (범용성을 위해 python2.x 버전)
            # super().__init__()                             # python3.x version
            self.input_size = input_size
            self.hidden_size  = hidden_size
            # layer 생성
            self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size)  # Linear 는 affine 혹은 fc layer 을 의미한다.
            self.relu = torch.nn.ReLU()
            self.linear_2 = torch.nn.Linear(self.hidden_size, 1)                # binary classifier 이므로 output size 는 1이다.
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, input_tensor):
            # __init__() 에서 생성한 layer 들을 순차적으로 forward
            linear1 = self.linear_1(input_tensor)
            relu = self.relu(linear1)
            linear2 = self.linear_2(relu)
            output = self.sigmoid(linear2)
            return output


model = NeuralNet(2, 5)
learning_rate = 0.03
criterion = torch.nn.BCELoss()  #이진(B) 교차(Cross) 엔트로피(E) 손실(Loss) 함수.
epochs = 2000

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # model.parameters() 로 model 의 가중치를
                                                                    # optimizer(SGD) 로 넘겨줌.

# train 하지 않고 loss 출력해보기.
model.eval()    # 평가 모드.
test_loss_before = criterion(model(x_test).squeeze(), y_test)   # squeeze 로 (20,1) -> (20,) (y 와 차원 동일시)
                                                                # model(data set) 이 forward 함수 호출을 대신 해준다.
print('Before Training, test loss is {}'.format(test_loss_before.item()))   # .item() : tensor 안 숫자를 scalar로 get


# 오차값이 0.73 이 나왔습니다. 이정도의 오차를 가진 모델은 사실상 분류하는 능력이 없다고 봐도 무방합니다.
# 자, 이제 드디어 인공신경망을 학습시켜 퍼포먼스를 향상시켜 보겠습니다.

for epoch in range(epochs):
    model.train()   # 훈련 모드.
    optimizer.zero_grad()   # epoch 마다 new gradient 계산할 것이므로 gradient 를 0으로 초기화
    train_output = model(x_train)   # forward
    train_loss = criterion(train_output.squeeze(), y_train)
    if epoch % 100 == 0:
        print('Train loss at {} is {}'.format(epoch, train_loss.item()))
    train_loss.backward()   # backward
    optimizer.step()        # update


model.eval()
test_loss = criterion(torch.squeeze(model(x_test)), y_test)
print('After Training, test loss is {}'.format(test_loss.item()))


# 학습을 하기 전과 비교했을때 현저하게 줄어든 오차값을 확인 하실 수 있습니다.
# 지금까지 인공신경망을 구현하고 학습시켜 보았습니다.
# 이제 학습된 모델을 .pt 파일로 저장해 보겠습니다.

torch.save(model.state_dict(), './model.pt')    # 학습한 가중치들을 dictionary 형태로 model.pt 에 save
print('state_dict format of the model: {}'.format(model.state_dict()))


# `save()` 를 실행하고 나면 학습된 신경망의 가중치를 내포하는 model.pt 라는 파일이 생성됩니다. 아래 코드처럼 새로운 신경망 객체에 model.pt 속의 가중치값을 입력시키는 것 또한 가능합니다.

new_model = NeuralNet(2, 5)
new_model.load_state_dict(torch.load('./model.pt'))     # 학습했던 dictionary 형태 가중치들을 new model 에 load
new_model.eval()                                        # 당연히 new model 은 기존 model 과 같은 구조 (NeuralNet(2, 5)) 이어야 한다. 
print('벡터 [-1, 1]이 레이블 1을 가질 확률은 {}'.format(new_model(torch.FloatTensor([-1,1])).item()))   # 이전과 같은 결과를 얻을 수 있다.

