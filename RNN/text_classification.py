# # 프로젝트 1. 영화 리뷰 감정 분석
# **RNN 을 이용해 IMDB 데이터를 가지고 텍스트 감정분석을 해 봅시다.**
# 이번 책에서 처음으로 접하는 텍스트 형태의 데이터셋인 IMDB 데이터셋은 50,000건의 영화 리뷰로 이루어져 있습니다.
# 각 리뷰는 다수의 영어 문장들로 이루어져 있으며, 
# 평점이 7점 이상의 긍정적인 영화 리뷰는 2로, 평점이 4점 이하인 부정적인 영화 리뷰는 1로 레이블링 되어 있습니다. 
# 영화 리뷰 텍스트를 RNN 에 입력시켜 영화평의 전체 내용을 압축하고, 
# 이렇게 압축된 리뷰가 긍정적인지 부정적인지 판단해주는 간단한 분류 모델을 만드는 것이 이번 프로젝트의 목표입니다.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets


# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", DEVICE)


# 데이터 로딩하기
print("데이터 로딩중...")
# 텍스트 형태의 영화 리뷰들과 그에 해당하는 레이블을 텐서로 바꿔줄 때 필요한 설정.
# sequential : 순차적 데이터셋 명시 (label 은 순차적 데이터 X)
# batch_first : 신경망에 입력되는 텐서의 첫 번째 차원값이 batch_size 가 되도록 설정.
# lower : 모든 알파벳 소문자 설정.
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

# datasets 객체의 splits() 함수에 앞에서 만든 설정들을 넘겨주어 모델에 입력되는 데이터셋 만들어줌.
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

# 워드 임베딩에 필요한 단어 사전 만들어줌.
# min_feq = 5 : 학습 데이터에서 최소 5번 이상 등장한 단어만을 사전에 담음. / 5번 미만 출현 단어는 'unk'토큰(unknown)으로 대체.
TEXT.build_vocab(trainset, min_freq=5)
LABEL.build_vocab(trainset)

# 학습용 데이터를 학습셋 80% 검증셋 20% 로 나누기
trainset, valset = trainset.split(split_ratio=0.8)

# 반복할 때마다 배치를 생성해주는 반복자 생성.
# 이 반복자를 enumerate() 함수에 입력시켜 루프를 구현하면 루프 때마다 전체 dataset 에서 배치 단위의 데이터가 생성된다.
# 각각의 data들의 token 수는 개수가 다르지만 batch 로 묶어 batch 마다 동일한 개수가 된다. (출력해보니 숫자로 표현되어있고 늘어난 경우는 1 로 채워진 것으로 '추정'된다.) 
train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)

vocab_size = len(TEXT.vocab)    # TEXT.vocab : 전체 data set 에서 추출한 단어사전
n_classes = 2

print(vars(trainset[0]))    # vars() 를 통해 trainset 의 element(dictionary)를 볼 수 있다.
print(vars(trainset[1]))

print("[학습셋]: %d [검증셋]: %d [테스트셋]: %d [단어수]: %d [클래스] %d"
      % (len(trainset),len(valset), len(testset), vocab_size, n_classes))


class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicGRU, self).__init__()
        print("Building Basic GRU model...")
        self.n_layers = n_layers                                                # 은닉 벡터의 층. (아주 복잡한 모델이 아닌 경우 보통 2 이하.)
        self.embed = nn.Embedding(n_vocab, embed_dim)                           # 워드 임베딩 함수.
                                                                                # n_vocab : 사전의 단어 수 / embed_dim : 임베딩된 단어 텐서의 차원
        self.hidden_dim = hidden_dim                                            # 은닉 벡터의 차원
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,                           # RNN 대신 RNN 의 단점을 보완한 GRU 사용.
                          num_layers=self.n_layers,                             
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)                        # 마지막은 역시 CNN 처럼 Linear 로 클래스에 대한 예측을 출력

    def forward(self, x):
        x = self.embed(x)                                                       # x.shape : [64, ~, 128] (batch 크기, token 수, embed_dim) (token 수 = batch 로 묶인 입력 x 들의 길이)
        h_0 = self._init_state(batch_size=x.size(0))                            # 첫번째 은닉벡터 생성 / h_o.shape : [1, 64, 256] (은닉 벡터 층, batch 크기, hidden_dim)        
        
        x, _ = self.gru(x, h_0)                                                 # 은닉 벡터들이 시계열 배열 형태로 반환된다. / x.shape : [64, ~, 256] (batch 크기, token 수, hidden_dim)
        h_t = x[:,-1,:]                                                         # 마지막 토큰에 해당하는 은닉벡터 추출. 즉, 리뷰를 전부 압축한 은닉벡터 / h_t.shape : [64, 256]
        
        # x, h_t = self.gru(x, h_0)                                             # 바로 위 두 줄을 이 두 줄로 대체 가능하다.
        # h_t = h_t.squeeze()                                                   # 두 번째 반환값은 마지막 압축 벡터(n_layers 가 추가됨) 이다. [1, 64, 256] -> squeeze -> [64, 256]
        
        self.dropout(h_t)                                                       # drop out
        logit = self.out(h_t)                                                   # [64, 256] -> FC layer -> [64, 2]
        return logit
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data                                   # parameters() : 가중치 정보들을 반복자 형태로 반환. 원소 type : tensor
                                                                                # next() 로 반복자 원소들을 순서대로 꺼낸다. (.data 굳이 안 써도 됨. tensor 로 동일.)
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()   # new()로 가중치와 같은 data type을 갖는 tensor를 원하는 shape으로 생성, 0으로 초기화


def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):                                      # index 와 batch 단위의 data를 반환. (batch.text , batch.label 로 data 에 접근할 수 있다.)
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)                    # x.shape = (batch 크기, token 수)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환                               # text 수는 원래 제각각 인데 배치로 묶어서 배치마다 token 수가 동일해짐
        optimizer.zero_grad()
        
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()


def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:                                                      # index 가 필요 없다면 enumerate 말고 그냥 iterator 써도 됨. 위의 train module 에서도 생략해도 된다.
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')                       # 한 배치 내에서의 평균이 아니라 전부 합.
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()       # max 로 행 기준(1) & index[1] / 예측 값과 실제 값이 일치하는 경우를 count.
    size = len(val_iter.dataset)                                                
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[이폭: %d] 검증 오차:%5.2f | 검증 정확도:%5.2f" % (e, val_loss, val_accuracy))
    
    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss


model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))

