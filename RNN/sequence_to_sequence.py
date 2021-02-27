# # Seq2Seq 기계 번역
# 이번 프로젝트에선 임의로 Seq2Seq 모델을 아주 간단화 시켰습니다.
# 한 언어로 된 문장을 다른 언어로 된 문장으로 번역하는 덩치가 큰 모델이 아닌
# 영어 알파벳 문자열("hello")을 스페인어 알파벳 문자열("hola")로 번역하는 Mini Seq2Seq 모델을 같이 구현해 보겠습니다.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


vocab_size = 256                # 간단한 모델이므로 워드 임베딩이 아닌 문자 임베딩이므로 총 아스키 코드 개수를 vocab_size 로 한다.
x_ = list(map(ord, "hello"))    # 아스키 코드 리스트로 변환 / map(function, iterable) -> iterator : function(iterable)
y_ = list(map(ord, "hola"))     # ord(character) -> ASCII code / list(iterator) -> array
print("hello -> ", x_)
print("hola  -> ", y_)


x = torch.LongTensor(x_)
y = torch.LongTensor(y_)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.n_layers = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # embed_dim 을 따로 정의하지 않고 hidden_size 를 사용.
                                                                # 실전에선 문자 체계가 완전히 다를 수 있기 때문에 원문용과 번역문용 임베딩을 따로 만들어야함.
                                                                # 이 예제에서는 영어, 스페인어 모두 아스키 코드로 나타내기 때문에 임베딩 하나여도 무방하다.
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.project = nn.Linear(hidden_size, vocab_size)       # 디코더가 번역문의 다음 토큰을 예상해내는 신경망 추가.

    def forward(self, inputs, targets):
        # 인코더에 들어갈 입력
        initial_state = self._init_state()                      # 첫번째 은닉벡터 생성 / inital_state.shape : [1, 1, 16]
        embedding = self.embedding(inputs).unsqueeze(1)         # charater embedding / embedding.shape : [5] (origin) -> [5, 16] (embeded) -> [5, 1, 16]
                                                                # embedding.shape = [seq_len, batch_size, embedding_size]
        
        # 인코더 (Encoder)
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        # encoder_output = [seq_len, batch_size, hidden_size]   # 은닉 벡터들의 배열 (앞에서 마지막 은닉벡터를 추출해서 사용했었다.)
        # encoder_state  = [n_layers, seq_len, hidden_size]     # 문맥 벡터 (앞에서 추출한 은닉벡터에 n_layers 가 추가된 형태이다.)

        # 디코더에 들어갈 입력
        decoder_state = encoder_state                           # encoder 에서  나온 문맥 벡터를 decoder 의 첫 은닉 벡터로 지정.
        decoder_input = torch.LongTensor([0])                   # 문장 시작 토큰으로 decoder 가 정상적으로 작동할 수 있도록 인위적으로 넣은 토큰.
        
        # 디코더 (Decoder)
        outputs = []
        
        for i in range(targets.size()[0]):  # 문장 시작 토큰으로 첫 토큰을 예측하고, 예측한 토큰으로 그 다음 토큰을 예측한다. 정답(target) 길이만큼 반복.
            decoder_input = self.embedding(decoder_input).unsqueeze(1)                  # charater embedding / shape : [1] -> [1, 16] -> [1, 1, 16]
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)  # decoder 의 출력값이 다시 decoder 로 들어간다.
            projection = self.project(decoder_output)                                   # character 예측
            outputs.append(projection)                                                  # 예측한 character 저장
            
            #티처 포싱(Teacher Forcing) 사용                   # 풍부한 data 로 학습하는 seq2seq 모델은 decoder 가 예측한 토큰을 다시 입력해 주는게 정석.
            decoder_input = torch.LongTensor([targets[i]])    # 이 예제에서는 데이터가 적고 잘못된 예측 토큰을 입력으로 사용할 확률이 높아 학습이 더디다.
                                                              # 실제 번역문의 토큰을 대신 입력으로 사용하여 학습을 가속시켜 해결. (Teacher Forcing)
        outputs = torch.stack(outputs).squeeze()              # tensor list를 합치고 squeeze [4, 1, 1, 256] -> [4, 256]
        return outputs
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()


seq2seq = Seq2Seq(vocab_size, 16)


criterion = nn.CrossEntropyLoss()                                   # 아래에서 loss 구할 때 F.cross_entropy(prediction, y) 로 바로 구할 수도 있다.
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)         

log = []
for i in range(1000):                                               # 1000번의 epoch 로 학습.
    prediction = seq2seq(x, y)                                      # forward
    loss = criterion(prediction, y)                                 # loss
    optimizer.zero_grad()
    loss.backward()                                                 # backward
    optimizer.step()                                                # update
    loss_val = loss.data
    log.append(loss_val)                                            # loss 기록
    if i % 100 == 0:
        print("\n 반복:%d 오차: %s" % (i, loss_val.item()))
        _, top1 = prediction.data.topk(1, 1)                        # torch.topk(input tensor, k, dim, ...) -> value, index
        print([chr(c) for c in top1.squeeze().numpy().tolist()])    # tensor 에서 top k 개의 원소들을 추출한다. (dim = 1 -> 행 고정)


plt.plot(log)
plt.ylabel('cross entropy loss')
plt.show()

