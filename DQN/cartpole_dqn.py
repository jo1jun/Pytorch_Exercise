# # 카트폴 게임 마스터하기

import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt


# ### 하이퍼파라미터
# 하이퍼파라미터
EPISODES = 50    # 애피소드 반복횟수 (총 플레이할 게임 수)
EPS_START = 0.9  # 학습 시작시 에이전트가 무작위로 행동할 확률 (학습이 진행되면서 조금씩 감소)
EPS_END = 0.05   # 학습 막바지에 에이전트가 무작위로 행동할 확률 (90% 에서 5% 까지 감소)
                 # 무작위로 행동하는 이유는 에이전트가 가능한 모든 행동을 경험하도록 하기 위함.
EPS_DECAY = 200  # 학습 진행시 에이전트가 무작위로 행동할 확률을 감소시키는 값
GAMMA = 0.8      # 할인계수 (현재 보상을 미래 보상보다 얼마나 가치 있게 여기는 지에 대한 값)
LR = 0.001       # 학습률
BATCH_SIZE = 64  # 배치 크기


# ## DQN 에이전트

class DQNAgent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(4, 256),              # input node 는 4개로, 카트의 위치, 속도와 막대기 각도, 속도를 의미한다.
            nn.ReLU(),
            nn.Linear(256, 2)               # output node 는 2개로, 카트를 왼쪽, 오른쪽으로 움직일 때의 가치를 의미한다.
        )
        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.steps_done = 0                 # 학습을 반복할 때마다 증가하는 변수.
        self.memory = deque(maxlen=10000)   # 이전 경험들에 관한 기억을 담는다. / 크기 초과시 오래된 원소부터 삭제 후 추가.
                                            # 오래된 기억을 잊는다고 볼 수 있다.
    def memorize(self, state, action, reward, next_state):
        self.memory.append((state,                              # 현재 상태
                            action,                             # 현재 상태에서 한 행동
                            torch.FloatTensor([reward]),        # 행동에 대한 보상
                            torch.FloatTensor([next_state])))   # 행동으로 인해 새로 생성된 상태
        
    def act(self, state):
        # 학습 초반에는 eps_threshold 값이 높아 최대한 다양한 경험을 하고 점점 낮춰가며 신경망이 결정하는 비율을 높인다.
        # epsilon-greedy algorithm
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            return self.model(state).data.max(1)[1].view(1, 1)  # 신경망이 결정하여 행동
        else:
            return torch.LongTensor([[random.randrange(2)]])    # 무작위로 행동
    
    def learn(self):
        if len(self.memory) < BATCH_SIZE:                       # 저장된 경험들의 수가 배치 크기보다 커질 경우에 학습.
            return
        batch = random.sample(self.memory, BATCH_SIZE)          # 경험들을 무작위로 가져와 경험간의 상관관계 줄인다.
        states, actions, rewards, next_states = zip(*batch)
        states = torch.cat(states)                              # 모두 리스트의 리스트 형태이므로 하나의 텐서로 만든다.
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)


        current_q = self.model(states).gather(1, actions)       # 현재 상태를 신경망에 통과시켜 행동에 대한 가치를 계산.
                                                                # gather: actinos의 열(1)에 있는 원소의 값을 index로 해서 tensor 값 추출.
                                                                # 즉, 형재 상태에서 했던 행동의 가치들을 추출해서 current_q 에 담는다.
        max_next_q = self.model(next_states).detach().max(1)[0] # 다음 상태에서 에이전트가 생각하는 행동의 최대 가치를 구한다.
                                                                # detach 를 없애면 학습이 잘 되지 않는다. 다음 상태가 모델을 통과하는 경우
                                                                # 이를 통해 가중치를 업데이트하면 안 된다. 할인된 미래 가치 계산용도일 뿐.
        
        expected_q = rewards + (GAMMA * max_next_q)             # 현재 상태에서 행동해 받았던 보상과 할인된 미래가치를 더해 e_q 에 담는다.
                                                                # 이를 극대화하는 방향으로 학습.
                                                                
        loss = F.mse_loss(current_q.squeeze(), expected_q)      # 현재 행동의 가치가 할인된 미래 가치를 따라가도록 학습.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ## 학습 준비하기
# `gym`을 이용하여 `CartPole-v0`환경을 준비하고 앞서 만들어둔 DQNAgent를 agent로 인스턴스화 합니다.
# 자, 이제 `agent` 객체를 이용하여 `CartPole-v0` 환경과 상호작용을 통해 게임을 배우도록 하겠습니다.

env = gym.make('CartPole-v0')   # 게임 환경 변수
agent = DQNAgent()              # agent 객체 생성
score_history = []


# ## 학습 시작

for e in range(1, EPISODES+1):
    state = env.reset()                                         # 게임을 시작할 때마다 초기화된 상태를 불러온다.
    steps = 0
    while True:                                                 # 게임이 끝날 때까지 무한 반복
        env.render()                                            # 게임화면 띄우
        state = torch.FloatTensor([state])                      # 현재 상태를 텐서로 만듦
        action = agent.act(state)                               # act 에 입력하여 현재 할 행동을 return
        next_state, reward, done, _ = env.step(action.item())   # 행동을 step에 입력하여 행동에 따른 다음 상태, 보상, 게임종료여부 return.

        # 막대가 쓰러져 게임이 끝났을 경우 마이너스 보상주기 (처벌)
        if done:
            reward = -1

        agent.memorize(state, action, reward, next_state)       # 결과를 기억.
        agent.learn()                                           # 기억된 결과들로 학습.

        state = next_state
        steps += 1

        if done:
            print("에피소드:{0} 점수: {1}".format(e, steps))     # 막대가 쓰러지거나 최대점수(200) 도달하면 게임이 종료됨.
            score_history.append(steps)
            break


plt.plot(score_history)
plt.ylabel('score')
plt.show()

