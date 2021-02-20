import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
import torch
import pickle
import matplotlib.pyplot as plt

# weird_function('원본') = '오염'
# weird_function('랜덤') = '가설'
# weird_function(x1) = weird_function(x2) -> x1 = x2 를 가정하는 듯.
# '오염' 과 '가설' 사이의 오차를 줄이며 '랜덤'을 갱신하면 '랜덤'과 '원본'이 같아져있을 것이다.
# 책에서 오차를 '가설'과 오염되기 전의 이미지 사이의 거리라고 되어있는데
# 오차는 '가설'과 오염된 후의 이미지 사이의 거리이다.


shp_original_img = (100, 100)
broken_image =  torch.FloatTensor( pickle.load(open('./broken_image_t.p', 'rb'),encoding='latin1' ) )

plt.figure()
plt.imshow(broken_image.view(100,100)) 


# weird_function 내부를 알 필요는 없음. 목적은 복원.
def weird_function(x, n_iter=5):
    h = x    
    filt = torch.tensor([-1./3, 1./3, -1./3])
    for i in range(n_iter):
        zero_tensor = torch.tensor([1.0*0])
        h_l = torch.cat( (zero_tensor, h[:-1]), 0)
        h_r = torch.cat((h[1:], zero_tensor), 0 )
        h = filt[0] * h + filt[2] * h_l + filt[1] * h_r
        if i % 2 == 0:
            h = torch.cat( (h[h.shape[0]//2:],h[:h.shape[0]//2]), 0  )
    return h

# 가설과 오염된 이미지 사이의 거리를 계산.
def distance_loss(hypothesis, broken_image):    
    return torch.dist(hypothesis, broken_image)         # default 로 2차 norm distance return.


random_tensor = torch.randn(10000, dtype = torch.float)


lr = 0.8
for i in range(0,20000):
    random_tensor.requires_grad_(True)                  # random_tensor.grad 에 gradient 저장.
    hypothesis = weird_function(random_tensor)
    loss = distance_loss(hypothesis, broken_image)
    loss.backward()
    with torch.no_grad():                               # 계산그래프를 생성하지 않고 값을 수동으로 갱신하기 위해 자동 기울기 계산 비활성화
        random_tensor = random_tensor - lr*random_tensor.grad   # 수동으로 gradient descent 구현
    if i % 1000 == 0:
        print('Loss at {} = {}'.format(i, loss.item()))

plt.figure()
plt.imshow(random_tensor.view(100,100).data)            # random_tensor 가 original_image 와 같아졌을 것이다.

