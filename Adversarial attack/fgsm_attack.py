# # 8.1 FGSM 공격
# 정상 이미지에 노이즈를 더해 머신러닝 모델을 헷갈리게 하는 이미지가
# 바로 적대적 예제(Adversarial Example) 입니다.
# 이 프로젝트에선 Fast Gradient Sign Method, 즉 줄여서 FGSM이라는 방식으로
# 적대적 예제를 생성해 미리 학습이 완료된 딥러닝 모델을 공격해보도록 하겠습니다.

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import json


import matplotlib.pyplot as plt

# current working directory 를 현재 file 이 위치한 path 로.
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 


# ## 학습된 모델 불러오기

model = models.resnet101(pretrained=True)
model.eval()
print(model)


# ## 데이터셋 불러오기

CLASSES = json.load(open('./imagenet_samples/imagenet_classes.json'))
idx2class = [CLASSES[str(i)] for i in range(1000)]


# ## 이미지 불러오기

# 이미지 불러오기 (실제 공격은 학습용 데이터에 존재하지 않는 이미지로 가해질 것이므로 새로운 이미지 준비.)
img = Image.open('imagenet_samples/corgie.jpg') # 해당 이미지는 웰시코기 사진으로 데이터 셋에는 펨브로크 웰시코기라는 클래스가 존재.


# 이미지를 텐서로 변환하기
img_transforms = transforms.Compose([
    transforms.Resize((224, 224), Image.BICUBIC),   # imagenet data 와 크기 동일하게. Image.BICUBIC 은 보간 방법. later.. If need
    transforms.ToTensor(),
])

img_tensor = img_transforms(img)
img_tensor = img_tensor.unsqueeze(0)                # batch dimention 추가. [3, 244, 244] -> [1, 3, 244, 244]

print("이미지 텐서 모양:", img_tensor.size())


# 시각화를 위해 넘파이 행렬 변환
original_img_view = img_tensor.squeeze(0).detach()  # [1, 3, 244, 244] -> [3, 244, 244]
                                                    # detach() : 기존 Tensor에서 gradient 전파가 안되는 텐서 생성
                                                    # 단, address 공유하므로 둘중 하나가 변경되면 나머지 하나도 변경된다.

original_img_view = original_img_view.transpose(0,2).transpose(0,1).numpy()

# 텐서 시각화
plt.imshow(original_img_view)


# ## 공격 전 성능 확인하기
output = model(img_tensor)
prediction = output.max(1, keepdim=False)[1]

prediction_idx = prediction.item()
prediction_name = idx2class[prediction_idx]

print("예측된 레이블 번호:", prediction_idx)
print("레이블 이름:", prediction_name)

# ## FGSM 공격 함수 정의

def fgsm_attack(image, epsilon, gradient):
    # 기울기값의 원소의 sign 값을 구함 (음수 -> -1 / 0 -> 0 / 양수 -> 1)
    sign_gradient = gradient.sign()
    # 이미지 각 픽셀의 값을 sign_gradient 방향으로 epsilon 만큼 조절 (학습 때 가중치를 update 하는 방향과 반대 방향)
    perturbed_image = image + epsilon * sign_gradient
    # [0,1] 범위를 벗어나는 값을 조절 (벗어나는 값은 경계값으로 설정)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# ## 적대적 예제 생성

# 이미지의 기울기값을 구하도록 설정 (가중치의 gradient 가 아니라 이미지의 gradient 이다. 따라서 이미지와 크기가 같은 gradient가 생성된다.)
img_tensor.requires_grad_(True)

# 이미지를 모델에 통과시킴
output = model(img_tensor)


# 오차값 구하기 (레이블 263은 웰시코기)
loss = F.nll_loss(output, torch.tensor([263])) # nll_loss : logSoftmax 에 대한 cross entropy loss

# 기울기값 구하기
model.zero_grad()   # optimizer.zero_grad() 와 동일하다. gradient 를 구하기 위해 사전에 model 내 모든 gradient 0 으로 초기화.
loss.backward()


# 이미지의 기울기값을 추출
gradient = img_tensor.grad.data

# FGSM 공격으로 적대적 예제 생성
epsilon = 0.03 # 임의로 변경하여 노이즈를 변경할 수 있다. 값이 클 수록 노이즈가 커져 사람 눈에도 잘 보인다. ( 0 ~ 1 값으로 변경해보자. )
perturbed_data = fgsm_attack(img_tensor, epsilon, gradient)

# 생성된 적대적 예제를 모델에 통과시킴
output = model(perturbed_data)


# ## 적대적 예제 성능 확인

perturbed_prediction = output.max(1, keepdim=True)[1]

perturbed_prediction_idx = perturbed_prediction.item()
perturbed_prediction_name = idx2class[perturbed_prediction_idx]

print("예측된 레이블 번호:", perturbed_prediction_idx)
print("레이블 이름:", perturbed_prediction_name)


# 시각화를 위해 넘파이 행렬 변환
perturbed_data_view = perturbed_data.squeeze(0).detach()
perturbed_data_view = perturbed_data_view.transpose(0,2).transpose(0,1).numpy()

plt.imshow(perturbed_data_view)


# ## 원본과 적대적 예제 비교

f, a = plt.subplots(1, 2, figsize=(10, 10))

# 원본
a[0].set_title(prediction_name)
a[0].imshow(original_img_view)

# 적대적 예제
a[1].set_title(perturbed_prediction_name)
a[1].imshow(perturbed_data_view)

plt.show()

