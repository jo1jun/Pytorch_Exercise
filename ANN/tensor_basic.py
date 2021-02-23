import torch
# numpy 와 사용법이 매우 유사.

x = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print(x)
print("Size:", x.size())
print("Shape:", x.shape)
print("랭크(차원):", x.ndimension())
print()


x = torch.unsqueeze(x, 0)
print(x)
print("Size:", x.size())
print("Shape:", x.shape)
print("랭크(차원):", x.ndimension())
print()


x = torch.squeeze(x)
print(x)
print("Size:", x.size())
print("Shape:", x.shape)
print("랭크(차원):", x.ndimension())
print()


# 랭크의 형태 바꾸기 (numpy 의 reshape 와 매우 유사)
x = x.view(9)
print(x)
print("Size:", x.size())
print("Shape:", x.shape)
print("랭크(차원):", x.ndimension())
print()


try:
    x = x.view(2,4) # tensor 모양 변화만 가능하다. 원소 값, 수 변경은 error.
except Exception as e:
    print(e)

