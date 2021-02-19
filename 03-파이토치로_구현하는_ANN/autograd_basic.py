import torch

w = torch.tensor(1.0, requires_grad=True)   # requires_grad=True 하면 w.grad 에 gradient 가 자동저장된다.
                                            # element 가 float 이어야 한다.

a = w*3
l = a**2
l.backward()                                # backward 하는 객체는 scalar 이어야 한다.
print(w.grad)
print('l을 w로 미분한 값은 {}'.format(w.grad))

# requires_grad=True 를 하기 위해선 data type 이 float 이어야 한다.
x = torch.tensor(([[1, 2, 3], [4, 5, 6]]), dtype=torch.float, requires_grad=True)
y = torch.tensor(([[1],[2],[3]]), dtype=torch.float, requires_grad=True)

z = torch.mm(x,y)
t = torch.sum(z)

t.backward()

print('x grad : \n', x.grad)
print('y grad : \n', y.grad)

# affine 자동 미분 아주 편리하군..