import torch

w = torch.randn(5,3, dtype=torch.float)
x = torch.tensor([[1.0,2.0], [3.0,4.0], [5.0,6.0]])
print("w size:", w.size())
print("x size:", x.size())
print("w:", w)
print("x:", x)
print()


b = torch.randn(5,2, dtype=torch.float)
print("b:", b.size())
print("b:", b)
print()


wx = torch.mm(w,x) # (5,3) * (3,2) = (5,2)
print("wx size:", wx.size())
print("wx:", wx)
print()


result = wx + b    # broadcast
print("result size:", result.size()) 
print("result:", result) 
print()

