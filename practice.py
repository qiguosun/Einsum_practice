import numpy as np
import torch
# Matrix trace
a = torch.randn(4, 4)
print(a)
aTrace = torch.einsum('ii', a)
print(aTrace)

# diagonal
aDiag = torch.einsum('ii->i', a)
print(aDiag)

# outer product
x = torch.randn(5)
y = torch.randn(4)
xy = torch.einsum('i,j->ij', x, y)
print("outer product of x and y", xy.shape)

# batch matrix multiplication
As = torch.randn(3, 2, 5)
Bs = torch.randn(3, 5, 4)
ABs = torch.einsum("bij,bjk->bik", As, Bs)
print('Batch matrix multiplication As x Bs, where As shape is {},Bs shape is {}, output size{}'.format(
    As.size(), Bs.size(), ABs.size()))

# batch permute
A = torch.randn(2, 3, 4, 5)
print("permute", torch.einsum('...ij->...ji', A).shape)

# torch.nn.functional.bilinear
A = torch.randn(3, 5, 4)
l = torch.randn(2, 5)
r = torch.randn(2, 4)
print(torch.einsum('bn,anm,bm->ba', l, A, r))
