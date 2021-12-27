import torch

a = [1, 2, 3]
b = [2, 3, 4]
c = [4, 5, 6]

for r, h, g in (b, a, c):
    print(r, h, g)
    break

for r, b, g in zip(b, a, c):
    print(r, b, g)
    break