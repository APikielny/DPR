import torch

def read(path):
    f = open(path)
    g = f.readline()
    x = []
    while(len(g) > 0):
        trimmed = g.splitlines()[0]
        # print(g)
        x.append(trimmed)
        g = f.readline()

    return torch.tensor(x).squeeze((3,3))



