import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time

from defineHourglass_512_gray_skip import HourglassNet, lightingNet
from loss import L1

EPOCHS = 1


def train(self, model, optimizer):
    #1. batch data

    #2. zero optimizer
    optimizer.zero_grad()

    #3. forward pass
    #hourglassNet forward pass requires: (self, x, target_light, skip_count)
    #for skip count i think it depends on the epoch? ¯\_(ツ)_/¯
    output = model.forward(data, more_data, skip_count)

    #4. loss
    #calculate num pixels for loss:
    N = data[0].size ** 2

    loss = L1(N, data, data, data, data)


model = HourglassNet()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #not sure if we should be using Adam or another optimizer

 for i in range(EPOCHS):
     train(model)