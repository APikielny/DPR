import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time

from .defineHourglass_512_gray_skip import HourglassNet, lightingNet
from .loss import L1

EPOCHS = 1
BATCH_SIZE = 100

class ImagePair:
    def __init__(self, I_s, I_t, L_s, L_t):
        self.I_s = I_s
        self.I_t = I_t
        self.L_s = L_s
        self.L_t = L_t

# 27,000 length array of ImagePair
def load_data():
    return None


def train(model, optimizer, data):

    num_batches = data.size // BATCH_SIZE

    for i in range(num_batches):
        total_loss = 0
        for j in range(i*BATCH_SIZE, min(i*BATCH_SIZE + BATCH_SIZE, data.size)):
            I_s = data[j].I_s
            I_t = data[j].I_t
            L_s = data[j].L_s
            L_t = data[j].L_t

            I_tp, L_sp = model.forward(I_s, L_t, skip_count)
            N = I_s.size ** 2
            loss = L1(N, I_t, I_tp, L_s, L_sp)
            total_loss += loss

        total_loss /= BATCH_SIZE

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


model = HourglassNet()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #not sure if we should be using Adam or another optimizer

for i in range(EPOCHS):
    data = load_data()
    train(model, optimizer, data)

#Save model
#https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
#Not sure if this^ is the best way, I think we should make sure to match the way we save if we want to run their test code