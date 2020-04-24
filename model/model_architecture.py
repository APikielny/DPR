import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time

from defineHourglass_512_gray_skip import HourglassNet, lightingNet
from loss import L1
from PIL import Image, ImageOps
from light import read
from torchvision import transforms

EPOCHS = 1
BATCH_SIZE = 1

class ImagePair:
    def __init__(self, I_s, I_t, L_s, L_t):
        self.I_s = I_s
        self.I_t = I_t
        self.L_s = L_s
        self.L_t = L_t

# 27,000 length array of ImagePair
def load_data():
    # img_s = 'data/imgHQ00000/imgHQ00000_00.jpg'
    # img_t = 'data/imgHQ00000/imgHQ00000_01.jpg'
    # l_s = read('data/imgHQ00000/imgHQ00000_light_01.txt')
    # l_t = read('data/imgHQ00000/imgHQ00000_light_01.txt')
    #
    # img_s = Image.open(img_s)
    # img_s = transforms.ToTensor()(img_s).unsqueeze(0)
    # img_t = Image.open(img_t).convert('LA')
    # img_t = transforms.ToTensor()(img_t).unsqueeze(0)
    # print(img_s.shape)
    # print(img_t.shape)
    img_s = torch.zeros((3, 1, 128, 128))
    img_t = torch.zeros((3, 1, 128, 128))
    l_s = torch.zeros((128, 27, 8, 8))
    l_t = torch.zeros((128, 27, 8, 8))

    return [ImagePair(img_s, img_t, l_s, l_t)]



def train(model, optimizer, data):

    num_batches = len(data) // BATCH_SIZE

    for i in range(num_batches):
        total_loss = 0
        for j in range(i*BATCH_SIZE, min(i*BATCH_SIZE + BATCH_SIZE, len(data))):
            I_s = data[j].I_s
            I_t = data[j].I_t
            L_s = data[j].L_s
            L_t = data[j].L_t

            I_tp, L_sp = model.forward(I_s, L_t, 4)
            N = I_s.size ** 2
            loss = L1(N, I_t, I_tp, L_s, L_sp)
            total_loss += loss

        total_loss /= BATCH_SIZE
        print(total_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


model = HourglassNet(gray=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #not sure if we should be using Adam or another optimizer

for i in range(EPOCHS):
    data = load_data()
    train(model, optimizer, data)

#Save model
#https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
#Not sure if this^ is the best way, I think we should make sure to match the way we save if we want to run their test code