import torch.nn as nn
import torch
import numpy as np

BATCH_SIZE = 100

#return average L1 loss across a batch
def L1_batch(num_pixels, target_images, train_images, target_light, train_light):
    # assuming length of each list = BATCH_SIZE
    assert(target_images.size() == train_images.size())

    assert(target_light.size() == train_light.size())

    assert(target_images.size() == target_light.size())

    # calculate sum loss then average

    subtracted = train_images - target_images

    #TODO add gradient here but I'm not sure how
    term1 = (1/num_pixels) * torch.norm(subtracted)

    subtracted2 = train_light - target_light

    term2 = subtracted2 ** 2

    sum = term1.add(term2)

    return torch.mean(sum)
